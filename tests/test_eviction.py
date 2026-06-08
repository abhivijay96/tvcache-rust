#!/usr/bin/env python3

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from cache_client import AsyncPersistentCacheClient


HOST = "127.0.0.1"
PORT = 7878
TIMEOUT_SECONDS = 30
RETRIES = 5
RETRY_BACKOFF_SECONDS = 0.1

HIGH_WATERMARK_BYTES = 2 * 1024 * 1024
NUM_TASKS = 24
NODES_PER_TASK = 8
PAYLOAD_KB = 16
HOT_TASKS = 6
FIND_ROUNDS = 80
EVICTION_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class InsertedNode:
    path: list[str]
    value: str


@dataclass(frozen=True)
class TaskData:
    task_id: str
    nodes: list[InsertedNode]


def make_payload(task_idx: int, node_idx: int) -> str:
    prefix = f"payload task={task_idx} node={node_idx}\n"
    target_size = PAYLOAD_KB * 1024
    return prefix + ("x" * (target_size - len(prefix)))


def count_nodes(node: dict[str, Any]) -> int:
    children = node.get("children", {})
    return 1 + sum(count_nodes(child) for child in children.values())


def prefixed_tcgs(cache_dump: object, task_prefix: str) -> dict[str, dict[str, Any]]:
    if not isinstance(cache_dump, dict):
        raise TypeError(f"expected dump_all to return dict, got {type(cache_dump)}")

    return {
        task_id: tcg
        for task_id, tcg in cache_dump.items()
        if task_id.startswith(task_prefix)
    }


def total_size_in_bytes(cache_dump: object, task_prefix: str) -> int:
    tcgs = prefixed_tcgs(cache_dump, task_prefix)
    return sum(tcg.get("size_in_bytes", 0) for tcg in tcgs.values())


def task_node_count(cache_dump: object, task_id: str) -> int:
    if not isinstance(cache_dump, dict) or task_id not in cache_dump:
        return 0

    root = cache_dump[task_id].get("root")
    if root is None:
        return 0

    return max(0, count_nodes(root) - 1)


def get_node(cache_dump: object, task_id: str, path: list[str]) -> dict[str, Any] | None:
    if not isinstance(cache_dump, dict) or task_id not in cache_dump:
        return None

    curr = cache_dump[task_id].get("root")
    if not isinstance(curr, dict):
        return None

    for path_entry in path:
        children = curr.get("children", {})
        if path_entry not in children:
            return None
        curr = children[path_entry]

    return curr


def prefixed_node_count(cache_dump: object, task_prefix: str) -> int:
    return sum(
        task_node_count(cache_dump, task_id)
        for task_id in prefixed_tcgs(cache_dump, task_prefix)
    )


async def dump_all(client: AsyncPersistentCacheClient) -> object:
    response = await client.dump_all_response()
    if response["type"] == "error":
        raise RuntimeError(response["message"])
    if response["type"] != "json":
        raise RuntimeError(f"expected json response, got {response['type']}")
    return json.loads(response["json"])


async def insert_task(
    client: AsyncPersistentCacheClient,
    task_prefix: str,
    task_idx: int,
) -> TaskData:
    task_id = f"{task_prefix}-task-{task_idx:02d}"
    root = f"root-{task_idx:02d}"
    nodes = []

    for node_idx in range(NODES_PER_TASK):
        if node_idx == 0:
            path = [root]
        else:
            path = [root, f"node-{node_idx:02d}"]

        value = make_payload(task_idx, node_idx)
        sandbox_id = f"{task_id}-sandbox-{node_idx:02d}"
        response = await client.insert(task_id, path, value, sandbox_id, runtime=1)

        expected = {
            "type": "insert_result",
            "value_inserted": True,
            "sandbox_inserted": True,
        }
        if response != expected:
            raise AssertionError(
                f"unexpected insert response for {task_id} path={path}: "
                f"expected={expected} actual={response}"
            )

        nodes.append(InsertedNode(path, value))

    return TaskData(task_id, nodes)


async def insert_task_range(
    client: AsyncPersistentCacheClient,
    task_prefix: str,
    start_idx: int,
    end_idx: int,
) -> list[TaskData]:
    task_data = []

    for task_idx in range(start_idx, end_idx):
        task_data.append(await insert_task(client, task_prefix, task_idx))

    return task_data


async def warm_hot_tasks(
    client: AsyncPersistentCacheClient,
    task_data: list[TaskData],
) -> set[str]:
    hot_tasks = {task.task_id for task in task_data[:HOT_TASKS]}

    for task in task_data[:HOT_TASKS]:
        for _ in range(FIND_ROUNDS):
            for node_idx, node in enumerate(task.nodes):
                response = await client.find(task.task_id, node.path)
                expected = {
                    "type": "find_result",
                    "value": node.value,
                    "sandbox_path": node.path,
                    "sandbox_id": f"{task.task_id}-sandbox-{node_idx:02d}",
                }
                if response != expected:
                    raise AssertionError(
                        f"unexpected find response for {task.task_id} path={node.path}: "
                        f"expected={expected} actual={response}"
                    )

    return hot_tasks


async def wait_for_eviction(
    client: AsyncPersistentCacheClient,
    task_prefix: str,
    before_size: int,
) -> object:
    deadline = time.monotonic() + EVICTION_TIMEOUT_SECONDS
    latest_dump = await dump_all(client)

    while time.monotonic() < deadline:
        await asyncio.sleep(1)
        latest_dump = await dump_all(client)

        if total_size_in_bytes(latest_dump, task_prefix) < before_size:
            return latest_dump

    return latest_dump


def print_retention(
    cache_dump: object,
    task_data: list[TaskData],
    hot_tasks: set[str],
) -> tuple[float, float]:
    hot_before = sum(len(task.nodes) for task in task_data if task.task_id in hot_tasks)
    cold_before = sum(len(task.nodes) for task in task_data if task.task_id not in hot_tasks)
    hot_after = sum(
        task_node_count(cache_dump, task.task_id)
        for task in task_data
        if task.task_id in hot_tasks
    )
    cold_after = sum(
        task_node_count(cache_dump, task.task_id)
        for task in task_data
        if task.task_id not in hot_tasks
    )

    hot_retention = hot_after / hot_before
    cold_retention = cold_after / cold_before

    print(f"Hot nodes retained: {hot_after}/{hot_before} ({hot_retention:.2%})")
    print(f"Cold nodes retained: {cold_after}/{cold_before} ({cold_retention:.2%})")

    return hot_retention, cold_retention


async def verify_hot_cache_hits_after_eviction(
    client: AsyncPersistentCacheClient,
    task_data: list[TaskData],
    hot_tasks: set[str],
    after_dump: object,
) -> None:
    for task in task_data:
        if task.task_id not in hot_tasks:
            continue

        for node_idx, node in enumerate(task.nodes):
            before_node = get_node(after_dump, task.task_id, node.path)
            if before_node is None:
                raise AssertionError(
                    f"expected hot path to survive eviction: "
                    f"task_id={task.task_id} path={node.path}"
                )

            before_hits = before_node.get("number_of_cache_hits", 0)
            response = await client.find(task.task_id, node.path)
            expected = {
                "type": "find_result",
                "value": node.value,
                "sandbox_path": node.path,
                "sandbox_id": f"{task.task_id}-sandbox-{node_idx:02d}",
            }
            if response != expected:
                raise AssertionError(
                    f"expected post-eviction hot path find to be an exact cache hit: "
                    f"task_id={task.task_id} path={node.path} "
                    f"expected={expected} actual={response}"
                )

            latest_dump = await dump_all(client)
            after_node = get_node(latest_dump, task.task_id, node.path)
            if after_node is None:
                raise AssertionError(
                    f"hot path disappeared after post-eviction find: "
                    f"task_id={task.task_id} path={node.path}"
                )

            after_hits = after_node.get("number_of_cache_hits", 0)
            if after_hits != before_hits + 1:
                raise AssertionError(
                    f"expected post-eviction find to increment cache hits by 1: "
                    f"task_id={task.task_id} path={node.path} "
                    f"before={before_hits} after={after_hits}"
                )


async def run_test() -> None:
    task_prefix = f"eviction-{int(time.time())}"

    async with AsyncPersistentCacheClient(
        HOST,
        PORT,
        TIMEOUT_SECONDS,
        RETRIES,
        RETRY_BACKOFF_SECONDS,
    ) as client:
        print(f"Creating eviction dataset with prefix={task_prefix}")
        hot_task_data = await insert_task_range(client, task_prefix, 0, HOT_TASKS)
        print(f"Warming {HOT_TASKS} hot tasks before adding cold bulk")
        hot_tasks = await warm_hot_tasks(client, hot_task_data)

        cold_task_data = await insert_task_range(
            client,
            task_prefix,
            HOT_TASKS,
            NUM_TASKS,
        )
        task_data = hot_task_data + cold_task_data

        expected_inserted_nodes = NUM_TASKS * NODES_PER_TASK
        before_dump = await dump_all(client)
        before_size = total_size_in_bytes(before_dump, task_prefix)
        before_nodes = prefixed_node_count(before_dump, task_prefix)
        print(f"Test-owned cache size before eviction wait: {before_size} bytes")
        print(f"Test-owned nodes before eviction wait: {before_nodes}")

        if before_size > HIGH_WATERMARK_BYTES:
            after_dump = await wait_for_eviction(client, task_prefix, before_size)
            after_size = total_size_in_bytes(after_dump, task_prefix)
            print(f"Test-owned cache size after eviction wait: {after_size} bytes")

            if after_size >= before_size:
                raise AssertionError(
                    f"expected eviction to reduce cache size below {before_size}, got {after_size}"
                )
        else:
            after_dump = before_dump
            if before_nodes >= expected_inserted_nodes:
                raise AssertionError(
                    "dataset did not exceed the configured 2 MB high watermark, "
                    "and eviction had not already removed any inserted nodes"
                )

            print("Eviction already ran before the first post-insert dump")

        hot_retention, cold_retention = print_retention(after_dump, task_data, hot_tasks)

        if hot_retention <= 0:
            raise AssertionError("expected at least one hot node to remain after eviction")

        if hot_retention < cold_retention:
            raise AssertionError(
                "expected hot TCGs to be retained at least as well as cold TCGs: "
                f"hot={hot_retention:.2%} cold={cold_retention:.2%}"
            )

        await verify_hot_cache_hits_after_eviction(
            client,
            task_data,
            hot_tasks,
            after_dump,
        )

    print("Eviction policy test passed")


def main() -> None:
    asyncio.run(run_test())


if __name__ == "__main__":
    main()

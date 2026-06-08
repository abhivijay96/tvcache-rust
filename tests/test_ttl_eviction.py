#!/usr/bin/env python3

import asyncio
import json
import time
from typing import Any

from cache_client import AsyncPersistentCacheClient


HOST = "127.0.0.1"
PORT = 7878
TIMEOUT_SECONDS = 30
RETRIES = 5
RETRY_BACKOFF_SECONDS = 0.1

EXPIRING_TTL_SECONDS = 1
EVICTION_TIMEOUT_SECONDS = 8
EXPIRING_SUBTREE_PATHS = [
    ["root", "expires", "child-a"],
    ["root", "expires", "child-b"],
    ["root", "expires", "child-a", "grandchild-a"],
    ["root", "expires", "child-a", "grandchild-b"],
]


async def dump_all(client: AsyncPersistentCacheClient) -> object:
    response = await client.dump_all_response()
    if response["type"] == "error":
        raise RuntimeError(response["message"])
    if response["type"] != "json":
        raise RuntimeError(f"expected json response, got {response['type']}")
    return json.loads(response["json"])


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


async def insert_node(
    client: AsyncPersistentCacheClient,
    task_id: str,
    path: list[str],
    value: str,
    sandbox_id: str,
    ttl: int | None,
) -> None:
    response = await client.insert(task_id, path, value, sandbox_id, runtime=1, ttl=ttl)
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


async def wait_for_node_removal(
    client: AsyncPersistentCacheClient,
    task_id: str,
    path: list[str],
) -> object:
    deadline = time.monotonic() + EVICTION_TIMEOUT_SECONDS
    latest_dump = await dump_all(client)

    while time.monotonic() < deadline:
        await asyncio.sleep(1)
        latest_dump = await dump_all(client)
        if get_node(latest_dump, task_id, path) is None:
            return latest_dump

    return latest_dump


async def run_test() -> None:
    task_id = f"ttl-{int(time.time())}"
    root_path = ["root"]
    expiring_path = ["root", "expires"]
    persistent_path = ["root", "persists"]

    root_value = "root-value"
    expiring_value = "expiring-value"
    persistent_value = "persistent-value"

    async with AsyncPersistentCacheClient(
        HOST,
        PORT,
        TIMEOUT_SECONDS,
        RETRIES,
        RETRY_BACKOFF_SECONDS,
    ) as client:
        await insert_node(client, task_id, root_path, root_value, "root-sandbox", ttl=None)
        await insert_node(
            client,
            task_id,
            expiring_path,
            expiring_value,
            "expiring-sandbox",
            ttl=EXPIRING_TTL_SECONDS,
        )

        for idx, path in enumerate(EXPIRING_SUBTREE_PATHS):
            await insert_node(
                client,
                task_id,
                path,
                f"expiring-subtree-value-{idx}",
                f"expiring-subtree-sandbox-{idx}",
                ttl=None,
            )

        await insert_node(
            client,
            task_id,
            persistent_path,
            persistent_value,
            "persistent-sandbox",
            ttl=None,
        )

        before_dump = await dump_all(client)
        if get_node(before_dump, task_id, expiring_path) is None:
            raise AssertionError("expiring node was missing before its TTL elapsed")
        for path in EXPIRING_SUBTREE_PATHS:
            if get_node(before_dump, task_id, path) is None:
                raise AssertionError(
                    f"expiring subtree node was missing before TTL eviction: {path}"
                )
        if get_node(before_dump, task_id, persistent_path) is None:
            raise AssertionError("persistent node was missing before TTL eviction")

        after_dump = await wait_for_node_removal(client, task_id, expiring_path)

        if get_node(after_dump, task_id, expiring_path) is not None:
            raise AssertionError("expected TTL eviction to remove the expired node")

        for path in EXPIRING_SUBTREE_PATHS:
            if get_node(after_dump, task_id, path) is not None:
                raise AssertionError(
                    f"expected TTL eviction to remove expiring subtree node: {path}"
                )

        if get_node(after_dump, task_id, persistent_path) is None:
            raise AssertionError("TTL eviction removed the non-expiring sibling node")

        expiring_find = await client.find(task_id, expiring_path)
        expected_expiring_find = {
            "type": "find_result",
            "value": None,
            "sandbox_path": root_path,
            "sandbox_id": "root-sandbox",
        }
        if expiring_find != expected_expiring_find:
            raise AssertionError(
                "expected expired path to fall back to the root sandbox: "
                f"expected={expected_expiring_find} actual={expiring_find}"
            )

        persistent_find = await client.find(task_id, persistent_path)
        expected_persistent_find = {
            "type": "find_result",
            "value": persistent_value,
            "sandbox_path": persistent_path,
            "sandbox_id": "persistent-sandbox",
        }
        if persistent_find != expected_persistent_find:
            raise AssertionError(
                "expected persistent path to remain readable: "
                f"expected={expected_persistent_find} actual={persistent_find}"
            )

    print("TTL eviction test passed")


def main() -> None:
    asyncio.run(run_test())


if __name__ == "__main__":
    main()

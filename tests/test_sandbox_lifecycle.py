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

TTL_SECONDS = 1
TTL_DRAIN_TIMEOUT_SECONDS = 15


async def dump_all(client: AsyncPersistentCacheClient) -> dict[str, Any]:
    response = await client.dump_all_response()
    if response["type"] == "error":
        raise RuntimeError(response["message"])
    if response["type"] != "json":
        raise RuntimeError(f"expected json response, got {response['type']}")
    return json.loads(response["json"])


def get_node(cache_dump: dict[str, Any], task_id: str, path: list[str]) -> dict[str, Any] | None:
    if task_id not in cache_dump:
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


async def drain(client: AsyncPersistentCacheClient) -> list[str]:
    response = await client.drain_reclaimable_sandboxes()
    if response["type"] != "reclaimable_sandboxes":
        raise RuntimeError(f"expected reclaimable_sandboxes response, got {response['type']}")
    return [sandbox["sandbox_id"] for sandbox in response["sandboxes"]]


async def insert_node(
    client: AsyncPersistentCacheClient,
    task_id: str,
    path: list[str],
    value: str,
    sandbox_id: str | None,
    ttl: int | None = None,
) -> None:
    response = await client.insert(task_id, path, value, sandbox_id, runtime=1, ttl=ttl)
    expected = {
        "type": "insert_result",
        "value_inserted": True,
        "sandbox_inserted": sandbox_id is not None,
    }
    if response != expected:
        raise AssertionError(
            f"unexpected insert response for {task_id} path={path}: "
            f"expected={expected} actual={response}"
        )


async def run_test() -> None:
    task_id = f"lifecycle-{int(time.time())}"
    sandbox_id = f"{task_id}-sb1"
    ttl_sandbox_id = f"{task_id}-sb2"
    sandbox_path = ["a", "b"]
    find_path = ["a", "b", "c"]

    async with AsyncPersistentCacheClient(
        HOST,
        PORT,
        TIMEOUT_SECONDS,
        RETRIES,
        RETRY_BACKOFF_SECONDS,
    ) as client:
        # Insert a prefix node, then the node carrying the sandbox.
        await insert_node(client, task_id, ["a"], "a-value", sandbox_id=None)
        await insert_node(client, task_id, sandbox_path, "b-value", sandbox_id=sandbox_id)

        # Find via longest prefix match returns the sandbox and counts a reference.
        find_result = await client.find(task_id, find_path)
        expected_find = {
            "type": "find_result",
            "value": None,
            "sandbox_path": sandbox_path,
            "sandbox_id": sandbox_id,
        }
        if find_result != expected_find:
            raise AssertionError(
                f"unexpected find result: expected={expected_find} actual={find_result}"
            )

        cache_dump = await dump_all(client)
        tcg = cache_dump[task_id]
        if tcg["sandbox_references"].get(sandbox_id) != 1:
            raise AssertionError(
                f"expected one reference to {sandbox_id}: {tcg['sandbox_references']}"
            )

        node_before_mark = get_node(cache_dump, task_id, sandbox_path)
        assert node_before_mark is not None
        node_size_before_mark = node_before_mark["size_in_bytes"]
        tcg_size_before_mark = tcg["size_in_bytes"]
        sandboxes_before_mark = tcg["number_of_sandboxes"]

        # Mark the sandbox evicted: victim list gains the id, node loses it.
        mark_result = await client.mark_sandboxes_evicted(
            [
                {
                    "task_id": task_id,
                    "sandbox_path": sandbox_path,
                    "sandbox_id": sandbox_id,
                }
            ]
        )
        expected_mark = {
            "type": "mark_sandboxes_evicted_result",
            "sandbox_ids_recorded": 1,
        }
        if mark_result != expected_mark:
            raise AssertionError(
                f"unexpected mark result: expected={expected_mark} actual={mark_result}"
            )

        cache_dump = await dump_all(client)
        tcg = cache_dump[task_id]
        if sandbox_id not in tcg["reclaimable_sandboxes"]:
            raise AssertionError(
                f"expected {sandbox_id} in victim list: {tcg['reclaimable_sandboxes']}"
            )

        node_after_mark = get_node(cache_dump, task_id, sandbox_path)
        assert node_after_mark is not None
        if node_after_mark["sandbox_id"] is not None:
            raise AssertionError("expected the node's sandbox_id to be cleared by mark")
        if tcg["number_of_sandboxes"] != sandboxes_before_mark - 1:
            raise AssertionError("expected number_of_sandboxes to drop by one after mark")
        if node_after_mark["size_in_bytes"] != node_size_before_mark - len(sandbox_id):
            raise AssertionError("expected node size to shrink by the sandbox id length")
        if tcg["size_in_bytes"] != tcg_size_before_mark - len(sandbox_id):
            raise AssertionError("expected tcg size to shrink by the sandbox id length")

        # A marked sandbox is never handed out again.
        find_after_mark = await client.find(task_id, find_path)
        if find_after_mark["sandbox_id"] is not None:
            raise AssertionError(
                f"expected no sandbox after mark, got {find_after_mark['sandbox_id']}"
            )

        # The drain must not release the sandbox while a rollout still references it.
        drained = await drain(client)
        if sandbox_id in drained:
            raise AssertionError("drain returned a sandbox that is still referenced")

        unref_result = await client.unref_sandbox(task_id, sandbox_id)
        expected_unref = {
            "type": "unref_sandbox_result",
            "remaining_references": 0,
        }
        if unref_result != expected_unref:
            raise AssertionError(
                f"unexpected unref result: expected={expected_unref} actual={unref_result}"
            )

        # Now the drain hands over ownership exactly once.
        drained = await drain(client)
        if sandbox_id not in drained:
            raise AssertionError("drain did not return the unreferenced victim sandbox")

        drained = await drain(client)
        if sandbox_id in drained:
            raise AssertionError("drain returned the same sandbox twice")

        # TTL leg: internally evicted sandboxes flow through the same drain path.
        await insert_node(
            client,
            task_id,
            ["expiring"],
            "expiring-value",
            sandbox_id=ttl_sandbox_id,
            ttl=TTL_SECONDS,
        )

        deadline = time.monotonic() + TTL_DRAIN_TIMEOUT_SECONDS
        ttl_sandbox_drained = False

        while time.monotonic() < deadline:
            if ttl_sandbox_id in await drain(client):
                ttl_sandbox_drained = True
                break
            await asyncio.sleep(1)

        if not ttl_sandbox_drained:
            raise AssertionError(
                "TTL-evicted sandbox never showed up in drain_reclaimable_sandboxes"
            )

    print("sandbox lifecycle test passed")


def main() -> None:
    asyncio.run(run_test())


if __name__ == "__main__":
    main()

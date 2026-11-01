"""Graph contraction server regressions. Run the server before this script."""

import asyncio
import json
from uuid import uuid4

from cache_client import AsyncPersistentCacheClient, dump_node


async def test_new_nodes_are_red(client: AsyncPersistentCacheClient) -> None:
    task_id = f"contraction-colors-{uuid4().hex}"
    await client.insert(task_id, ["a"], "value", None, runtime=1)
    await client.insert(
        task_id, ["a", "b"], "nested", None, runtime=1, embedding=[1.0, 0.0]
    )
    await client.insert(task_id, ["c"], None, f"snapshot-{task_id}", runtime=1)

    response = await client.dump_all_response()
    assert response["type"] == "json", response
    dump = json.loads(response["json"])
    nodes = [dump_node(dump, task_id, path) for path in ([], ["a"], ["a", "b"], ["c"])]
    for node in nodes:
        assert node is not None and node["settled"] is False, f"expected a new red node, got {node}"


async def drained_ids(client: AsyncPersistentCacheClient) -> list[str]:
    response = await client.drain_reclaimable_sandboxes()
    return [sandbox["sandbox_id"] for sandbox in response["sandboxes"]]


async def test_lease_skips_hit_counts(client: AsyncPersistentCacheClient) -> None:
    task_id = f"contraction-lease-{uuid4().hex}"
    snapshot = f"snapshot-{task_id}"
    await client.insert(task_id, ["a"], "value", snapshot, runtime=1)
    await client.insert(task_id, ["a", "b"], "nested", None, runtime=1)

    expected = {"type": "lease_result", "sandbox_path": ["a"], "sandbox_id": snapshot}
    for path in (["a", "b"], ["a", "missing"]):
        response = await client.lease_sandbox(task_id, path)
        assert response == expected, response
    unknown_task_id = f"{task_id}-unknown"
    for lease_task_id, path in ((task_id, ["missing"]), (unknown_task_id, ["a"])):
        response = await client.lease_sandbox(lease_task_id, path)
        assert response["sandbox_path"] is None and response["sandbox_id"] is None, response

    dump = json.loads((await client.dump_all_response())["json"])
    assert unknown_task_id not in dump, "expected a lease not to create a graph"
    for path in (["a"], ["a", "b"]):
        node = dump_node(dump, task_id, path)
        assert node["number_of_cache_hits"] == node["number_of_lpm_hits"] == 0, node

    # The evicted snapshot stays out of drains until both leases are released.
    record = {"task_id": task_id, "sandbox_path": ["a"], "sandbox_id": snapshot}
    await client.mark_sandboxes_evicted([record])
    for remaining in (1, 0):
        assert snapshot not in await drained_ids(client)
        response = await client.unref_sandbox(task_id, snapshot)
        assert response["remaining_references"] == remaining, response
    assert snapshot in await drained_ids(client)


async def run_tests() -> None:
    async with AsyncPersistentCacheClient("127.0.0.1", 7878) as client:
        await test_new_nodes_are_red(client)
        await test_lease_skips_hit_counts(client)
    print("all contraction server tests passed")


if __name__ == "__main__":
    asyncio.run(run_tests())

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


async def run_tests() -> None:
    async with AsyncPersistentCacheClient("127.0.0.1", 7878) as client:
        await test_new_nodes_are_red(client)
    print("all contraction server tests passed")


if __name__ == "__main__":
    asyncio.run(run_tests())

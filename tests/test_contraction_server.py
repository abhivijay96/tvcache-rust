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


async def test_sibling_groups(client: AsyncPersistentCacheClient) -> None:
    task_id = f"contraction-siblings-{uuid4().hex}"
    for path, embedding, snapshot in (
        (["a"], None, None),
        (["b"], None, None),
        (["a", "x"], [1.0, 0.0], None),
        (["a", "y"], [0.0, 1.0], None),
        (["b", "z"], [1.0, 0.0], None),
        (["b", "missing"], None, None),
        (["b", "v"], [0.6, 0.8], f"snapshot-{task_id}"),
        (["b", "wide"], [1.0, 0.0, 0.0], None),
        (["b", "z", "w"], None, None),
    ):
        await client.insert(task_id, path, "value", snapshot, runtime=1, embedding=embedding)

    def expected(deeper_nodes_exist: bool, groups: list) -> dict:
        return {
            "type": "sibling_groups",
            "deeper_nodes_exist": deeper_nodes_exist,
            "groups": groups,
        }

    response = await client.list_sibling_groups(task_id, 1, 0.5)
    assert response == expected(True, []), response

    # Missing embeddings and mismatched dimensions never pair, and cousins x and z stay apart.
    response = await client.list_sibling_groups(task_id, 2, 0.5)
    assert response["deeper_nodes_exist"] and len(response["groups"]) == 1, response
    [group] = response["groups"]
    assert group["parent_path"] == ["b"], group
    assert group["children"] == [
        {"calls": ["z"], "settled": False, "has_snapshot": False},
        {"calls": ["v"], "settled": False, "has_snapshot": True},
    ], group
    assert len(group["pairs"]) == 1, group
    [(i, j, score)] = group["pairs"]
    assert (i, j) == (0, 1) and abs(score - 0.6) < 1e-6, group

    for task, depth, threshold, deeper in (
        (task_id, 2, 0.61, True),
        (task_id, 3, 0.0, False),
        (task_id, 4, 0.0, False),
        (f"{task_id}-unknown", 1, 0.0, False),
    ):
        response = await client.list_sibling_groups(task, depth, threshold)
        assert response == expected(deeper, []), response


async def test_apply_contraction_smoke(client: AsyncPersistentCacheClient) -> None:
    task_id = f"contraction-apply-{uuid4().hex}"
    for path in (["p"], ["p", "a"], ["p", "b"]):
        await client.insert(task_id, path, f"value-{path[-1]}", None, runtime=1)

    classes = [["a", "b", "missing", "gone"]]
    response = await client.apply_contraction(task_id, ["p"], classes, ["a"])
    assert response == {"type": "contraction_result", "merged_classes": 1, "skipped_calls": 2}

    response = await client.find(task_id, ["p", "b"])
    assert response["value"] == "value-b", response
    dump = json.loads((await client.dump_all_response())["json"])
    [child] = dump_node(dump, task_id, ["p"])["children"]
    assert set(child["tool_calls"]) == {"a", "b"} and child["settled"] is True, child

    unknown_task_id = f"{task_id}-unknown"
    response = await client.apply_contraction(unknown_task_id, [], [["a", "b"]], ["a"])
    assert response == {"type": "contraction_result", "merged_classes": 0, "skipped_calls": 2}
    dump = json.loads((await client.dump_all_response())["json"])
    assert unknown_task_id not in dump, "expected an apply not to create a graph"


async def checked_dump(client: AsyncPersistentCacheClient, task_id: str) -> dict:
    """Returns the dump after checking the task's counters against its nodes."""
    dump = json.loads((await client.dump_all_response())["json"])
    graph = dump[task_id]
    nodes, stack = [], list(graph["root"]["children"])
    while stack:
        nodes.append(stack.pop())
        stack.extend(nodes[-1]["children"])
    counted = (
        len(nodes),
        sum(node["sandbox_id"] is not None for node in nodes),
        sum(node["size_in_bytes"] for node in nodes),
    )
    stored = (graph["number_of_nodes"], graph["number_of_sandboxes"], graph["size_in_bytes"])
    assert counted == stored, (counted, stored)
    return dump


async def apply(
    client: AsyncPersistentCacheClient,
    task_id: str,
    parent_path: list[str],
    classes: list[list[str]],
    settled: list[str],
    expected: tuple[int, int],
) -> dict:
    """Applies a contraction, checks its counts and returns the checked dump."""
    response = await client.apply_contraction(task_id, parent_path, classes, settled)
    assert (response["merged_classes"], response["skipped_calls"]) == expected, response
    return await checked_dump(client, task_id)


def child_calls(node: dict) -> list[list[str]]:
    return [sorted(child["tool_calls"]) for child in node["children"]]


async def test_apply_merges_recursively(client: AsyncPersistentCacheClient) -> None:
    task_id = f"contraction-merge-{uuid4().hex}"
    kept, dropped = f"kept-{task_id}", f"dropped-{task_id}"
    for path, snapshot in (
        (["a"], kept),
        (["b"], dropped),
        (["a", "x"], None),
        (["a", "w"], None),
        (["b", "x"], None),
        (["b", "y"], None),
    ):
        await client.insert(task_id, path, "-".join(path), snapshot, runtime=1)
    await apply(client, task_id, [], [], ["a", "b"], (0, 0))
    await apply(client, task_id, ["a"], [], ["x", "w"], (0, 0))
    await apply(client, task_id, ["b"], [], ["x", "y"], (0, 0))
    assert (await client.lease_sandbox(task_id, ["b"]))["sandbox_id"] == dropped

    dump = await apply(client, task_id, [], [["a", "b"]], [], (1, 0))
    [merged] = dump[task_id]["root"]["children"]
    assert sorted(merged["tool_calls"]) == ["a", "b"] and merged["sandbox_id"] == kept, merged
    assert sorted(child_calls(merged)) == [["w"], ["x"], ["y"]], merged
    # Merged and absorbed children turn red, and a lone survivor child stays blue.
    colors = ((["a"], False), (["b", "x"], False), (["b", "y"], False), (["a", "w"], True))
    for path, settled in colors:
        assert dump_node(dump, task_id, path)["settled"] is settled, path
    assert dump_node(dump, task_id, ["b", "x"])["value"] == "a-x"

    await client.insert(task_id, ["b"], "b-new", None, runtime=1)
    response = await client.find(task_id, ["b"])
    assert (response["value"], response["sandbox_id"]) == ("b-new", kept), response
    await client.unref_sandbox(task_id, kept)
    dump = await apply(client, task_id, [], [], ["b"], (0, 0))
    assert dump_node(dump, task_id, ["a"])["value"] == "a"
    assert dump_node(dump, task_id, ["a"])["settled"] is True

    # The absorbed snapshot drains only after its lease is released.
    assert dropped not in await drained_ids(client)
    assert (await client.unref_sandbox(task_id, dropped))["remaining_references"] == 0
    assert dropped in await drained_ids(client)


async def test_apply_colors_and_embeddings(client: AsyncPersistentCacheClient) -> None:
    task_id = f"contraction-embeddings-{uuid4().hex}"
    await client.insert(task_id, ["a"], "a", None, runtime=1, embedding=[1.0, 0.0])
    await client.insert(task_id, ["b"], "b", None, runtime=1)
    await apply(client, task_id, [], [], ["a"], (0, 0))
    dump = await apply(client, task_id, [], [["a", "b"]], [], (1, 0))
    assert dump_node(dump, task_id, ["a"])["settled"] is False, "expected a merge to turn red"

    # Only a previously missing embedding makes the settled node red.
    steps = (("b", None, True), ("a", [0.0, 1.0], True), ("b", [0.0, 1.0], False))
    for call, embedding, settled in steps:
        await apply(client, task_id, [], [], ["a", "b"], (0, 0))
        await client.insert(task_id, [call], None, None, runtime=1, embedding=embedding)
        dump = await checked_dump(client, task_id)
        assert dump_node(dump, task_id, [call])["settled"] is settled, (call, embedding)
    assert dump_node(dump, task_id, ["a"])["embedding"] == [1.0, 0.0]


async def test_apply_skips_evicted_member(client: AsyncPersistentCacheClient) -> None:
    task_id = f"contraction-evicted-{uuid4().hex}"
    for call, ttl in (("a", None), ("gone", 1), ("b", None)):
        await client.insert(task_id, [call], call, None, runtime=1, ttl=ttl)
    for _ in range(10):
        await asyncio.sleep(1)
        if dump_node(await checked_dump(client, task_id), task_id, ["gone"]) is None:
            break
    else:
        raise AssertionError("expected TTL eviction to remove gone")

    dump = await apply(client, task_id, [], [["a", "gone", "b"]], ["gone", "a", "b"], (1, 1))
    [child] = dump[task_id]["root"]["children"]
    assert sorted(child["tool_calls"]) == ["a", "b"] and child["settled"] is True, child


async def test_sibling_groups_after_apply(client: AsyncPersistentCacheClient) -> None:
    task_id = f"contraction-scores-{uuid4().hex}"
    for call, embedding in (("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [0.0, 1.0])):
        await client.insert(task_id, [call], call, None, runtime=1, embedding=embedding)
    await apply(client, task_id, [], [["a", "b"]], [], (1, 0))

    # The merged node scores by its best entry pair, not their mean.
    [group] = (await client.list_sibling_groups(task_id, 1, 0.9))["groups"]
    assert [sorted(child["calls"]) for child in group["children"]] == [["a", "b"], ["c"]]
    [(i, j, score)] = group["pairs"]
    assert (i, j) == (0, 1) and abs(score - 1.0) < 1e-6, group

    # Blue siblings never pair with each other, only with a red sibling.
    await apply(client, task_id, [], [], ["a", "c"], (0, 0))
    assert (await client.list_sibling_groups(task_id, 1, 0.9))["groups"] == []
    await client.insert(task_id, ["d"], "d", None, runtime=1, embedding=[1.0, 0.0])
    [group] = (await client.list_sibling_groups(task_id, 1, 0.9))["groups"]
    children = [(sorted(child["calls"]), child["settled"]) for child in group["children"]]
    assert children == [(["a", "b"], True), (["d"], False)], group
    assert [(i, j) for i, j, _ in group["pairs"]] == [(0, 1)], group


async def run_tests() -> None:
    async with AsyncPersistentCacheClient("127.0.0.1", 7878) as client:
        await test_new_nodes_are_red(client)
        await test_lease_skips_hit_counts(client)
        await test_sibling_groups(client)
        await test_apply_contraction_smoke(client)
        await test_apply_merges_recursively(client)
        await test_apply_colors_and_embeddings(client)
        await test_apply_skips_evicted_member(client)
        await test_sibling_groups_after_apply(client)
    print("all contraction server tests passed")


if __name__ == "__main__":
    asyncio.run(run_tests())

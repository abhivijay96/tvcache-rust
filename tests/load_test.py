#!/usr/bin/env python3

import argparse
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from cache_client import CacheClient


@dataclass(frozen=True)
class FindTarget:
    task_id: str
    path: list[str]
    expected_value: str


@dataclass(frozen=True)
class TaskStats:
    task_id: str
    inserted_nodes: int
    inserted_sandboxes: int
    checked_finds: int
    find_targets: list[FindTarget]


def build_child_name(task_idx: int, depth: int, parent_idx: int, child_idx: int) -> str:
    return f"task-{task_idx:02d}\ndepth-{depth}\nparent-{parent_idx}\nchild-{child_idx}"


def insert_task_graph(
    client: CacheClient,
    task_id: str,
    task_idx: int,
    depth: int,
    children_min: int,
    children_max: int,
    find_checks: int,
    insert_delay: float,
    rng: random.Random,
) -> TaskStats:
    parents: list[list[str]] = [[]]
    leaf_paths: list[list[str]] = []
    values_by_path: dict[tuple[str, ...], str] = {}
    find_targets: list[FindTarget] = []
    inserted_nodes = 0
    inserted_sandboxes = 0

    for current_depth in range(1, depth + 1):
        next_parents: list[list[str]] = []

        for parent_idx, parent_path in enumerate(parents):
            child_count = rng.randint(children_min, children_max)

            for child_idx in range(child_count):
                child_name = build_child_name(task_idx, current_depth, parent_idx, child_idx)
                path = parent_path + [child_name]
                value = f"value for {task_id} depth={current_depth} child={child_idx}"
                sandbox_id = (
                    f"sandbox-{task_id}-{current_depth}-{parent_idx}-{child_idx}"
                    if rng.random() < 0.25
                    else None
                )
                runtime = rng.randint(0, 500)

                response = client.insert(task_id, path, value, sandbox_id, runtime)
                if response != {
                    "type": "insert_result",
                    "value_inserted": True,
                    "sandbox_inserted": sandbox_id is not None,
                }:
                    raise AssertionError(
                        f"unexpected insert response for {task_id} path={path}: {response}"
                    )

                inserted_nodes += 1
                if sandbox_id is not None:
                    inserted_sandboxes += 1
                values_by_path[tuple(path)] = value
                find_targets.append(FindTarget(task_id, path, value))
                next_parents.append(path)
                if insert_delay > 0:
                    time.sleep(insert_delay)

        parents = next_parents
        leaf_paths = next_parents

    checked_finds = 0
    sample_size = min(find_checks, len(leaf_paths))
    for path in rng.sample(leaf_paths, sample_size):
        response = client.find(task_id, path)
        expected_value = values_by_path[tuple(path)]
        if response["type"] != "find_result" or response["value"] != expected_value:
            raise AssertionError(
                f"unexpected find response for {task_id} path={path}: {response}"
            )
        checked_finds += 1

    return TaskStats(
        task_id, inserted_nodes, inserted_sandboxes, checked_finds, find_targets
    )


def timed_find(client: CacheClient, target: FindTarget) -> float:
    started_at = time.perf_counter()
    response = client.find(target.task_id, target.path)
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    if response["type"] != "find_result" or response["value"] != target.expected_value:
        raise AssertionError(
            f"unexpected find response for {target.task_id} path={target.path}: {response}"
        )

    return elapsed_ms


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty list")
    sorted_values = sorted(values)
    index = math.ceil((percent / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def run_find_benchmark(
    client: CacheClient,
    targets: list[FindTarget],
    concurrency: int,
    rng: random.Random,
) -> None:
    selected_targets = [rng.choice(targets) for _ in range(concurrency)]
    batch_started_at = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(timed_find, client, target) for target in selected_targets]
        rcts = [future.result() for future in as_completed(futures)]

    batch_elapsed = time.perf_counter() - batch_started_at

    print(f"Find benchmark concurrency={concurrency}")
    print(f"  queries: {len(rcts)}")
    print(f"  mean rct ms: {statistics.mean(rcts):.2f}")
    print(f"  median rct ms: {statistics.median(rcts):.2f}")
    print(f"  p95 rct ms: {percentile(rcts, 95):.2f}")
    print(f"  batch elapsed seconds: {batch_elapsed:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7878, type=int)
    parser.add_argument("--tasks", default=32, type=int)
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--children-min", default=1, type=int)
    parser.add_argument("--children-max", default=8, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--find-checks", default=8, type=int)
    parser.add_argument("--find-concurrency", default="32,64,128")
    parser.add_argument("--insert-delay", default=0.0, type=float)
    parser.add_argument("--timeout", default=30, type=float)
    parser.add_argument("--retries", default=5, type=int)
    parser.add_argument("--retry-backoff", default=0.1, type=float)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--task-prefix", default=None)
    args = parser.parse_args()

    if args.depth < 1:
        raise ValueError("--depth must be at least 1")
    if args.children_min < 1:
        raise ValueError("--children-min must be at least 1")
    if args.children_min > args.children_max:
        raise ValueError("--children-min must be <= --children-max")

    find_concurrency_levels = [
        int(value.strip()) for value in args.find_concurrency.split(",") if value.strip()
    ]
    if not find_concurrency_levels or any(value < 1 for value in find_concurrency_levels):
        raise ValueError("--find-concurrency must contain positive integers")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")
    if args.retry_backoff < 0:
        raise ValueError("--retry-backoff must be >= 0")
    if args.insert_delay < 0:
        raise ValueError("--insert-delay must be >= 0")

    run_id = args.task_prefix or f"load-{int(time.time())}"
    seed_rng = random.Random(args.seed)
    benchmark_rng = random.Random(args.seed + 1)
    client = CacheClient(
        args.host,
        args.port,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )
    started_at = time.perf_counter()
    stats: list[TaskStats] = []

    print(
        "Starting load test:",
        f"tasks={args.tasks}",
        f"depth={args.depth}",
        f"children={args.children_min}-{args.children_max}",
        f"workers={args.workers}",
        f"insert_delay={args.insert_delay}",
        f"seed={args.seed}",
        f"timeout={args.timeout}",
        f"retries={args.retries}",
        f"prefix={run_id}",
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for task_idx in range(args.tasks):
            task_id = f"{run_id}-task-{task_idx:02d}"
            task_rng = random.Random(seed_rng.randint(0, 2**63 - 1))
            futures.append(
                executor.submit(
                    insert_task_graph,
                    client,
                    task_id,
                    task_idx,
                    args.depth,
                    args.children_min,
                    args.children_max,
                    args.find_checks,
                    args.insert_delay,
                    task_rng,
                )
            )

        for future in as_completed(futures):
            task_stats = future.result()
            stats.append(task_stats)
            print(
                f"{task_stats.task_id}:",
                f"nodes={task_stats.inserted_nodes}",
                f"sandboxes={task_stats.inserted_sandboxes}",
                f"finds={task_stats.checked_finds}",
            )

    elapsed = time.perf_counter() - started_at
    total_nodes = sum(item.inserted_nodes for item in stats)
    total_sandboxes = sum(item.inserted_sandboxes for item in stats)
    total_finds = sum(item.checked_finds for item in stats)
    find_targets = [
        target for task_stats in stats for target in task_stats.find_targets
    ]

    print("TCG creation complete")
    print(f"  tasks: {len(stats)}")
    print(f"  inserted nodes: {total_nodes}")
    print(f"  inserted sandboxes: {total_sandboxes}")
    print(f"  checked finds: {total_finds}")
    print(f"  elapsed seconds: {elapsed:.2f}")
    print(f"  requests/sec: {(total_nodes + total_finds) / elapsed:.2f}")

    if not find_targets:
        raise RuntimeError("no find targets were created")

    print("Starting parallel find benchmarks")
    for concurrency in find_concurrency_levels:
        run_find_benchmark(client, find_targets, concurrency, benchmark_rng)

    print("Load test complete")


if __name__ == "__main__":
    main()

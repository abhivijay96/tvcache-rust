#!/usr/bin/env python3

import argparse
import asyncio
import math
import random
import statistics
import time
from dataclasses import dataclass

from cache_client import AsyncPersistentCacheClient


@dataclass(frozen=True)
class FindTarget:
    task_id: str
    path: list[str]
    expected_value: str
    expected_sandbox_path: list[str] | None
    expected_sandbox_id: str | None


@dataclass(frozen=True)
class TaskData:
    task_id: str
    inserted_nodes: int
    inserted_sandboxes: int
    find_targets: list[FindTarget]


@dataclass(frozen=True)
class FindResult:
    task_id: str
    rct_ms: float


@dataclass(frozen=True)
class LoadConfig:
    host: str
    port: int
    timeout: float
    retries: int
    retry_backoff: float


def build_child_name(task_idx: int, depth: int, parent_idx: int, child_idx: int) -> str:
    return f"task-{task_idx:02d}\ndepth-{depth}\nparent-{parent_idx}\nchild-{child_idx}"


def parse_rollout_counts(value: str) -> list[int]:
    rollout_counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not rollout_counts or any(count < 1 for count in rollout_counts):
        raise ValueError("--num-rollouts must contain positive integers")
    return rollout_counts


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty list")
    sorted_values = sorted(values)
    index = math.ceil((percent / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def expected_insert_response(sandbox_id: str | None) -> dict:
    return {
        "type": "insert_result",
        "value_inserted": True,
        "sandbox_inserted": sandbox_id is not None,
    }


def expected_find_response(target: FindTarget) -> dict:
    return {
        "type": "find_result",
        "value": target.expected_value,
        "sandbox_path": target.expected_sandbox_path,
        "sandbox_id": target.expected_sandbox_id,
    }


async def create_task_tcg(
    config: LoadConfig,
    build_semaphore: asyncio.Semaphore,
    task_id: str,
    task_idx: int,
    depth: int,
    rng: random.Random,
) -> TaskData:
    async with build_semaphore:
        return await create_task_tcg_unlimited(config, task_id, task_idx, depth, rng)


async def create_task_tcg_unlimited(
    config: LoadConfig,
    task_id: str,
    task_idx: int,
    depth: int,
    rng: random.Random,
) -> TaskData:
    parents: list[list[str]] = [[]]
    parent_sandboxes: list[tuple[list[str] | None, str | None]] = [(None, None)]
    find_targets: list[FindTarget] = []
    inserted_nodes = 0
    inserted_sandboxes = 0

    async with AsyncPersistentCacheClient(
        config.host,
        config.port,
        config.timeout,
        config.retries,
        config.retry_backoff,
    ) as client:
        for current_depth in range(1, depth + 1):
            next_parents: list[list[str]] = []
            next_parent_sandboxes: list[tuple[list[str] | None, str | None]] = []

            for parent_idx, parent_path in enumerate(parents):
                child_count = rng.randint(1, 8)
                parent_sandbox_path, parent_sandbox_id = parent_sandboxes[parent_idx]

                for child_idx in range(child_count):
                    path = parent_path + [
                        build_child_name(task_idx, current_depth, parent_idx, child_idx)
                    ]
                    value = (
                        f"value for {task_id} depth={current_depth} "
                        f"parent={parent_idx} child={child_idx}"
                    )
                    sandbox_id = (
                        f"sandbox-{task_id}-{current_depth}-{parent_idx}-{child_idx}"
                        if rng.random() < 0.25
                        else None
                    )
                    runtime = rng.randint(0, 500)

                    response = await client.insert(
                        task_id, path, value, sandbox_id, runtime
                    )
                    if response != expected_insert_response(sandbox_id):
                        raise AssertionError(
                            f"unexpected insert response for {task_id} path={path}: "
                            f"{response}"
                        )

                    inserted_nodes += 1
                    if sandbox_id is not None:
                        inserted_sandboxes += 1

                    if sandbox_id is not None:
                        expected_sandbox_path = path
                        expected_sandbox_id = sandbox_id
                    else:
                        expected_sandbox_path = parent_sandbox_path
                        expected_sandbox_id = parent_sandbox_id

                    find_targets.append(
                        FindTarget(
                            task_id,
                            path,
                            value,
                            expected_sandbox_path,
                            expected_sandbox_id,
                        )
                    )
                    next_parents.append(path)
                    next_parent_sandboxes.append(
                        (expected_sandbox_path, expected_sandbox_id)
                    )

            parents = next_parents
            parent_sandboxes = next_parent_sandboxes

    return TaskData(task_id, inserted_nodes, inserted_sandboxes, find_targets)


async def run_one_find_rollout(
    config: LoadConfig,
    task_data: TaskData,
    rng: random.Random,
) -> FindResult:
    target = rng.choice(task_data.find_targets)

    async with AsyncPersistentCacheClient(
        config.host,
        config.port,
        config.timeout,
        config.retries,
        config.retry_backoff,
    ) as client:
        started_at = time.perf_counter()
        response = await client.find(target.task_id, target.path)
        rct_ms = (time.perf_counter() - started_at) * 1000

    expected_response = expected_find_response(target)
    if response != expected_response:
        raise AssertionError(
            f"unexpected find response for {target.task_id} path={target.path}: "
            f"expected={expected_response} actual={response}"
        )

    return FindResult(target.task_id, rct_ms)


async def build_dataset(args: argparse.Namespace, config: LoadConfig) -> list[TaskData]:
    build_started_at = time.perf_counter()
    seed_rng = random.Random(args.seed)
    build_semaphore = asyncio.Semaphore(args.build_workers)
    jobs = []

    for task_idx in range(args.num_tasks):
        task_id = f"{args.run_id}-task-{task_idx:02d}"
        task_rng = random.Random(seed_rng.randint(0, 2**63 - 1))
        jobs.append(
            asyncio.create_task(
                create_task_tcg(
                    config,
                    build_semaphore,
                    task_id,
                    task_idx,
                    args.depth,
                    task_rng,
                )
            )
        )

    task_data = await asyncio.gather(*jobs)
    build_elapsed = time.perf_counter() - build_started_at
    total_nodes = sum(item.inserted_nodes for item in task_data)
    total_sandboxes = sum(item.inserted_sandboxes for item in task_data)

    for data in task_data:
        print(
            f"{data.task_id}:",
            f"nodes={data.inserted_nodes}",
            f"sandboxes={data.inserted_sandboxes}",
            f"valid_finds={len(data.find_targets)}",
        )

    print("TCG creation complete")
    print(f"  tasks: {len(task_data)}")
    print(f"  inserted nodes: {total_nodes}")
    print(f"  inserted sandboxes: {total_sandboxes}")
    print(f"  elapsed seconds: {build_elapsed:.2f}")
    print(f"  inserts/sec: {total_nodes / build_elapsed:.2f}")

    return task_data


async def run_rollout_phase(
    config: LoadConfig,
    task_data: list[TaskData],
    num_rollouts: int,
    seed: int,
) -> None:
    print(f"Starting rollout find phase: num_rollouts={num_rollouts}")
    rollout_started_at = time.perf_counter()
    rollout_seed_rng = random.Random(seed)
    jobs = []

    for data in task_data:
        for _ in range(num_rollouts):
            rollout_rng = random.Random(rollout_seed_rng.randint(0, 2**63 - 1))
            jobs.append(
                asyncio.create_task(run_one_find_rollout(config, data, rollout_rng))
            )

    results = await asyncio.gather(*jobs)
    rollout_elapsed = time.perf_counter() - rollout_started_at
    rcts = [result.rct_ms for result in results]

    print("Rollout find phase complete")
    print(f"  num_rollouts: {num_rollouts}")
    print(f"  async_tasks: {len(jobs)}")
    print(f"  finds: {len(results)}")
    print(f"  mean rct ms: {statistics.mean(rcts):.2f}")
    print(f"  median rct ms: {statistics.median(rcts):.2f}")
    print(f"  p95 rct ms: {percentile(rcts, 95):.2f}")
    print(f"  elapsed seconds: {rollout_elapsed:.2f}")
    print(f"  finds/sec: {len(results) / rollout_elapsed:.2f}")


async def run_load_test(args: argparse.Namespace) -> None:
    config = LoadConfig(
        args.host,
        args.port,
        args.timeout,
        args.retries,
        args.retry_backoff,
    )
    rollout_counts = parse_rollout_counts(args.num_rollouts)
    total_rollouts = args.num_tasks * sum(rollout_counts)

    print(
        "Starting load test:",
        f"num_tasks={args.num_tasks}",
        f"num_rollouts={rollout_counts}",
        f"total_rollouts={total_rollouts}",
        f"depth={args.depth}",
        "width=random(1..8)",
        f"build_workers={args.build_workers}",
        f"seed={args.seed}",
        f"prefix={args.run_id}",
    )

    task_data = await build_dataset(args, config)

    for idx, num_rollouts in enumerate(rollout_counts):
        await run_rollout_phase(config, task_data, num_rollouts, args.seed + 1 + idx)

    print(f"Total rollouts complete: {total_rollouts}")
    print("Load test complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7878, type=int)
    parser.add_argument("--num-tasks", default=32, type=int)
    parser.add_argument("--num-rollouts", default="1")
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--build-workers", default=8, type=int)
    parser.add_argument("--timeout", default=30, type=float)
    parser.add_argument("--retries", default=5, type=int)
    parser.add_argument("--retry-backoff", default=0.1, type=float)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--task-prefix", default=None)
    args = parser.parse_args()

    if args.num_tasks < 1:
        raise ValueError("--num-tasks must be >= 1")
    if args.depth < 1:
        raise ValueError("--depth must be >= 1")
    if args.build_workers < 1:
        raise ValueError("--build-workers must be >= 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")
    if args.retry_backoff < 0:
        raise ValueError("--retry-backoff must be >= 0")

    args.run_id = args.task_prefix or f"load-{int(time.time())}"
    return args


def main() -> None:
    asyncio.run(run_load_test(parse_args()))


if __name__ == "__main__":
    main()

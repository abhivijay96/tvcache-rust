# Synthetic benchmark results — 2026-11-07

All **288 policy runs passed** (144 Rust, 144 Python references). All four input traces passed integrity and causal-feature validation. `cargo test`: **26 passed**. The 14 legacy-fixture controls passed.

See [compatibility audit](../../docs/evaluator-compatibility.md) for reproduction, accounting differences, client API status, and the missing real-trace dependency.

## Rust results at the smallest nonzero budget

These are synthetic seconds, not measured wall-clock speedup. Modeled net time includes prefix reconstruction; cache overheads are zero placeholders. Original-enforcer ties depend on HashMap order, so that policy may vary across processes.

| Workload | MiB | Policy | Hit rate | Gross saved s | Modeled net saved s |
| --- | ---: | --- | ---: | ---: | ---: |
| balanced | 1 | original-ttl-and-watermark | 54.92% | 2856.1 | 2227.5 |
| balanced | 1 | fifo | 39.27% | 2316.5 | 1546.7 |
| balanced | 1 | linear | 51.18% | 3142.7 | 1692.9 |
| balanced | 1 | lru | 46.33% | 2785.8 | 1751.5 |
| balanced | 1 | lfu | 58.26% | 3073.9 | 2030.3 |
| balanced | 1 | lfu-persistent | 58.96% | 3269.0 | 2039.1 |
| expensive_reuse | 1 | original-ttl-and-watermark | 79.66% | 62021.5 | 60376.1 |
| expensive_reuse | 1 | fifo | 70.54% | 54265.7 | 52688.2 |
| expensive_reuse | 1 | linear | 76.69% | 60231.3 | 54437.0 |
| expensive_reuse | 1 | lru | 77.86% | 59855.0 | 57961.3 |
| expensive_reuse | 1 | lfu | 83.68% | 64732.0 | 61338.1 |
| expensive_reuse | 1 | lfu-persistent | 85.72% | 65587.6 | 63652.9 |
| popularity_shift | 1 | original-ttl-and-watermark | 20.51% | 1115.9 | 779.6 |
| popularity_shift | 1 | fifo | 59.02% | 2659.4 | 2161.2 |
| popularity_shift | 1 | linear | 70.03% | 3325.9 | 2391.3 |
| popularity_shift | 1 | lru | 66.51% | 3067.1 | 2376.2 |
| popularity_shift | 1 | lfu | 57.35% | 2804.4 | 1421.2 |
| popularity_shift | 1 | lfu-persistent | 47.06% | 2145.8 | 1440.0 |
| memory_pressure | 133 | original-ttl-and-watermark | 46.12% | 3261.7 | 1894.6 |
| memory_pressure | 133 | fifo | 34.05% | 2728.1 | 753.8 |
| memory_pressure | 133 | linear | 43.95% | 3576.2 | 738.0 |
| memory_pressure | 133 | lru | 40.74% | 3189.7 | 985.8 |
| memory_pressure | 133 | lfu | 49.63% | 3527.6 | 1235.8 |
| memory_pressure | 133 | lfu-persistent | 52.39% | 3566.0 | 959.1 |

At these budgets, linear has the highest gross savings on popularity shift and memory pressure. After reconstruction, it leads only popularity shift in this run. Persistent LFU leads expensive reuse; the original enforcer leads balanced and memory pressure on modeled net time. These fixtures do not establish generalization.

## Artifacts

- `summary.csv`: all policy metrics, including separate Rust byte accounting and measured policy callback time.
- `completion.json`: successful completion and check counts.
- `run.json`, `validation.json`: revision, capacities, assumptions, and validated input hashes.
- `<workload>/<budget>mib/`: Rust reports, decisions, independently scored reports, and Python references with/without snapshots.
- `legacy-fixture-checks.json`: old single-stream fixture controls.
- `server-checks.json` and `*.log`: live-server checks and load results.

Real TCG traces remain unavailable. Offline-learned and online-adaptive policies are plans and have no runnable implementation.

Live load runs use cache enabled with eviction disabled, depth five and seed 12345; 32/64/128 tasks use 8/32/128 build workers respectively. The find phases use 2/4/8/16 rollouts per task. Setup and local timing conditions limit performance comparisons.

## Live checks and load results

All eight supported live checks passed: protocol, sandbox lifecycle, current client, TTL, watermark, and the three load runs. All **6,720 lookups** passed across 12 load cells. The old `test_tvclient.py` fails on removed API imports; this is recorded separately in `server-checks.json`. The initial lifecycle setup used eviction disabled and failed its TTL leg; the corrected TTL-enabled rerun passed.

| Tasks | Maximum concurrent lookups | Mean ms | p95 ms |
| ---: | ---: | ---: | ---: |
| 32 | 512 | 35.27 | 41.56 |
| 64 | 1024 | 74.24 | 90.66 |
| 128 | 2048 | 163.96 | 205.70 |

All load cells are in `load-summary.csv`; the 128-task raw log is `final-load/load-128.log`. Temporary servers were stopped after the checks. Reproduce with `python3 scripts/benchmark_server.py --output results/server-checks` (defaults to 32 build workers; use `--build-workers` to reproduce the individual setup worker counts above).

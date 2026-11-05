# LRU/LFU budget sweep

Every replay policy on the four synthetic traces, at about 10/25/50/75/100% of
each trace's unique entry bytes. Reproduce from the repository root with:

```sh
cargo run --release --bin sweep    # writes results/lru-lfu/summary.csv (about 2.5 min)
cargo test --test replay           # correctness checks, including the flat-trace reference
```

`summary.csv` has one row per run, taken from the Rust replay report: hit rate,
gross time and USD saved, peak bytes, evictions, missing-parent bypasses, and
policy time per request. Before writing a row, the sweep checks that run's
decision log with `replay::check_decisions` (every hit and victim resident, the
budget held after each event). `lfu-persistent` is `LinearPolicy` with
`configs/lfu-persistent.json`.

These are gross synthetic estimates, not measured training speedups. Net savings
(prefix replay, snapshot restores, rollout latency) are not computed here; the
Rust replay is result-only. The original policy breaks ties by HashMap order, so
its numbers vary between runs. See [replay assumptions](../../docs/replay.md).

Inputs (repo at 9560bd4 plus this change), SHA-256:

```text
eb78438ff11a48575027d641bd4a6ae0b2f7e4c3e46e998ed23ff56a769d5cbc  balanced/events.jsonl
0184c35e2b24677b3d606f9b198b6519df22abe3c6471c7b4d110471c8bae58f  balanced/nodes.jsonl
1e2ba7c880a123a2350257e487f70819e3a927f8dcab18ae2e83f2fda7216d6e  expensive_reuse/events.jsonl
f0810bc7d27ccfaa65a0d2ad4d89231ebf12708f3cc93a7924aed02225af2534  expensive_reuse/nodes.jsonl
b768fddb810a47c9f21aa9acacba586fce9e72c817fcec0cb73aa93d47c11188  memory_pressure/events.jsonl
063cff6df862d47bd7d9fd1326f0b19050e83bf81466b65c0efc939885545efc  memory_pressure/nodes.jsonl
a4395a1cda1cd4f2fbdaad7206d7bedf334b2453a5455cdbb6ceca4df04fec4c  popularity_shift/events.jsonl
3b8f3fc6e3d3f44657ff59202fc346b78377cd47a7e1a1c9c89b9b5311000835  popularity_shift/nodes.jsonl
```

Hit rate (higher is better):

| Trace | MiB | original | FIFO | linear | LRU | LFU | LFU persistent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1 | 0.524 | 0.393 | 0.512 | 0.463 | 0.583 | 0.590 |
| balanced | 3 | 0.797 | 0.675 | 0.794 | 0.736 | 0.796 | 0.797 |
| balanced | 5 | 0.888 | 0.809 | 0.886 | 0.853 | 0.881 | 0.883 |
| expensive_reuse | 1 | 0.797 | 0.705 | 0.767 | 0.779 | 0.837 | 0.857 |
| expensive_reuse | 2 | 0.866 | 0.819 | 0.881 | 0.869 | 0.901 | 0.901 |
| memory_pressure | 133 | 0.423 | 0.340 | 0.440 | 0.407 | 0.496 | 0.524 |
| memory_pressure | 331 | 0.651 | 0.560 | 0.644 | 0.620 | 0.655 | 0.680 |
| popularity_shift | 1 | 0.320 | 0.590 | 0.700 | 0.665 | 0.574 | 0.471 |
| popularity_shift | 3 | 0.490 | 0.845 | 0.877 | 0.875 | 0.798 | 0.802 |
| popularity_shift | 5 | 0.631 | 0.911 | 0.907 | 0.912 | 0.865 | 0.881 |

At the largest budget every policy reaches the trace's full repeat fraction.

Policy time per request (mean over all runs): FIFO 555 ns, linear 582 ns,
LFU persistent 457 ns, LRU 21.5 µs (max 89 µs), LFU 19.6 µs (max 81 µs).
LRU/LFU pay for the max-over-subtree score: each `choose_victim` walks every
candidate's `affected` list. LRU/LFU never hit a missing-parent bypass (0),
versus 23,259 for the original enforcer in this run and 11,058 for persistent
LFU, which scores roots only.

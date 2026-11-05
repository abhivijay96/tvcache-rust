# Recorded smoke runs

`balanced-{0,1,16}mib.json` records original and FIFO runs on the balanced
synthetic trace. Each file includes the exact command and input hashes. These
are integration smoke results, not evidence of performance on real workloads.
See [replay assumptions](../docs/replay.md).

| Capacity | Original hit rate | FIFO hit rate |
| --- | --- | --- |
| 0 MiB | 0% | 0% |
| 1 MiB | 56.11% | 39.27% |
| 16 MiB | 92.26% | 92.26% |

Each run processes 9,613 requests. At 16 MiB, both match all 8,869 repeats and
the trace summary's total unique entry estimate of 9,982,756 bytes. Every run
stays within capacity after enforcement. Transient post-insertion overshoot is
recorded separately. Original-policy ties use randomized HashMap iteration, so
the 1 MiB original result may change when rerun; this is one recorded run.

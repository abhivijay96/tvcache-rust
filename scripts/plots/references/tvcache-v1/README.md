# TVCache reference values

Manually transcribed from the versioned primary source:
[arXiv:2602.10986v1](https://arxiv.org/html/2602.10986v1).

`workloads.csv` records mean hit rates over epochs from Sections 4.1–4.3
and the reported median per-tool-call speedups from Table 2. Preserve the
reported speedups; rounding of the displayed medians means recomputing their
ratios can give slightly different values. Empty fields mean unreported here,
not zero. These are summary values, not digitized epoch curves or raw samples.

Our synthetic tasks, model, budgets, and runtime assumptions do not match these
experiments. Paper-reference panels provide context; they cannot establish
reproduction, superiority, or a statistically meaningful performance ratio.
The plotting scripts never overlay these values on our synthetic curves.

The [published ICML proceedings](https://proceedings.mlr.press/v306/kumar26d.html)
are a separate version. Do not mix their abstract or figures into these v1 data.
Add other papers or versions in their own reference directories.

For scaling, Section 4.5 / Figure 8(a) uses controlled request rates, 8K keys,
and a 128-core host, including sharded servers. Our load script creates bursts
against one server with larger graphs; achieved burst throughput is not offered
request rate. No direct scaling overlay is generated. Training epoch curves,
reward preservation, rollout/batch makespan, and proactive-fork RSS require
measurements that our current summary files do not contain.

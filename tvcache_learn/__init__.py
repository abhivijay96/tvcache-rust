"""Training and fitting for learned TVCache eviction policies.

Pipeline: collect (Rust replay dumps eviction candidates) -> dataset (add
future-reuse labels) -> train (export tvcache-mlp-v1 JSON for the Rust
MlpPolicy). fit_linear searches LinearPolicy weights by replay directly, and
evaluate scores any set of policies with tvcache_eval.
"""

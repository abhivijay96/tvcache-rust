//! Reusable policy contracts and trace adapters for experiments.
pub mod candidates;
pub mod policy;
pub mod replay;
// Share server components with controlled replay experiments.
pub mod eviction;
pub mod tcg;
pub mod tvcache;
pub mod types;
pub mod utils;

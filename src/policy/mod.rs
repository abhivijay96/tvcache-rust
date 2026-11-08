//! Storage-independent contracts. See docs/policy-contract.md for lifecycle rules.
pub mod classic;
pub mod features;
pub mod fifo;
pub(crate) mod history;
pub mod linear;
pub mod mlp;
pub mod synthetic;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Namespace IDs by task even when a trace's node hashes appear globally unique.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntryId {
    pub task_id: String,
    pub node_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SnapshotCost {
    pub external_bytes: u64,
    pub handle_bytes: u64,
    pub create_ms: f64,
    pub restore_ms: f64,
}

#[derive(Clone, Debug)]
pub struct EntryMeta {
    pub id: EntryId,
    pub parent: Option<EntryId>,
    pub path: Vec<String>,
    pub depth: u64,
    pub mutates_state: bool,
    pub tool: String,
    /// Trace's resident entry estimate; distinct from external snapshot storage.
    pub cache_entry_bytes: u64,
    pub value_bytes: u64,
    /// Estimates available at decision time, never a promise of actual savings.
    pub runtime_ms: Option<f64>,
    pub tool_cost_usd: Option<f64>,
    pub snapshot: Option<SnapshotCost>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AccessFeatures {
    pub access_count_before: u64,
    pub last_access_index: Option<u64>,
    pub events_since_last_access: Option<u64>,
    pub time_since_last_access_ms: Option<f64>,
    pub events_since_subtree_access: Option<u64>,
    pub observed_children_before: u64,
    pub observed_parent_children_before: u64,
    pub stateful_prefix_depth: u64,
    pub subtree_access_count_before: u64,
    /// Optional model/heuristic inputs supplied by the engine, from past data only.
    #[serde(default)]
    pub extra: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct AccessContext {
    pub event_index: u64,
    pub time_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryCharge {
    pub resident_bytes: u64,
    pub external_snapshot_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct CacheContext {
    pub access: AccessContext,
    pub used: MemoryCharge,
    pub capacity: MemoryCharge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    Bypass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessOutcome {
    ResultHit,
    SnapshotHit,
    Miss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemovalReason {
    Capacity,
    Expired,
    Invalidated,
    Replaced,
}

/// One legal atomic action, constructed by the engine (single entry or subtree).
/// Policy returns its index, never an arbitrary entry that might be pinned.
#[derive(Debug)]
pub struct EvictionCandidate<'a> {
    pub root: &'a EntryMeta,
    pub affected: &'a [EntryId],
    pub reclaimable: MemoryCharge,
    pub features: &'a AccessFeatures,
}

/// Admit a candidate that individually fits both total budgets; eviction may still be needed.
pub fn admit_if_fits(charge: MemoryCharge, context: &CacheContext) -> AdmissionDecision {
    if charge.resident_bytes <= context.capacity.resident_bytes
        && charge.external_snapshot_bytes <= context.capacity.external_snapshot_bytes
    {
        AdmissionDecision::Admit
    } else {
        AdmissionDecision::Bypass
    }
}

/// Object-safe: evaluator may select Box<dyn CachePolicy> at runtime.
/// Policy owns ranking/model state; engine owns entries, legality and accounting.
pub trait CachePolicy: Send {
    fn name(&self) -> &str;

    /// Optional configuration recorded by the evaluator for reproducibility.
    fn parameters(&self) -> Option<serde_json::Value> {
        None
    }

    /// Called once per logical request, including misses and bypassed requests.
    /// `features` describe history BEFORE this access.
    fn on_access(
        &mut self,
        entry: &EntryMeta,
        features: &AccessFeatures,
        outcome: AccessOutcome,
        context: &AccessContext,
    );

    /// Called after a miss has executed and candidate costs are available.
    fn admission(
        &mut self,
        entry: &EntryMeta,
        charge: MemoryCharge,
        context: &CacheContext,
    ) -> AdmissionDecision;

    /// Return an index in this slice, or None to decline further eviction.
    /// Engine validates the index and rebuilds candidates after each removal.
    fn choose_victim(
        &mut self,
        incoming: Option<&EntryMeta>,
        candidates: &[EvictionCandidate<'_>],
        context: &CacheContext,
    ) -> Option<usize>;

    fn on_insert(&mut self, entry: &EntryMeta, context: &CacheContext);

    /// Called for EVERY removed entry, including descendants of a removed root.
    fn on_remove(&mut self, id: &EntryId, reason: RemovalReason, context: &CacheContext);
}

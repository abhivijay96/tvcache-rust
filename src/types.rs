use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{tcg::ToolCallGraph, tvcache::TVCache};

pub type CacheSharedObject = Arc<Mutex<TVCache>>;
pub type TcgSharedObject = Arc<Mutex<ToolCallGraph>>;
pub type CacheSnapshotObject = Vec<(String, TcgSharedObject)>;

// Lock ordering rule:
// Acquire the cache map lock only to look up or insert a TcgSharedObject, and release
// it before locking the TCG (snapshot pattern). Never hold two TCG locks at once;
// cross-TCG operations (DumpAll, ListSandboxCandidates, DrainReclaimableSandboxes,
// eviction) iterate a snapshot and lock TCGs one at a time.
//
// All per-TCG sandbox reference-count and victim state lives inside ToolCallGraph,
// covered by that TCG's own lock, so request handlers never share a lock across TCGs.

pub enum Request {
    Insert {
        task_id: String,
        path: Vec<String>,
        value: Option<String>,
        sandbox_id: Option<String>,
        // Milliseconds. u32 (~49.7 days) because long tool calls (multi-hour
        // test suites) are precisely the ones worth caching.
        runtime: u32,
        ttl: Option<u32>,
        ttl_present: bool,
    },
    Find {
        task_id: String,
        path: Vec<String>,
    },
    DumpAll,
    UnrefSandbox {
        task_id: String,
        sandbox_id: String,
    },
    MarkSandboxesEvicted {
        sandbox_eviction_records: Vec<SandboxEvictionRecord>,
    },
    ListSandboxCandidates {
        limit: u16,
    },
    DrainReclaimableSandboxes,
    ListHotSandboxes {
        limit: u16,
    },
    ReportForkCost {
        task_id: String,
        fork_cost_ms: u32,
    },
    GetForkCost {
        task_id: String,
    },
}

pub enum Response {
    InsertResult {
        value_inserted: bool,
        sandbox_inserted: bool,
    },
    FindResult {
        value: Option<String>,
        sandbox_path: Option<Vec<String>>,
        sandbox_id: Option<String>,
    },
    UnrefSandboxResult {
        remaining_references: u32,
    },
    MarkSandboxesEvictedResult {
        sandbox_ids_recorded: u32,
    },
    SandboxCandidates {
        candidates: Vec<SandboxCandidate>,
    },
    ReclaimableSandboxes {
        sandboxes: Vec<ReclaimableSandbox>,
    },
    ForkCost {
        fork_cost_ms: u32,
    },
    Error {
        message: String,
    },
    Json {
        json: String,
    },
}

pub struct SandboxCandidate {
    pub task_id: String,
    pub sandbox_path: Vec<String>,
    pub sandbox_id: String,
    pub score: u64,
}

pub struct SandboxEvictionRecord {
    pub task_id: String,
    pub sandbox_path: Vec<String>,
    pub sandbox_id: String,
}

// Drain returns the task_id with each sandbox_id because the Python deletion
// loop constructs its environment objects as env_class(task_name, env_id).
pub struct ReclaimableSandbox {
    pub task_id: String,
    pub sandbox_id: String,
}

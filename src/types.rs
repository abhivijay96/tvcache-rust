use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{eviction::sandbox_tracker::SandboxTracker, tcg::ToolCallGraph, tvcache::TVCache};

pub type CacheSharedObject = Arc<Mutex<TVCache>>;
pub type TcgSharedObject = Arc<Mutex<ToolCallGraph>>;
pub type CacheSnapshotObject = Vec<(String, TcgSharedObject)>;

// Lock ordering rule:
// If code needs both a TCG lock and the sandbox tracker lock, acquire them in this order:
// 1. TCG lock
// 2. Sandbox tracker lock
//
// Never acquire the sandbox tracker lock first and then a TCG lock, or concurrent
// find and eviction paths can deadlock.
pub type SandboxTrackerSharedObject = Arc<Mutex<SandboxTracker>>;

pub enum Request {
    Insert {
        task_id: String,
        path: Vec<String>,
        value: Option<String>,
        sandbox_id: Option<String>,
        runtime: u16,
        ttl: Option<u32>,
        ttl_present: bool,
    },
    Find {
        task_id: String,
        path: Vec<String>,
    },
    DumpAll,
    UnrefSandbox {
        sandbox_id: String,
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
    Error {
        message: String,
    },
    Json {
        json: String,
    },
}

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{tcg::ToolCallGraph, tvcache::TVCache};

pub type CacheSharedObject = Arc<Mutex<TVCache>>;
pub type TcgSharedObject = Arc<Mutex<ToolCallGraph>>;

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
    Error {
        message: String,
    },
    Json {
        json: String,
    },
}

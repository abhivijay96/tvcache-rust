use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

use crate::{tcg::ToolCallGraph, types::TcgSharedObject};

#[derive(Debug, Default)]
pub struct TVCache {
    pub tcgs: HashMap<String, TcgSharedObject>,
}

impl TVCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_tcg(&mut self, key: String) -> TcgSharedObject {
        // check if key in tcgs, if present, return the tcg, if not, create a new tcg, store it with the key and return that.
        Arc::clone(
            self.tcgs
                .entry(key)
                .or_insert_with(|| Arc::new(Mutex::new(ToolCallGraph::new()))),
        )
    }

    pub fn entries_snapshot(&self) -> Vec<(String, TcgSharedObject)> {
        let mut result = Vec::<(String, TcgSharedObject)>::new();

        for (task_id, tcg_arc_object) in self.tcgs.iter() {
            result.push((task_id.clone(), tcg_arc_object.clone()));
        }

        result
    }
}

pub async fn tcg_entries_to_json(
    entries: Vec<(String, TcgSharedObject)>,
) -> Result<String, serde_json::Error> {
    let mut result: HashMap<String, ToolCallGraph> = HashMap::new();

    for (task_id, tcg) in entries {
        let graph = tcg.lock().await;
        result.insert(task_id, graph.clone());
    }

    serde_json::to_string_pretty(&result)
}

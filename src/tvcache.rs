#![allow(dead_code)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

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

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let mut result: HashMap<String, ToolCallGraph> = HashMap::new();

        for (task_id, tcg) in &self.tcgs {
            let graph = tcg.lock().unwrap();
            result.insert(task_id.clone(), graph.clone());
        }

        serde_json::to_string_pretty(&result)
    }
}

use std::collections::{hash_map::Entry, HashMap};

use crate::utils;
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Node {
    pub children: HashMap<String, Node>,
    pub sandbox_id: Option<String>,
    pub value: Option<String>,
    pub runtime: u16,
    pub ttl_present: bool,
    pub ttl: u32, // seconds
    pub inserted_at_seconds: u64,
    pub number_of_hits: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ToolCallGraph {
    pub root: Node,
    pub number_of_nodes: u32,
    pub number_of_sandboxes: u32,
}

impl ToolCallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        path: Vec<String>,
        value: Option<String>,
        sandbox_id: Option<String>,
        runtime: u16,
        ttl: Option<u32>,
        ttl_exists: bool,
    ) -> (bool, bool) {
        if path.is_empty() {
            return (false, false);
        }

        let last_idx = path.len() - 1;
        let mut curr: &mut Node = &mut self.root;

        for (idx, p) in path.into_iter().enumerate() {
            if idx < last_idx {
                if let Some(next) = curr.children.get_mut(&p) {
                    curr = next;
                } else {
                    return (false, false);
                }
            } else {
                let new_node = match curr.children.entry(p) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => {
                        self.number_of_nodes += 1;
                        entry.insert(Node::default())
                    }
                };

                new_node.inserted_at_seconds = utils::current_utc_seconds();
                new_node.value = value;
                new_node.runtime = runtime;
                new_node.ttl_present = ttl_exists;

                if let Some(ttl_value) = ttl {
                    // assert that ttl_present is true
                    assert!(
                        new_node.ttl_present,
                        "ttl_present should be true when ttl has a value"
                    );
                    new_node.ttl = ttl_value;
                }

                if sandbox_id.is_some() && new_node.sandbox_id.is_none() {
                    new_node.sandbox_id = sandbox_id;
                    self.number_of_sandboxes += 1;
                    return (true, true);
                }

                return (true, false);
            }
        }

        (false, false)
    }

    pub fn find(
        &mut self,
        path: Vec<String>,
    ) -> (Option<String>, (Option<Vec<String>>, Option<String>)) {
        // this function tries to find the exact path in the tcg and return the value of the last node or return the longest prefix match with a sandbox id

        let mut curr = &mut self.root;
        let mut curr_path: Vec<String> = Vec::new();
        let mut sandbox_path: Option<Vec<String>> = None;
        let mut sandbox_id: Option<String> = None;
        let mut value_of_path: Option<String> = None;
        let path_len = path.len();

        if let Some(id) = &curr.sandbox_id {
            sandbox_id = Some(id.clone());
        }

        for p in path {
            if let Some(next) = curr.children.get_mut(&p) {
                curr_path.push(p);
                curr = next;

                if let Some(id) = &curr.sandbox_id {
                    sandbox_id = Some(id.clone());
                    sandbox_path = Some(curr_path.clone());
                }

                if curr_path.len() == path_len {
                    curr.number_of_hits += 1;
                    value_of_path = curr.value.clone();
                }
            } else {
                // The path has diverged from the tcg, so return what you have seen so far, if value is found, return it
                break;
            }
        }

        return (value_of_path, (sandbox_path, sandbox_id));
    }
}

// TODO: Write TTL and then pruning logic and persistence
// also If I have returned a sandbox id during insertion, I want to be able to prevent from removing that sandbox because someone might be copying the sandbox. also I want to maintain an evicted sandbox id list

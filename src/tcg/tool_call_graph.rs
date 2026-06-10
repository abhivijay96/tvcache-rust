use crate::tcg::Node;
use crate::types::SandboxCandidate;
use crate::utils;
use serde::Serialize;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, Serialize)]
pub struct ToolCallGraph {
    pub root: Node,
    pub number_of_nodes: u32,
    pub number_of_sandboxes: u32,
    pub size_in_bytes: u64,
    // sandbox_id -> reference count; only positive counts are stored, an
    // absent entry means a count of zero.
    pub sandbox_references: BTreeMap<String, u32>,
    // Sandboxes marked for eviction (by the sandbox manager or the internal
    // eviction loop) that are waiting to be handed to the deletion loop via
    // drain_reclaimable_sandboxes once their reference count is zero.
    pub reclaimable_sandboxes: BTreeSet<String>,
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
        runtime: u32,
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
                    // Not inserting, so no need to call update size.
                    return (false, false);
                }
            } else {
                let path_entry_size_in_bytes = p.len() as u32;
                let mut sandbox_id_consumed = false;

                let (new_node, old_size_in_bytes) = match curr.children.entry(p) {
                    Entry::Occupied(entry) => {
                        let node = entry.into_mut();
                        let old_size_in_bytes = node.size_in_bytes;

                        (node, old_size_in_bytes)
                    }
                    Entry::Vacant(entry) => {
                        self.number_of_nodes += 1;

                        (entry.insert(Node::default()), 0)
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
                    sandbox_id_consumed = true;
                }

                new_node.update_size_in_bytes(path_entry_size_in_bytes);
                self.size_in_bytes =
                    self.size_in_bytes - old_size_in_bytes as u64 + new_node.size_in_bytes as u64;

                return (true, sandbox_id_consumed);
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
                    curr.number_of_cache_hits += 1;
                    value_of_path = curr.value.clone();
                }
            } else {
                // The path has diverged from the tcg, so return what you have seen so far, if value is found, return it
                break;
            }
        }

        if value_of_path.is_none() {
            if sandbox_id.is_some() {
                let mut sandbox_node = &mut self.root;

                for p in sandbox_path.as_ref().expect("Sandbox path is invalid") {
                    sandbox_node = sandbox_node
                        .children
                        .get_mut(p)
                        .expect("The LPM path that should be valid is empty now");
                }

                // if we start seeing errors here, lets add more logs and reproduce it.

                sandbox_node.number_of_lpm_hits += 1;
            }
        }

        return (value_of_path, (sandbox_path, sandbox_id));
    }

    pub fn collect_sandbox_candidates(&self, task_id: &str) -> Vec<SandboxCandidate> {
        let mut candidates = Vec::new();
        let mut current_path = Vec::new();

        collect_sandbox_candidates_recursive(
            task_id,
            &self.root,
            &mut current_path,
            &mut candidates,
        );

        candidates
    }

    pub fn increment_sandbox_reference(&mut self, sandbox_id: &str) -> u32 {
        let reference_count = self
            .sandbox_references
            .entry(sandbox_id.to_string())
            .or_insert(0);
        *reference_count += 1;
        *reference_count
    }

    pub fn decrement_sandbox_reference(&mut self, sandbox_id: &str) -> u32 {
        let Some(reference_count) = self.sandbox_references.get_mut(sandbox_id) else {
            return 0;
        };

        *reference_count -= 1;

        if *reference_count == 0 {
            self.sandbox_references.remove(sandbox_id);
            return 0;
        }

        *reference_count
    }

    pub fn add_reclaimable_sandboxes(&mut self, sandbox_ids: &[String]) {
        for sandbox_id in sandbox_ids {
            self.reclaimable_sandboxes.insert(sandbox_id.clone());
        }
    }

    pub fn mark_sandbox_evicted(&mut self, sandbox_path: &[String], sandbox_id: &str) {
        // Always record the victim, even if the node is gone or mismatched,
        // so the id flows back to the deletion loop through drain instead of leaking.
        self.reclaimable_sandboxes.insert(sandbox_id.to_string());

        let mut curr: &mut Node = &mut self.root;

        for p in sandbox_path {
            if let Some(next) = curr.children.get_mut(p) {
                curr = next;
            } else {
                // Node already removed (e.g. by the internal eviction loop).
                return;
            }
        }

        // as_deref turns Option<String> into Option<&str> for an allocation-free compare.
        if sandbox_path.is_empty() || curr.sandbox_id.as_deref() != Some(sandbox_id) {
            return;
        }

        curr.sandbox_id = None;
        self.number_of_sandboxes -= 1;

        let path_entry_size_in_bytes = sandbox_path
            .last()
            .map_or(0, |path_entry| path_entry.len() as u32);
        let old_size_in_bytes = curr.size_in_bytes;
        curr.update_size_in_bytes(path_entry_size_in_bytes);
        self.size_in_bytes =
            self.size_in_bytes - old_size_in_bytes as u64 + curr.size_in_bytes as u64;
    }

    pub fn drain_reclaimable_sandboxes(&mut self) -> Vec<String> {
        // Victims still referenced by a rollout stay in the set for a later drain.
        let sandbox_references = &self.sandbox_references;
        let mut ready = Vec::new();

        self.reclaimable_sandboxes.retain(|sandbox_id| {
            if sandbox_references.contains_key(sandbox_id) {
                true
            } else {
                ready.push(sandbox_id.clone());
                false
            }
        });

        ready
    }
}

fn collect_sandbox_candidates_recursive(
    task_id: &str,
    node: &Node,
    current_path: &mut Vec<String>,
    candidates: &mut Vec<SandboxCandidate>,
) {
    if let Some(sandbox_id) = &node.sandbox_id {
        candidates.push(SandboxCandidate {
            task_id: task_id.to_string(),
            sandbox_path: current_path.clone(),
            sandbox_id: sandbox_id.clone(),
            score: node.number_of_lpm_hits,
        });
    }

    for (path_entry, child) in &node.children {
        current_path.push(path_entry.clone());
        collect_sandbox_candidates_recursive(task_id, child, current_path, candidates);
        current_path.pop();
    }
}

// TODO: Write TTL and then pruning logic and persistence

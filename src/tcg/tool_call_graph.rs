use crate::tcg::Node;
use crate::types::SandboxCandidate;
use crate::utils;
use serde::Serialize;
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
    pub reclaimable_sandboxes: BTreeSet<String>, // Sandboxes marked for eviction (by the sandbox manager or the internal eviction loop) that are waiting to be handed to the deletion loop
    pub fork_cost_ema_ms: u32, // EMA of client-reported sandbox fork times for this task, in milliseconds, u32 is ok here because, it can store up to 4 million seconds which is > 40 days.
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
        embedding: Option<Vec<f32>>,
    ) -> (bool, bool) {
        if path.is_empty() {
            return (false, false);
        }

        let last_idx = path.len() - 1;
        let mut curr: &mut Node = &mut self.root;

        for (idx, p) in path.into_iter().enumerate() {
            if idx < last_idx {
                if let Some(next) = curr.child_mut(&p) {
                    curr = next;
                } else {
                    // Not inserting, so no need to call update size.
                    return (false, false);
                }
            } else {
                let mut sandbox_id_consumed = false;

                let (new_node, is_new_node) = curr.get_or_create_child(p.clone());
                let old_size_in_bytes = new_node.size_in_bytes;
                if is_new_node {
                    self.number_of_nodes += 1;
                }
                let entry = new_node.tool_calls.get_mut(&p).expect("Missing tool call");

                // Merge semantics: absent fields leave existing node state
                // untouched, so an insert with only sandbox cannot destroy a cached
                // value or clear a TTL. New nodes start from Node::default(),
                // so the same code will fully initialize them.
                if is_new_node || value.is_some() {
                    new_node.inserted_at_seconds = utils::current_utc_seconds();
                    entry.value = value;
                    entry.runtime = runtime;
                }

                if ttl_exists {
                    new_node.ttl_present = true;

                    if let Some(ttl) = ttl {
                        new_node.ttl = ttl;
                    }
                }

                if sandbox_id.is_some() && new_node.sandbox_id.is_none() {
                    new_node.sandbox_id = sandbox_id;
                    self.number_of_sandboxes += 1;
                    sandbox_id_consumed = true;
                }

                // A tool call's embedding never changes, so keep the first one.
                if embedding.is_some() && entry.embedding.is_none() {
                    entry.embedding = embedding;
                    new_node.settled = false;
                }

                new_node.update_size_in_bytes();
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

        let path_len = path.len();
        let mut value_of_path: Option<String> = None;

        let (sandbox_path, sandbox_id) = self.walk_path(&path, |node, depth| {
            if depth == path_len {
                node.number_of_cache_hits += 1;
                value_of_path = node.tool_calls[&path[depth - 1]].value.clone();
            }
        });

        if value_of_path.is_none() && sandbox_id.is_some() {
            let mut sandbox_node = &mut self.root;

            for p in sandbox_path.as_ref().expect("Sandbox path is invalid") {
                sandbox_node = sandbox_node
                    .child_mut(p)
                    .expect("The LPM path that should be valid is empty now");
            }

            // if we start seeing errors here, lets add more logs and reproduce it.

            sandbox_node.number_of_lpm_hits += 1;
        }

        return (value_of_path, (sandbox_path, sandbox_id));
    }

    /// Returns the deepest snapshot on path and references it, without counting hits.
    pub fn lease_sandbox(&mut self, path: &[String]) -> (Option<Vec<String>>, Option<String>) {
        let lease = self.walk_path(path, |_, _| {});
        if let Some(sandbox_id) = &lease.1 {
            self.increment_sandbox_reference(sandbox_id);
        }
        lease
    }

    /// Follows path while the graph matches it, visiting each matched node with its depth.
    /// Returns the deepest snapshot on the matched prefix.
    fn walk_path<F>(
        &mut self,
        path: &[String],
        mut visit: F,
    ) -> (Option<Vec<String>>, Option<String>)
    where
        F: FnMut(&mut Node, usize),
    {
        let mut curr = &mut self.root;
        let mut sandbox_path = None;
        let mut sandbox_id = None;

        // if let Some(id) = &curr.sandbox_id {
        //     sandbox_id = Some(id.clone());
        // } // No need to store root's sandbox id

        for (index, p) in path.iter().enumerate() {
            // The path has diverged from the tcg, so return what has been seen so far.
            let Some(next) = curr.child_mut(p) else {
                break;
            };
            curr = next;
            visit(curr, index + 1);

            if let Some(id) = &curr.sandbox_id {
                sandbox_id = Some(id.clone());
                sandbox_path = Some(path[..=index].to_vec());
            }
        }

        (sandbox_path, sandbox_id)
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
            if let Some(next) = curr.child_mut(p) {
                curr = next;
            } else {
                // Node already removed (this can happen by TTL eviction or TCG size based eviction that runs in the background).
                return;
            }
        }

        // as_deref turns Option<String> into Option<&str> for an allocation-free compare.
        if sandbox_path.is_empty() || curr.sandbox_id.as_deref() != Some(sandbox_id) {
            return;
        }

        curr.sandbox_id = None;
        self.number_of_sandboxes -= 1;

        let old_size_in_bytes = curr.size_in_bytes;

        curr.update_size_in_bytes();

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

    pub fn record_fork_cost(&mut self, sample_ms: u32) -> u32 {
        // Integer EMA with alpha = 0.3; the first sample seeds it. u64
        // intermediate so 7 * ema cannot overflow u32.
        // EMA means exponential moving average.
        if self.fork_cost_ema_ms == 0 {
            self.fork_cost_ema_ms = sample_ms;
        } else {
            self.fork_cost_ema_ms =
                ((7 * self.fork_cost_ema_ms as u64 + 3 * sample_ms as u64) / 10) as u32;
        }

        self.fork_cost_ema_ms
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

    for (path_entry, child) in node.iter_children() {
        current_path.push(path_entry.clone());
        collect_sandbox_candidates_recursive(task_id, child, current_path, candidates);
        current_path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_call_entries_preserve_insert_semantics() {
        let mut graph = ToolCallGraph::new();
        assert!(graph.root.tool_calls.is_empty());
        graph.insert(
            vec!["a".into()],
            Some("value".into()),
            Some("snapshot".into()),
            7,
            Some(60),
            true,
            None,
        );
        let original_size = graph.size_in_bytes;
        for (embedding, settled) in [
            (None, true),
            (Some(vec![1.0, 0.0]), false),
            (Some(vec![0.0, 1.0]), true),
            (None, true),
        ] {
            graph.root.child_mut("a").unwrap().settled = true;
            assert_eq!(
                graph.insert(
                    vec!["a".into()],
                    None,
                    Some("replacement".into()),
                    99,
                    None,
                    false,
                    embedding,
                ),
                (true, false)
            );
            assert_eq!(graph.root.child("a").unwrap().settled, settled);
        }
        assert_eq!(graph.find(vec!["a".into()]).0.as_deref(), Some("value"));
        assert_eq!((graph.number_of_nodes, graph.number_of_sandboxes), (1, 1));
        assert_eq!(graph.size_in_bytes, original_size + 8);
        let node = graph.root.child_mut("a").unwrap();
        assert_eq!(node.sandbox_id.as_deref(), Some("snapshot"));
        assert!(node.ttl_present && node.ttl == 60);
        assert_eq!(node.tool_calls["a"].runtime, 7);
        assert_eq!(node.tool_calls["a"].embedding, Some(vec![1.0, 0.0]));
        let original_size = node.size_in_bytes;
        node.tool_calls
            .insert("alias".into(), node.tool_calls["a"].clone());
        node.update_size_in_bytes();
        assert_eq!(node.size_in_bytes, original_size + 5 + 5 + 4 + 8);
    }

    #[test]
    fn lease_returns_deepest_snapshot_without_counting_hits() {
        let path = |calls: &[&str]| {
            calls
                .iter()
                .map(|call| call.to_string())
                .collect::<Vec<_>>()
        };
        let mut graph = ToolCallGraph::new();
        graph.insert(
            path(&["a"]),
            Some("a".into()),
            Some("shallow".into()),
            1,
            None,
            false,
            None,
        );
        graph.insert(
            path(&["a", "b"]),
            Some("b".into()),
            Some("deep".into()),
            1,
            None,
            false,
            None,
        );
        graph.insert(
            path(&["a", "b", "c"]),
            Some("c".into()),
            None,
            1,
            None,
            false,
            None,
        );

        let deep = (Some(path(&["a", "b"])), Some("deep".to_string()));
        assert_eq!(graph.lease_sandbox(&path(&["a", "b", "c"])), deep);
        assert_eq!(graph.lease_sandbox(&path(&["a", "b", "missing"])), deep);
        assert_eq!(graph.lease_sandbox(&path(&["missing"])), (None, None));
        assert_eq!(graph.lease_sandbox(&[]), (None, None));
        assert_eq!(graph.sandbox_references["deep"], 2);
        assert!(!graph.sandbox_references.contains_key("shallow"));
        let hits = |graph: &ToolCallGraph| {
            graph.root.walk_subtcg(0, &mut |acc, node| {
                acc + node.number_of_cache_hits + node.number_of_lpm_hits
            })
        };
        assert_eq!(hits(&graph), 0);

        assert_eq!(
            graph.find(path(&["a", "b", "c"])),
            (Some("c".into()), deep.clone())
        );
        assert_eq!(
            graph.find(path(&["a", "b"])),
            (Some("b".into()), deep.clone())
        );
        assert_eq!(graph.find(path(&["a", "b", "missing"])), (None, deep));
        assert_eq!(graph.find(Vec::new()), (None, (None, None)));
        let b = graph.root.child("a").unwrap().child("b").unwrap();
        assert_eq!((b.number_of_cache_hits, b.number_of_lpm_hits), (1, 1));
        assert_eq!(b.child("c").unwrap().number_of_cache_hits, 1);
        assert_eq!(hits(&graph), 3);
        assert_eq!(graph.sandbox_references["deep"], 2);
    }
}

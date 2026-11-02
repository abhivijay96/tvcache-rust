use std::collections::HashMap;

use crate::{
    eviction::{EvictionPolicyConfig, EvictionResult},
    tcg::Node,
    types::{CacheSharedObject, CacheSnapshotObject},
    utils,
};

#[derive(Debug)]
struct EvictionCandidate {
    task_id: String,
    parent_path: Vec<String>,
    child_key: String,
    total_hits: u64,
}

fn get_stats_of_removed_node(node: &Node) -> EvictionResult {
    let mut result = EvictionResult {
        nodes_removed: 1,
        sandboxes_removed: if node.sandbox_id.is_some() { 1 } else { 0 },
        size_removed_in_bytes: node.size_in_bytes as u64,
        sandbox_ids_removed: Vec::new(),
    };

    if let Some(sandbox_id) = &node.sandbox_id {
        result.sandbox_ids_removed.push(sandbox_id.clone());
    }

    // Node will not be none because it is not optional, if there is a key, there is a node

    for (_, child) in node.children.iter() {
        let child_count = get_stats_of_removed_node(child);
        result.update(child_count);
    }

    return result;
}

fn has_node_expired(node: &Node) -> bool {
    let now_seconds = utils::current_utc_seconds();
    return node.ttl_present && (node.inserted_at_seconds + (node.ttl as u64) < now_seconds);
}

fn prune_expired_recursive(node: &mut Node, eviction_result: &mut EvictionResult) {
    node.children.retain(|_, child| {
        if has_node_expired(child) {
            let eviction_result_child = get_stats_of_removed_node(child);

            eviction_result.update(eviction_result_child);

            false
        } else {
            prune_expired_recursive(child, eviction_result);
            true
        }
    });
}

fn get_high_watermark_size_in_bytes(config: &EvictionPolicyConfig) -> u64 {
    config.high_watermark_mb as u64 * 1024 * 1024
}

async fn get_total_cache_size_in_bytes(entries: &CacheSnapshotObject) -> u64 {
    let mut total_size_in_bytes = 0;

    for (_task_id, tcg) in entries {
        let tcg = tcg.lock().await;
        total_size_in_bytes += tcg.size_in_bytes;
    }

    total_size_in_bytes
}

fn populate_tcg_eviction_candidates(
    task_id: &str,
    node: &Node,
    parent_path: &mut Vec<String>,
    candidates: &mut Vec<EvictionCandidate>,
) {
    let mut children = Vec::<(&String, &Node)>::new();

    for (node_key, child) in node.children.iter() {
        children.push((node_key, child));
    }

    for (node_key, child) in children {
        let total_hits = child.walk_subtcg(0, &mut |acc, node| acc + node.number_of_cache_hits);

        candidates.push(EvictionCandidate {
            task_id: task_id.to_string(),
            parent_path: parent_path.clone(),
            child_key: node_key.clone(),
            total_hits,
        });

        parent_path.push(node_key.clone());
        populate_tcg_eviction_candidates(task_id, child, parent_path, candidates);
        parent_path.pop();
    }
}

fn find_parent_node_mut<'a>(root: &'a mut Node, parent_path: &[String]) -> Option<&'a mut Node> {
    let mut curr = root;

    for path_entry in parent_path {
        curr = curr.children.get_mut(path_entry)?;
    }

    Some(curr)
}

async fn prune_cache_to_fit(
    entries: &CacheSnapshotObject,
    required_size: u64,
    eviction_result: &mut EvictionResult,
) {
    // Works on a snapshot of the TCG handles: the global map lock is never held
    // here, and TCG locks are taken one at a time (see the lock ordering rule
    // in types.rs).
    // go over every node in every tcg
    // get the total hits in that subtcg starting at a node
    // make a list with all the visited nodes and their parent
    let mut candidates = Vec::<EvictionCandidate>::new();

    for (task_id, tcg) in entries.iter() {
        let tcg = tcg.lock().await;
        populate_tcg_eviction_candidates(task_id, &tcg.root, &mut Vec::new(), &mut candidates);
    }

    // sort it in ascending order of total hits
    candidates.sort_by_key(|candidate| candidate.total_hits);

    // evict nodes until required_size amount of bytes are evicted
    let mut size_evicted_in_bytes = 0;
    let tcgs_by_task: HashMap<&str, _> = entries
        .iter()
        .map(|(task_id, tcg)| (task_id.as_str(), tcg))
        .collect();

    for candidate in candidates {
        if size_evicted_in_bytes >= required_size {
            break;
        }

        let Some(tcg) = tcgs_by_task.get(candidate.task_id.as_str()) else {
            continue;
        };

        let mut tcg = tcg.lock().await;
        let Some(parent_node) = find_parent_node_mut(&mut tcg.root, &candidate.parent_path) else {
            continue;
        };

        let Some(removed_node) = parent_node.children.remove(&candidate.child_key) else {
            continue;
        };

        let candidate_result = get_stats_of_removed_node(&removed_node);
        size_evicted_in_bytes += candidate_result.size_removed_in_bytes;

        tcg.number_of_nodes -= candidate_result.nodes_removed;
        tcg.number_of_sandboxes -= candidate_result.sandboxes_removed;
        tcg.size_in_bytes -= candidate_result.size_removed_in_bytes;

        // The trie node is gone, but the actual sandbox is only deleted once the
        // deletion loop drains it with a zero reference count.
        tcg.add_reclaimable_sandboxes(&candidate_result.sandbox_ids_removed);

        eviction_result.update(candidate_result);
    }

    // while evicting, update the parent tcg's total bytes value
}

pub async fn enforce(cache: CacheSharedObject, config: &EvictionPolicyConfig) -> EvictionResult {
    let mut result = EvictionResult::default();
    let entries = {
        let cache = cache.lock().await;
        cache.entries_snapshot()
    };

    for (_task_id, tcg) in &entries {
        let mut tcg = tcg.lock().await;
        let mut tcg_result = EvictionResult::default();

        prune_expired_recursive(&mut tcg.root, &mut tcg_result);

        tcg.number_of_nodes -= tcg_result.nodes_removed;
        tcg.number_of_sandboxes -= tcg_result.sandboxes_removed;
        tcg.size_in_bytes -= tcg_result.size_removed_in_bytes;

        // The trie nodes are gone, but the actual sandboxes are only deleted once
        // the deletion loop drains them with a zero reference count.
        tcg.add_reclaimable_sandboxes(&tcg_result.sandbox_ids_removed);

        result.update(tcg_result);
    }

    let high_watermark_size_in_bytes = get_high_watermark_size_in_bytes(config);
    let total_cache_size_in_bytes = get_total_cache_size_in_bytes(&entries).await;

    if total_cache_size_in_bytes > high_watermark_size_in_bytes {
        let required_size = total_cache_size_in_bytes - high_watermark_size_in_bytes;
        prune_cache_to_fit(&entries, required_size, &mut result).await;
    }

    let _low_watermark_mb = config.low_watermark_mb;
    let _total_tcgs = entries.len();

    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::Mutex;

    use super::*;
    use crate::tcg::ToolCallGraph;

    fn p(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|e| e.to_string()).collect()
    }

    fn insert(tcg: &mut ToolCallGraph, path: &[&str], value: &str, sandbox: Option<&str>, ttl: Option<u32>) {
        tcg.insert(
            p(path),
            Some(value.to_string()),
            sandbox.map(str::to_string),
            1,
            ttl,
            ttl.is_some(),
        );
    }

    #[test]
    fn ttl_prune_removes_expired_subtree_and_keeps_siblings() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["old"], "v", Some("s-old"), Some(0));
        insert(&mut tcg, &["old", "child"], "v", Some("s-child"), None);
        insert(&mut tcg, &["fresh"], "v", Some("s-fresh"), None);
        tcg.root.children.get_mut("old").unwrap().inserted_at_seconds = 1;

        let mut result = EvictionResult::default();
        prune_expired_recursive(&mut tcg.root, &mut result);

        tcg.number_of_nodes -= result.nodes_removed;
        tcg.number_of_sandboxes -= result.sandboxes_removed;
        tcg.size_in_bytes -= result.size_removed_in_bytes;
        tcg.add_reclaimable_sandboxes(&result.sandbox_ids_removed);

        assert_eq!(result.nodes_removed, 2);
        assert!(tcg.root.children.contains_key("fresh"));
        assert!(!tcg.root.children.contains_key("old"));
        let mut drained = tcg.drain_reclaimable_sandboxes();
        drained.sort();
        assert_eq!(drained, vec!["s-child".to_string(), "s-old".to_string()]);
        assert!(tcg.check_invariants().is_empty(), "{:?}", tcg.check_invariants());
    }

    #[tokio::test]
    async fn watermark_prune_evicts_coldest_first_and_keeps_counters_consistent() {
        let mut tcg = ToolCallGraph::new();
        insert(&mut tcg, &["cold"], "x".repeat(1000).as_str(), Some("s-cold"), None);
        insert(&mut tcg, &["hot"], "y".repeat(1000).as_str(), Some("s-hot"), None);
        for _ in 0..5 {
            tcg.find(p(&["hot"]));
        }
        let tcg = Arc::new(Mutex::new(tcg));
        let entries: CacheSnapshotObject = vec![("task".to_string(), Arc::clone(&tcg))];

        let mut result = EvictionResult::default();
        prune_cache_to_fit(&entries, 500, &mut result).await;

        let tcg = tcg.lock().await;
        assert!(!tcg.root.children.contains_key("cold"));
        assert!(tcg.root.children.contains_key("hot"));
        assert_eq!(result.sandbox_ids_removed, vec!["s-cold".to_string()]);
        assert!(tcg.reclaimable_sandboxes.contains("s-cold"));
        assert!(tcg.check_invariants().is_empty(), "{:?}", tcg.check_invariants());
    }

    #[tokio::test]
    async fn watermark_prune_does_not_need_the_global_map_lock() {
        // Regression for bug B5: the global map lock is held by someone else
        // while eviction runs; eviction must still make progress.
        let cache = Arc::new(Mutex::new(crate::tvcache::TVCache::new()));
        let tcg = {
            let mut guard = cache.lock().await;
            guard.get_tcg("task".to_string())
        };
        insert(&mut *tcg.lock().await, &["a"], "z".repeat(1000).as_str(), None, None);
        let entries = cache.lock().await.entries_snapshot();

        let _held = cache.lock().await;
        let mut result = EvictionResult::default();
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            prune_cache_to_fit(&entries, 1, &mut result),
        )
        .await
        .expect("prune_cache_to_fit blocked on the global map lock");
        assert_eq!(result.nodes_removed, 1);
    }
}

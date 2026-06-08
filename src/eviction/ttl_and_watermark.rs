use crate::{
    eviction::{EvictionPolicyConfig, EvictionResult},
    tcg::Node,
    tvcache::TVCache,
    types::{CacheSharedObject, CacheSnapshotObject, SandboxTrackerSharedObject},
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
    cache: &mut TVCache,
    required_size: u64,
    eviction_result: &mut EvictionResult,
) {
    // go over every node in every tcg
    // get the total hits in that subtcg starting at a node
    // make a list with all the visited nodes and their parent
    let mut candidates = Vec::<EvictionCandidate>::new();

    for (task_id, tcg) in cache.tcgs.iter() {
        let tcg = tcg.lock().await;
        populate_tcg_eviction_candidates(task_id, &tcg.root, &mut Vec::new(), &mut candidates);
    }

    // sort it in ascending order of total hits
    candidates.sort_by_key(|candidate| candidate.total_hits);

    // evict nodes until required_size amount of bytes are evicted
    let mut size_evicted_in_bytes = 0;

    for candidate in candidates {
        if size_evicted_in_bytes >= required_size {
            break;
        }

        let Some(tcg) = cache.tcgs.get(&candidate.task_id) else {
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

        eviction_result.update(candidate_result);
    }

    // while evicting, update the parent tcg's total bytes value
}

pub async fn enforce(
    cache: CacheSharedObject,
    config: &EvictionPolicyConfig,
    sandbox_tracker: SandboxTrackerSharedObject,
) -> EvictionResult {
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

        result.update(tcg_result);
    }

    let high_watermark_size_in_bytes = get_high_watermark_size_in_bytes(config);
    let total_cache_size_in_bytes = get_total_cache_size_in_bytes(&entries).await;

    if total_cache_size_in_bytes > high_watermark_size_in_bytes {
        let required_size = total_cache_size_in_bytes - high_watermark_size_in_bytes;
        let mut cache = cache.lock().await;
        prune_cache_to_fit(&mut cache, required_size, &mut result).await;
    }

    let _low_watermark_mb = config.low_watermark_mb;
    let _total_tcgs = entries.len();

    if !result.sandbox_ids_removed.is_empty() {
        let mut sandbox_tracker = sandbox_tracker.lock().await;
        sandbox_tracker.record_many(&result.sandbox_ids_removed);
    }

    result
}

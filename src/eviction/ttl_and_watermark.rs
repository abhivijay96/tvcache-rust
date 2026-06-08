use crate::{
    eviction::{EvictionPolicyConfig, EvictionResult},
    tcg::Node,
    tvcache::TVCache,
    types::{CacheSharedObject, CacheSnapshotObject},
    utils,
};

fn get_removed_count(node: &Node) -> EvictionResult {
    let mut result = EvictionResult {
        nodes_removed: 1,
        sandboxes_removed: if node.sandbox_id.is_some() { 1 } else { 0 },
        size_removed_in_bytes: node.size_in_bytes as u64,
    };

    // Node will not be none because it is not optional, if there is a key, there is a node

    for (_, child) in node.children.iter() {
        let child_count = get_removed_count(child);
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
            let eviction_result_child = get_removed_count(child);

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

fn prune_cache_to_fit(
    _cache: &mut TVCache,
    _required_size: u64,
    _eviction_result: &mut EvictionResult,
) {
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

        result.update(tcg_result);
    }

    let high_watermark_size_in_bytes = get_high_watermark_size_in_bytes(config);
    let total_cache_size_in_bytes = get_total_cache_size_in_bytes(&entries).await;

    if total_cache_size_in_bytes > high_watermark_size_in_bytes {
        let required_size = total_cache_size_in_bytes - high_watermark_size_in_bytes;
        let mut cache = cache.lock().await;
        prune_cache_to_fit(&mut cache, required_size, &mut result);
    }

    let _low_watermark_mb = config.low_watermark_mb;
    let _total_tcgs = entries.len();

    result
}

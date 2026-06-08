use crate::{
    eviction::{EvictionPolicyConfig, EvictionResult},
    tcg::Node,
    types::CacheSnapshotObject,
    utils,
};

fn get_removed_count(node: &Node) -> EvictionResult {
    let mut result = EvictionResult {
        nodes_removed: 1,
        sandboxes_removed: if node.sandbox_id.is_some() { 1 } else { 0 },
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

pub async fn enforce(
    entries: CacheSnapshotObject,
    config: &EvictionPolicyConfig,
) -> EvictionResult {
    let mut result = EvictionResult::default();

    for (_task_id, tcg) in &entries {
        let mut tcg = tcg.lock().await;
        let mut tcg_result = EvictionResult::default();

        prune_expired_recursive(&mut tcg.root, &mut tcg_result);

        tcg.number_of_nodes -= tcg_result.nodes_removed;
        tcg.number_of_sandboxes -= tcg_result.sandboxes_removed;

        result.update(tcg_result);
    }

    let _high_watermark_mb = config.high_watermark_mb;
    let _low_watermark_mb = config.low_watermark_mb;
    let _total_tcgs = entries.len();

    result
}

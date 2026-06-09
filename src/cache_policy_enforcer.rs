use std::time::Duration;

use tracing::info;

use crate::{
    eviction::{ttl_and_watermark, EvictionPolicyConfig, EvictionPolicyName, EvictionResult},
    types::{CacheSharedObject, SandboxTrackerSharedObject, TcgSharedObject},
};

#[derive(Debug)]
struct TcgSizeLogEntry {
    task_id: String,
    size_in_bytes: u64,
    number_of_nodes: u32,
    number_of_sandboxes: u32,
}

pub async fn run_eviction_loop(
    cache: CacheSharedObject,
    config: EvictionPolicyConfig,
    sandbox_tracker: SandboxTrackerSharedObject,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));

    loop {
        interval.tick().await;

        let entries = {
            let cache = cache.lock().await;
            cache.entries_snapshot()
        };
        let total_tcgs = entries.len();
        let tcg_sizes = get_tcg_size_log_entries(entries).await;
        let total_cache_size_in_bytes: u64 =
            tcg_sizes.iter().map(|entry| entry.size_in_bytes).sum();

        log_cache_size_summary(&config, total_cache_size_in_bytes, total_tcgs);

        for tcg_size in &tcg_sizes {
            info!(
                task_id = %tcg_size.task_id,
                size_in_bytes = tcg_size.size_in_bytes,
                number_of_nodes = tcg_size.number_of_nodes,
                number_of_sandboxes = tcg_size.number_of_sandboxes,
                "tcg size before eviction enforcement"
            );
        }

        let result = enforce_policy(&config, cache.clone(), sandbox_tracker.clone()).await;

        info!(
            nodes_removed = result.nodes_removed,
            sandboxes_removed = result.sandboxes_removed,
            size_removed_in_bytes = result.size_removed_in_bytes,
            sandbox_ids_removed = result.sandbox_ids_removed.len(),
            "eviction loop completed"
        );
    }
}

async fn get_tcg_size_log_entries(entries: Vec<(String, TcgSharedObject)>) -> Vec<TcgSizeLogEntry> {
    let mut result = Vec::with_capacity(entries.len());

    for (task_id, tcg) in entries {
        let tcg = tcg.lock().await;
        result.push(TcgSizeLogEntry {
            task_id,
            size_in_bytes: tcg.size_in_bytes,
            number_of_nodes: tcg.number_of_nodes,
            number_of_sandboxes: tcg.number_of_sandboxes,
        });
    }

    result
}

fn log_cache_size_summary(
    config: &EvictionPolicyConfig,
    total_cache_size_in_bytes: u64,
    total_tcgs: usize,
) {
    info!(
        policy_name = ?config.policy_name,
        total_cache_size_in_bytes,
        high_watermark_bytes = config.high_watermark_mb as u64 * 1024 * 1024,
        low_watermark_bytes = config.low_watermark_mb as u64 * 1024 * 1024,
        total_tcgs,
        "cache size before eviction enforcement"
    );
}

pub async fn enforce_policy(
    config: &EvictionPolicyConfig,
    cache: CacheSharedObject,
    sandbox_tracker: SandboxTrackerSharedObject,
) -> EvictionResult {
    match &config.policy_name {
        EvictionPolicyName::None => EvictionResult::default(),
        EvictionPolicyName::TtlAndWatermark => {
            ttl_and_watermark::enforce(cache, config, sandbox_tracker).await
        }
    }
}

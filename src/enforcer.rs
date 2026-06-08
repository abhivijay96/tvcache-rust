use std::time::Duration;

use crate::{
    eviction::{ttl_and_watermark, EvictionPolicyConfig, EvictionPolicyName, EvictionResult},
    types::{CacheSharedObject, CacheSnapshotObject},
};

pub async fn run_eviction_loop(cache: CacheSharedObject, config: EvictionPolicyConfig) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    loop {
        interval.tick().await;

        // Taking the snapshot is an optimization to avoid holding the lock on the entire cache. We might end up doing work on tcgs that might be removed later but that is okay for performance gains. and when the eviction completes and the tcg has been removed, the Arc counting of the TCG will track it and drop it.
        let snapshot = {
            let cache = cache.lock().await;
            cache.entries_snapshot()
        };

        enforce_policy(&config, snapshot).await;
    }
}

pub async fn enforce_policy(
    config: &EvictionPolicyConfig,
    snapshot: CacheSnapshotObject,
) -> EvictionResult {
    match &config.policy_name {
        EvictionPolicyName::None => EvictionResult::default(),
        EvictionPolicyName::TtlAndWatermark => ttl_and_watermark::enforce(snapshot, config).await,
    }
}

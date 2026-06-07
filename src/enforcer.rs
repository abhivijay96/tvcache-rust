use std::time::Duration;

use crate::{
    eviction::{ttl_and_watermark, EvictionPolicyConfig, EvictionResult},
    types::{CacheSharedObject, CacheSnapshotObject},
    utils,
};

pub async fn run_eviction_loop(cache: CacheSharedObject) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    loop {
        interval.tick().await;

        let now_seconds = utils::current_utc_seconds();
        // Taking the snapshot is an optimization to avoid holding the lock on the entire cache. We might end up doing work on tcgs that might be removed later but that is okay for performance gains. and when the eviction completes and the tcg has been removed, the Arc counting of the TCG will track it and drop it.
        let (snapshot, config) = {
            let cache = cache.lock().await;
            (
                cache.entries_snapshot(),
                cache.eviction_policy_config.clone(),
            )
        };

        enforce_policy(config, snapshot, now_seconds).await;
    }
}

pub async fn enforce_policy(
    config: EvictionPolicyConfig,
    snapshot: CacheSnapshotObject,
    now_seconds: u64,
) -> EvictionResult {
    match config {
        EvictionPolicyConfig::None => EvictionResult::default(),
        EvictionPolicyConfig::TtlAndWatermark(config) => {
            ttl_and_watermark::enforce(snapshot, config, now_seconds).await
        }
    }
}

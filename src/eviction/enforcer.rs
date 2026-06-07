use std::time::Duration;

use crate::{
    eviction::{ttl_and_watermark, EvictionPolicyConfig, EvictionResult},
    tvcache::EvictionEntriesSnapshot,
    types::CacheSharedObject,
    utils,
};

pub async fn run_eviction_loop(cache: CacheSharedObject) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    loop {
        interval.tick().await;

        let now_seconds = utils::current_utc_seconds();
        let snapshot = {
            let cache = cache.lock().await;
            cache.eviction_entries_snapshot()
        };

        enforce_policy(snapshot, now_seconds).await;
    }
}

pub async fn enforce_policy(snapshot: EvictionEntriesSnapshot, now_seconds: u64) -> EvictionResult {
    match snapshot.policy {
        EvictionPolicyConfig::None => EvictionResult::default(),
        EvictionPolicyConfig::TtlAndWatermark(config) => {
            ttl_and_watermark::enforce(snapshot.entries, config, now_seconds).await
        }
    }
}

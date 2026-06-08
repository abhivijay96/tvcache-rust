use std::time::Duration;

use crate::{
    eviction::{ttl_and_watermark, EvictionPolicyConfig, EvictionPolicyName, EvictionResult},
    types::CacheSharedObject,
};

pub async fn run_eviction_loop(cache: CacheSharedObject, config: EvictionPolicyConfig) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));

    loop {
        interval.tick().await;

        enforce_policy(&config, cache.clone()).await;
    }
}

pub async fn enforce_policy(
    config: &EvictionPolicyConfig,
    cache: CacheSharedObject,
) -> EvictionResult {
    match &config.policy_name {
        EvictionPolicyName::None => EvictionResult::default(),
        EvictionPolicyName::TtlAndWatermark => ttl_and_watermark::enforce(cache, config).await,
    }
}

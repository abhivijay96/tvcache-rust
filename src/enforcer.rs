use std::time::Duration;

use crate::{
    eviction::{ttl_and_watermark, EvictionPolicyConfig, EvictionPolicyName, EvictionResult},
    types::{CacheSharedObject, SandboxTrackerSharedObject},
};

pub async fn run_eviction_loop(
    cache: CacheSharedObject,
    config: EvictionPolicyConfig,
    sandbox_tracker: SandboxTrackerSharedObject,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));

    loop {
        interval.tick().await;

        enforce_policy(&config, cache.clone(), sandbox_tracker.clone()).await;
    }
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

use crate::{
    eviction::{EvictionResult, TtlAndWatermarkEvictionConfig},
    types::TcgSharedObject,
};

pub async fn enforce(
    entries: Vec<(String, TcgSharedObject)>,
    config: TtlAndWatermarkEvictionConfig,
    now_seconds: u64,
) -> EvictionResult {
    let mut result = EvictionResult::default();

    for (_task_id, tcg) in &entries {
        let mut tcg = tcg.lock().await;
        result.merge(tcg.prune_expired(now_seconds));
    }

    let _high_watermark_mb = config.high_watermark_mb;
    let _low_watermark_mb = config.low_watermark_mb;
    let _total_tcgs = entries.len();

    result
}

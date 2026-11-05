//! Naive weighted sum of bounded retention features; evict the lowest score.
use super::*;

/// Each input is in [0, 1], with fixed scales documented in docs/linear-policy.md.
/// Missing fields in a JSON config use defaults; unknown fields are errors.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LinearWeights {
    pub frequency: f64,
    pub recency: f64,
    pub runtime: f64,
    pub dollar_cost: f64,
    pub inverse_size: f64,
    pub depth: f64,
    pub branching: f64,
    pub subtree_frequency: f64,
    pub subtree_recency: f64,
}

impl Default for LinearWeights {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            recency: 1.0,
            runtime: 1.0,
            dollar_cost: 1.0,
            inverse_size: 1.0,
            depth: 1.0,
            branching: 1.0,
            subtree_frequency: 1.0,
            subtree_recency: 1.0,
        }
    }
}

#[derive(Debug, Default)]
pub struct LinearPolicy {
    weights: LinearWeights,
}

fn bounded(value: f64, scale: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        1.0 - scale / (scale + value)
    } else {
        0.0
    }
}

fn recent(age: Option<u64>) -> f64 {
    age.map_or(0.0, |age| 100.0 / (100.0 + age as f64))
}

impl LinearPolicy {
    pub fn new(weights: LinearWeights) -> Result<Self, &'static str> {
        let values = [
            weights.frequency,
            weights.recency,
            weights.runtime,
            weights.dollar_cost,
            weights.inverse_size,
            weights.depth,
            weights.branching,
            weights.subtree_frequency,
            weights.subtree_recency,
        ];
        if values.iter().any(|v| !v.is_finite())
            || !values.iter().map(|v| v.abs()).sum::<f64>().is_finite()
        {
            return Err("linear weights and their absolute sum must be finite");
        }
        Ok(Self { weights })
    }

    /// Root statistics are a proxy for an action's benefit, not a sum of all
    /// descendant runtimes/dollars. Larger scores mean retain longer.
    pub fn score(&self, candidate: &EvictionCandidate<'_>) -> f64 {
        let f = candidate.features;
        let e = candidate.root;
        let w = &self.weights;
        w.frequency * bounded(f.access_count_before as f64, 1.0)
            + w.recency * recent(f.events_since_last_access)
            + w.runtime * bounded(e.runtime_ms.unwrap_or(0.0), 1000.0)
            + w.dollar_cost * bounded(e.tool_cost_usd.unwrap_or(0.0), 0.001)
            + w.inverse_size * (1024.0 / (1024.0 + candidate.reclaimable.resident_bytes as f64))
            + w.depth * bounded(e.depth as f64, 1.0)
            + w.branching * bounded(f.observed_children_before as f64, 1.0)
            + w.subtree_frequency * bounded(f.subtree_access_count_before as f64, 1.0)
            + w.subtree_recency * recent(f.events_since_subtree_access)
    }
}

impl CachePolicy for LinearPolicy {
    fn name(&self) -> &str {
        "linear"
    }
    fn parameters(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(&self.weights).expect("validated finite weights"))
    }
    fn on_access(
        &mut self,
        _: &EntryMeta,
        _: &AccessFeatures,
        _: AccessOutcome,
        _: &AccessContext,
    ) {
    }
    fn admission(
        &mut self,
        _: &EntryMeta,
        charge: MemoryCharge,
        ctx: &CacheContext,
    ) -> AdmissionDecision {
        admit_if_fits(charge, ctx)
    }
    fn choose_victim(
        &mut self,
        _: Option<&EntryMeta>,
        candidates: &[EvictionCandidate<'_>],
        _: &CacheContext,
    ) -> Option<usize> {
        candidates
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                self.score(a)
                    .total_cmp(&self.score(b))
                    .then_with(|| a.root.id.cmp(&b.root.id))
            })
            .map(|(i, _)| i)
    }
    fn on_insert(&mut self, _: &EntryMeta, _: &CacheContext) {}
    fn on_remove(&mut self, _: &EntryId, _: RemovalReason, _: &CacheContext) {}
}

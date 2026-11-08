//! Feed-forward retention scorer loaded from a JSON export (`tvcache_learn`).
//! Scores every legal action and evicts the lowest, like `LinearPolicy`.
//! The forward pass is hand-written so the server needs no ML runtime.
use super::{features::FeatureSet, *};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Relu,
    Tanh,
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// Row-major `[outputs][inputs]`, as in `torch.nn.Linear.weight`.
    pub weight: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub activation: Activation,
}

/// How the network output becomes a retention score (higher = keep longer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Output {
    /// Use the output directly, e.g. a reuse logit or a negated log distance.
    Score,
    /// sigmoid(output) * root runtime_ms / reclaimable bytes: expected time
    /// saved per byte if the output is a reuse logit.
    ProbBenefitPerByte,
    /// expm1(output) / reclaimable bytes: if the output predicts log1p of the
    /// tool time an eviction would lose, this is time lost per byte freed.
    ExpPerByte,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlpModel {
    pub format: String,
    pub name: String,
    pub features: FeatureSet,
    pub feature_names: Vec<String>,
    pub input_mean: Vec<f64>,
    pub input_std: Vec<f64>,
    pub layers: Vec<Layer>,
    pub output: Output,
    /// Choose an action that removes the incoming entry only when nothing else
    /// is legal, as `ClassicPolicy` does.
    #[serde(default)]
    pub protect_incoming: bool,
    /// Score at most this many candidates per eviction, sampled deterministically
    /// (LRB-style), to bound policy time on large caches. None scores all.
    #[serde(default)]
    pub max_candidates: Option<usize>,
    /// Free-form training provenance; ignored by the policy.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

pub const FORMAT: &str = "tvcache-mlp-v1";

impl MlpModel {
    pub fn validate(&self) -> Result<(), String> {
        if self.format != FORMAT {
            return Err(format!("expected format {FORMAT}"));
        }
        let names = self.features.names();
        if self.feature_names.len() != names.len()
            || self.feature_names.iter().zip(names).any(|(a, b)| a != b)
        {
            return Err("feature_names do not match this build's feature set".into());
        }
        let mut width = names.len();
        if self.input_mean.len() != width || self.input_std.len() != width {
            return Err("input_mean/input_std length must equal the feature count".into());
        }
        if self.input_std.iter().any(|s| !(s.is_finite() && *s > 0.0))
            || self.input_mean.iter().any(|m| !m.is_finite())
        {
            return Err("input_std must be positive and finite, input_mean finite".into());
        }
        if self.max_candidates == Some(0) {
            return Err("max_candidates must be at least 1".into());
        }
        if self.layers.is_empty() {
            return Err("model needs at least one layer".into());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.weight.is_empty() || layer.bias.len() != layer.weight.len() {
                return Err(format!("layer {i}: bias length must equal output rows"));
            }
            if layer.weight.iter().any(|row| row.len() != width) {
                return Err(format!("layer {i}: expected {width} inputs per row"));
            }
            if layer.weight.iter().flatten().chain(&layer.bias).any(|v| !v.is_finite()) {
                return Err(format!("layer {i}: nonfinite parameter"));
            }
            width = layer.weight.len();
        }
        if width != 1 {
            return Err("final layer must have one output".into());
        }
        Ok(())
    }

    pub fn forward(&self, input: &[f64]) -> f64 {
        let mut x: Vec<f64> = input
            .iter()
            .zip(&self.input_mean)
            .zip(&self.input_std)
            .map(|((v, m), s)| (v - m) / s)
            .collect();
        for layer in &self.layers {
            x = layer
                .weight
                .iter()
                .zip(&layer.bias)
                .map(|(row, b)| {
                    let z = row.iter().zip(&x).fold(*b, |acc, (w, v)| acc + w * v);
                    match layer.activation {
                        Activation::Relu => z.max(0.0),
                        Activation::Tanh => z.tanh(),
                        Activation::None => z,
                    }
                })
                .collect();
        }
        x[0]
    }
}

/// k distinct indices from 0..n, a pure function of (n, k, seed), so a replay
/// is reproducible. Uses a partial Fisher-Yates shuffle with xorshift64*.
fn sample(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut state = (seed ^ (n as u64).rotate_left(32)) | 1;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545f4914f6cdd1d)
    };
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (next() % (n - i) as u64) as usize;
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

#[derive(Debug)]
pub struct MlpPolicy {
    model: MlpModel,
}

impl MlpPolicy {
    pub fn new(model: MlpModel) -> Result<Self, String> {
        model.validate()?;
        Ok(Self { model })
    }

    pub fn from_json(reader: impl std::io::Read) -> Result<Self, String> {
        Self::new(serde_json::from_reader(reader).map_err(|e| e.to_string())?)
    }

    pub fn score(&self, candidate: &EvictionCandidate<'_>, context: &CacheContext) -> f64 {
        let raw = self
            .model
            .forward(&self.model.features.extract(candidate, context));
        match self.model.output {
            Output::Score => raw,
            Output::ProbBenefitPerByte => {
                let p = 1.0 / (1.0 + (-raw).exp());
                let runtime = candidate.root.runtime_ms.unwrap_or(0.0).max(1e-3);
                p * runtime / candidate.reclaimable.resident_bytes.max(1) as f64
            }
            Output::ExpPerByte => {
                raw.exp_m1().max(0.0) / candidate.reclaimable.resident_bytes.max(1) as f64
            }
        }
    }
}

impl CachePolicy for MlpPolicy {
    fn name(&self) -> &str {
        &self.model.name
    }
    fn parameters(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "features": self.model.features,
            "layers": self.model.layers.iter().map(|l| l.weight.len()).collect::<Vec<_>>(),
            "output": self.model.output,
            "protect_incoming": self.model.protect_incoming,
            "max_candidates": self.model.max_candidates,
            "metadata": self.model.metadata,
        }))
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
        incoming: Option<&EntryMeta>,
        candidates: &[EvictionCandidate<'_>],
        ctx: &CacheContext,
    ) -> Option<usize> {
        let pool: Vec<usize> = match self.model.max_candidates {
            Some(k) if candidates.len() > k => sample(candidates.len(), k, ctx.access.event_index),
            _ => (0..candidates.len()).collect(),
        };
        pool.into_iter()
            .map(|i| (i, &candidates[i]))
            .map(|(i, c)| {
                let removes_incoming = self.model.protect_incoming
                    && incoming.is_some_and(|e| c.root.id == e.id || c.affected.contains(&e.id));
                (removes_incoming, self.score(c, ctx), &c.root.id, i)
            })
            .min_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| a.1.total_cmp(&b.1))
                    .then_with(|| a.2.cmp(b.2))
            })
            .map(|(.., i)| i)
    }
    fn on_insert(&mut self, _: &EntryMeta, _: &CacheContext) {}
    fn on_remove(&mut self, _: &EntryId, _: RemovalReason, _: &CacheContext) {}
}

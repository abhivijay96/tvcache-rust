//! Feature vectors for learned policies. The candidate dump and the serving
//! policies both call these functions, so training and inference see identical
//! inputs. Python never recomputes features; it reads them from the dump.
use super::*;

/// The nine bounded features of `LinearPolicy`, in its weight order.
pub const BASIC_NAMES: [&str; 9] = [
    "frequency",
    "recency",
    "runtime",
    "dollar_cost",
    "inverse_size",
    "depth",
    "branching",
    "subtree_frequency",
    "subtree_recency",
];

const TOOL_BUCKETS: usize = 8;

/// `BASIC_NAMES` followed by raw log-scaled history, cost, size, structure,
/// cache-pressure and hashed tool features. Append only: models record the
/// length they were trained on and reject a mismatch.
pub const RICH_NAMES: [&str; 40] = [
    "frequency",
    "recency",
    "runtime",
    "dollar_cost",
    "inverse_size",
    "depth",
    "branching",
    "subtree_frequency",
    "subtree_recency",
    "log_access_count",
    "recency_missing",
    "log_events_since_access",
    "log_ms_since_access",
    "log_subtree_access_count",
    "subtree_recency_missing",
    "log_events_since_subtree_access",
    "raw_depth",
    "log_children",
    "log_parent_children",
    "runtime_missing",
    "log_runtime_ms",
    "cost_missing",
    "log_cost_micro_usd",
    "log_entry_bytes",
    "log_value_bytes",
    "log_reclaimable_bytes",
    "log_affected_entries",
    "mutates_state",
    "has_snapshot",
    "log_snapshot_bytes",
    "reclaimable_fraction_of_capacity",
    "used_fraction_of_capacity",
    "tool_bucket_0",
    "tool_bucket_1",
    "tool_bucket_2",
    "tool_bucket_3",
    "tool_bucket_4",
    "tool_bucket_5",
    "tool_bucket_6",
    "tool_bucket_7",
];

/// Maps [0, inf) to [0, 1); invalid or negative values contribute zero.
pub fn bounded(value: f64, scale: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        1.0 - scale / (scale + value)
    } else {
        0.0
    }
}

fn recent(age: Option<u64>) -> f64 {
    age.map_or(0.0, |age| 100.0 / (100.0 + age as f64))
}

fn log1p(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.ln_1p()
    } else {
        0.0
    }
}

fn flag(condition: bool) -> f64 {
    if condition {
        1.0
    } else {
        0.0
    }
}

fn fraction(part: u64, whole: u64) -> f64 {
    if whole == 0 || whole == u64::MAX {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// FNV-1a, so buckets are stable across processes and platforms.
fn tool_bucket(tool: &str) -> usize {
    let hash = tool.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    });
    (hash % TOOL_BUCKETS as u64) as usize
}

pub fn basic(candidate: &EvictionCandidate<'_>) -> [f64; 9] {
    let f = candidate.features;
    let e = candidate.root;
    [
        bounded(f.access_count_before as f64, 1.0),
        recent(f.events_since_last_access),
        bounded(e.runtime_ms.unwrap_or(0.0), 1000.0),
        bounded(e.tool_cost_usd.unwrap_or(0.0), 0.001),
        1024.0 / (1024.0 + candidate.reclaimable.resident_bytes as f64),
        bounded(e.depth as f64, 1.0),
        bounded(f.observed_children_before as f64, 1.0),
        bounded(f.subtree_access_count_before as f64, 1.0),
        recent(f.events_since_subtree_access),
    ]
}

pub fn rich(candidate: &EvictionCandidate<'_>, context: &CacheContext) -> Vec<f64> {
    let f = candidate.features;
    let e = candidate.root;
    let mut x = Vec::with_capacity(RICH_NAMES.len());
    x.extend(basic(candidate));
    x.extend([
        log1p(f.access_count_before as f64),
        flag(f.events_since_last_access.is_none()),
        log1p(f.events_since_last_access.unwrap_or(0) as f64),
        log1p(f.time_since_last_access_ms.unwrap_or(0.0)),
        log1p(f.subtree_access_count_before as f64),
        flag(f.events_since_subtree_access.is_none()),
        log1p(f.events_since_subtree_access.unwrap_or(0) as f64),
        e.depth as f64,
        log1p(f.observed_children_before as f64),
        log1p(f.observed_parent_children_before as f64),
        flag(e.runtime_ms.is_none()),
        log1p(e.runtime_ms.unwrap_or(0.0)),
        flag(e.tool_cost_usd.is_none()),
        log1p(e.tool_cost_usd.unwrap_or(0.0) * 1e6),
        log1p(e.cache_entry_bytes as f64),
        log1p(e.value_bytes as f64),
        log1p(candidate.reclaimable.resident_bytes as f64),
        log1p(candidate.affected.len() as f64),
        flag(e.mutates_state),
        flag(e.snapshot.is_some()),
        log1p(e.snapshot.as_ref().map_or(0.0, |s| s.external_bytes as f64)),
        fraction(candidate.reclaimable.resident_bytes, context.capacity.resident_bytes),
        fraction(context.used.resident_bytes, context.capacity.resident_bytes),
    ]);
    let bucket = tool_bucket(&e.tool);
    x.extend((0..TOOL_BUCKETS).map(|i| flag(i == bucket)));
    debug_assert_eq!(x.len(), RICH_NAMES.len());
    x
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureSet {
    Basic,
    Rich,
}

impl FeatureSet {
    pub fn names(self) -> &'static [&'static str] {
        match self {
            FeatureSet::Basic => &BASIC_NAMES,
            FeatureSet::Rich => &RICH_NAMES,
        }
    }

    pub fn extract(self, candidate: &EvictionCandidate<'_>, context: &CacheContext) -> Vec<f64> {
        match self {
            FeatureSet::Basic => basic(candidate).to_vec(),
            FeatureSet::Rich => rich(candidate, context),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rich_starts_with_basic_and_matches_names() {
        let meta = EntryMeta {
            id: EntryId {
                task_id: "t".into(),
                node_id: "n".into(),
            },
            parent: None,
            path: vec!["a".into()],
            depth: 1,
            mutates_state: true,
            tool: "shell".into(),
            cache_entry_bytes: 2048,
            value_bytes: 1024,
            runtime_ms: Some(50.0),
            tool_cost_usd: None,
            snapshot: None,
        };
        let features = AccessFeatures {
            access_count_before: 3,
            events_since_last_access: Some(10),
            ..Default::default()
        };
        let affected = [meta.id.clone()];
        let candidate = EvictionCandidate {
            root: &meta,
            affected: &affected,
            reclaimable: MemoryCharge {
                resident_bytes: 2048,
                external_snapshot_bytes: 0,
            },
            features: &features,
        };
        let context = CacheContext {
            access: AccessContext {
                event_index: 0,
                time_ms: 0.0,
            },
            used: MemoryCharge {
                resident_bytes: 4096,
                external_snapshot_bytes: 0,
            },
            capacity: MemoryCharge {
                resident_bytes: 8192,
                external_snapshot_bytes: u64::MAX,
            },
        };
        let x = rich(&candidate, &context);
        assert_eq!(x.len(), RICH_NAMES.len());
        assert_eq!(&x[..9], &basic(&candidate));
        assert!(x.iter().all(|v| v.is_finite()));
        assert_eq!(x[30], 0.25);
        assert_eq!(x[31], 0.5);
        assert_eq!(x[32..].iter().sum::<f64>(), 1.0);
        assert_eq!(tool_bucket("shell"), tool_bucket("shell"));
    }
}

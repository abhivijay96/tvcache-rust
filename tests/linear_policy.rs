use tvcache::policy::{linear::*, *};

fn entry(id: &str) -> EntryMeta {
    EntryMeta {
        id: EntryId {
            task_id: "t".into(),
            node_id: id.into(),
        },
        parent: None,
        path: vec![id.into()],
        depth: 1,
        mutates_state: false,
        tool: "tool".into(),
        cache_entry_bytes: 1024,
        value_bytes: 1024,
        runtime_ms: None,
        tool_cost_usd: None,
        snapshot: None,
    }
}
fn zero_weights() -> LinearWeights {
    LinearWeights {
        frequency: 0.0,
        recency: 0.0,
        runtime: 0.0,
        dollar_cost: 0.0,
        inverse_size: 0.0,
        depth: 0.0,
        branching: 0.0,
        subtree_frequency: 0.0,
        subtree_recency: 0.0,
    }
}
fn ctx() -> CacheContext {
    CacheContext {
        access: AccessContext {
            event_index: 100,
            time_ms: 100.0,
        },
        used: MemoryCharge::default(),
        capacity: MemoryCharge {
            resident_bytes: 1024,
            external_snapshot_bytes: 0,
        },
    }
}
fn candidate<'a>(root: &'a EntryMeta, features: &'a AccessFeatures) -> EvictionCandidate<'a> {
    EvictionCandidate {
        root,
        features,
        affected: &[],
        reclaimable: MemoryCharge {
            resident_bytes: root.cache_entry_bytes,
            external_snapshot_bytes: 0,
        },
    }
}

#[test]
fn each_weight_changes_score_in_the_expected_direction() {
    for field in [
        "frequency",
        "recency",
        "runtime",
        "dollar_cost",
        "inverse_size",
        "depth",
        "branching",
        "subtree_frequency",
        "subtree_recency",
    ] {
        let mut weights = serde_json::to_value(zero_weights()).unwrap();
        weights[field] = 1.0.into();
        let policy = LinearPolicy::new(serde_json::from_value(weights).unwrap()).unwrap();
        let low = entry("a");
        let mut high = low.clone();
        let lo = AccessFeatures::default();
        let mut hi = lo.clone();
        match field {
            "frequency" => hi.access_count_before = 10,
            "recency" => hi.events_since_last_access = Some(0),
            "runtime" => high.runtime_ms = Some(1000.0),
            "dollar_cost" => high.tool_cost_usd = Some(0.001),
            "inverse_size" => high.cache_entry_bytes = 512,
            "depth" => high.depth = 10,
            "branching" => hi.observed_children_before = 3,
            "subtree_frequency" => hi.subtree_access_count_before = 10,
            "subtree_recency" => hi.events_since_subtree_access = Some(0),
            _ => unreachable!(),
        }
        assert!(
            policy.score(&candidate(&high, &hi)) > policy.score(&candidate(&low, &lo)),
            "{field}"
        );
    }
}

#[test]
fn weighted_sum_and_victim_direction_are_explicit() {
    let mut weights = zero_weights();
    weights.frequency = 2.0;
    weights.runtime = 3.0;
    let mut policy = LinearPolicy::new(weights).unwrap();
    let a = entry("a");
    let mut b = entry("b");
    b.runtime_ms = Some(1000.0);
    let cold = AccessFeatures::default();
    let hot = AccessFeatures {
        access_count_before: 1,
        ..Default::default()
    };
    let candidates = [candidate(&b, &hot), candidate(&a, &cold)];
    assert_eq!(policy.score(&candidates[0]), 2.5);
    assert_eq!(policy.choose_victim(None, &candidates, &ctx()), Some(1));
    assert_eq!(policy.choose_victim(None, &[], &ctx()), None);
}

#[test]
fn ties_use_stable_ids_and_nonfinite_weights_are_rejected() {
    let mut policy = LinearPolicy::new(zero_weights()).unwrap();
    let a = entry("a");
    let b = entry("b");
    let f = AccessFeatures::default();
    assert_eq!(
        policy.choose_victim(None, &[candidate(&b, &f), candidate(&a, &f)], &ctx()),
        Some(1)
    );
    let mut w = zero_weights();
    w.runtime = f64::NAN;
    assert!(LinearPolicy::new(w).is_err());
    assert!(serde_json::from_str::<LinearWeights>(r#"{"typo":1}"#).is_err());
    assert_eq!(
        policy.admission(
            &a,
            MemoryCharge {
                resident_bytes: 1025,
                external_snapshot_bytes: 0
            },
            &ctx()
        ),
        AdmissionDecision::Bypass
    );
}

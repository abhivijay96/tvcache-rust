use tvcache::{
    policy::{fifo::FifoPolicy, synthetic::*, CachePolicy},
    replay::replay,
};

fn fixture(sequence: &[&str], bytes: u64) -> (TraceCatalog, Vec<AccessRecord>) {
    let mut nodes = String::new();
    for (i, id) in ["a", "b"].iter().enumerate() {
        let first = sequence.iter().position(|s| s == id).unwrap_or(i);
        nodes.push_str(
            &serde_json::json!({
                "task_id": "t", "node_id": id, "parent_id": null,
                "key": [id], "depth": 1, "first_seen_event": first,
                "cache_entry_bytes": bytes, "value_bytes": bytes,
                "runtime_ms": 100.0, "tool_cost_usd": 0.1,
                "snapshot_candidate": null,
                "tool_call": {"tool": id, "mutates_state": false}
            })
            .to_string(),
        );
        nodes.push('\n');
    }
    let events = sequence
        .iter()
        .enumerate()
        .map(|(i, id)| AccessRecord {
            event: "access".into(),
            event_index: i as u64,
            time_ms: i as f64,
            task_id: "t".into(),
            node_id: (*id).into(),
            rollout_id: "r".into(),
            features: Default::default(),
        })
        .collect();
    (TraceCatalog::read(nodes.as_bytes()).unwrap(), events)
}

fn fifo() -> Option<Box<dyn CachePolicy>> {
    Some(Box::new(FifoPolicy::default()))
}

#[tokio::test]
async fn original_retains_hot_entry_while_fifo_evicts_oldest() {
    let (catalog, events) = fixture(&["a", "a", "b", "a", "b"], 600_000);
    let original = replay(&catalog, &events, 1, None).await.unwrap();
    let fifo = replay(&catalog, &events, 1, fifo()).await.unwrap();
    assert_eq!(original.hits, 2);
    assert_eq!(fifo.hits, 1);
    assert_eq!(original.gross_tool_ms_saved, 200.0);
    assert_eq!(fifo.gross_tool_ms_saved, 100.0);
    assert_eq!(original.evicted_entries, 2);
    assert_eq!(fifo.evicted_entries, 3);
    for report in [original, fifo] {
        assert_eq!(report.requests, report.hits + report.misses);
        assert_eq!(report.final_bytes, 600_000);
        assert!(report.peak_bytes_after_enforcement <= report.capacity_bytes);
        assert_eq!(report.peak_bytes_before_enforcement, 1_200_000);
    }
}

#[tokio::test]
async fn zero_budget_and_unlimited_controls() {
    let (catalog, events) = fixture(&["a", "a", "b", "a", "b"], 600_000);
    for policy in [None, fifo()] {
        let report = replay(&catalog, &events, 0, policy).await.unwrap();
        assert_eq!(report.hits, 0);
        assert_eq!(report.bypassed_oversized, 5);
        assert_eq!(report.final_bytes, 0);
    }
    for policy in [None, fifo()] {
        let report = replay(&catalog, &events, 2, policy).await.unwrap();
        assert_eq!(report.hits, 3);
        assert_eq!(report.evicted_entries, 0);
        assert_eq!(report.final_bytes, 1_200_000);
    }
}

#[tokio::test]
async fn rejects_out_of_order_requests() {
    let (catalog, mut events) = fixture(&["a", "a", "b"], 64);
    events.swap(0, 1);
    assert!(replay(&catalog, &events, 1, fifo()).await.is_err());
}

#[tokio::test]
async fn linear_uses_recomputed_history_and_records_weights() {
    use tvcache::policy::linear::{LinearPolicy, LinearWeights};
    let (catalog, events) = fixture(&["a", "a", "b", "a", "b"], 600_000);
    // The fixture events contain empty/default features: replay must compute them.
    let weights = LinearWeights {
        frequency: 1.0,
        recency: 0.0,
        runtime: 0.0,
        dollar_cost: 0.0,
        inverse_size: 0.0,
        depth: 0.0,
        branching: 0.0,
        subtree_frequency: 0.0,
        subtree_recency: 0.0,
    };
    let report = replay(
        &catalog,
        &events,
        1,
        Some(Box::new(LinearPolicy::new(weights).unwrap())),
    )
    .await
    .unwrap();
    assert_eq!(report.hits, 2);
    assert_eq!(report.policy_parameters.unwrap()["frequency"], 1.0);
    assert!(report.peak_bytes_after_enforcement <= report.capacity_bytes);
    for capacity in [0, 2] {
        let report = replay(
            &catalog,
            &events,
            capacity,
            Some(Box::new(LinearPolicy::default())),
        )
        .await
        .unwrap();
        assert_eq!(report.hits, if capacity == 0 { 0 } else { 3 });
    }
}

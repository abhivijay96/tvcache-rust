use std::{
    collections::BTreeMap,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};
use tvcache::{
    policy::{classic::ClassicPolicy, fifo::FifoPolicy, synthetic::*, CachePolicy},
    replay::{check_decisions, replay, ReplayReport},
};

fn fixture(sequence: &[&str], bytes: u64) -> (TraceCatalog, Vec<AccessRecord>) {
    let mut nodes = String::new();
    let mut ids: Vec<&str> = Vec::new();
    for id in sequence {
        if !ids.contains(id) {
            ids.push(id);
        }
    }
    for id in ids {
        let first = sequence.iter().position(|s| *s == id).unwrap();
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

fn checked(report: &ReplayReport, catalog: &TraceCatalog, events: &[AccessRecord]) {
    check_decisions(report, catalog, events).unwrap();
}

fn dataset(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data/synthetic").join(name)
}

fn load(dir: &Path, nodes: impl Fn(serde_json::Value) -> serde_json::Value) -> (TraceCatalog, Vec<AccessRecord>) {
    let mut lines = String::new();
    for line in std::fs::read_to_string(dir.join("nodes.jsonl")).unwrap().lines() {
        lines.push_str(&nodes(serde_json::from_str(line).unwrap()).to_string());
        lines.push('\n');
    }
    let events = read_jsonl(BufReader::new(File::open(dir.join("events.jsonl")).unwrap())).unwrap();
    (TraceCatalog::read(lines.as_bytes()).unwrap(), events)
}

/// Every node becomes top-level, so each eviction action removes exactly one entry.
fn flat(mut node: serde_json::Value) -> serde_json::Value {
    node["key"] = serde_json::json!([node["node_id"]]);
    node["parent_id"] = serde_json::Value::Null;
    node["depth"] = 1.into();
    node
}

/// Textbook single-entry LRU/LFU, independent of the engine and ClassicPolicy: on a
/// miss, evict in key order until the newcomer fits, then admit it. Returns
/// (hit, admit, sorted victims) per event.
fn reference(
    catalog: &TraceCatalog,
    events: &[AccessRecord],
    capacity: u64,
    lfu: bool,
) -> Vec<(bool, bool, Vec<String>)> {
    let mut cache: BTreeMap<String, (u64, u64, u64)> = BTreeMap::new(); // hits, last, bytes
    let mut used = 0;
    events
        .iter()
        .map(|event| {
            let now = event.event_index;
            if let Some(entry) = cache.get_mut(&event.node_id) {
                entry.0 += 1;
                entry.1 = now;
                return (true, false, vec![]);
            }
            let bytes = catalog.resolve(event).unwrap().0.cache_entry_bytes;
            if bytes > capacity {
                return (false, false, vec![]);
            }
            let mut order: Vec<_> = cache
                .iter()
                .map(|(id, &(hits, last, _))| (if lfu { hits } else { 0 }, last, id.clone()))
                .collect();
            order.sort();
            let mut evict = vec![];
            for (_, _, id) in order {
                if used + bytes <= capacity {
                    break;
                }
                used -= cache.remove(&id).unwrap().2;
                evict.push(id);
            }
            cache.insert(event.node_id.clone(), (1, now, bytes));
            used += bytes;
            evict.sort();
            (false, true, evict)
        })
        .collect()
}

async fn assert_flat_matches_reference(name: &str) {
    let (catalog, events) = load(&dataset(name), flat);
    for mib in [1, 2, 4] {
        for (is_lfu, make) in [(false, lru as fn() -> _), (true, lfu)] {
            let report = replay(&catalog, &events, mib, make()).await.unwrap();
            checked(&report, &catalog, &events);
            let expected = reference(&catalog, &events, report.capacity_bytes, is_lfu);
            for (d, want) in report.decisions.iter().zip(&expected) {
                let mut evict = d.evict.clone();
                evict.sort();
                assert_eq!(
                    (d.hit, d.admit, evict),
                    *want,
                    "{name} {mib} MiB {}: event {}",
                    report.policy,
                    d.event_index
                );
            }
        }
    }
}

#[tokio::test]
async fn flat_balanced_matches_reference_lru_lfu() {
    assert_flat_matches_reference("balanced").await;
}

#[tokio::test]
async fn flat_popularity_shift_matches_reference_lru_lfu() {
    assert_flat_matches_reference("popularity_shift").await;
}

fn lru() -> Option<Box<dyn CachePolicy>> {
    Some(Box::new(ClassicPolicy::lru()))
}

fn lfu() -> Option<Box<dyn CachePolicy>> {
    Some(Box::new(ClassicPolicy::lfu()))
}

#[tokio::test]
async fn lfu_keeps_hot_entry_that_lru_evicts() {
    // 400 KB entries in 1 MiB: two fit. c evicts a under LRU, b under LFU.
    let (catalog, events) = fixture(&["a", "a", "a", "b", "c", "a"], 400_000);
    let lru = replay(&catalog, &events, 1, lru()).await.unwrap();
    let lfu = replay(&catalog, &events, 1, lfu()).await.unwrap();
    assert_eq!(lru.hits, 2);
    assert_eq!(lfu.hits, 3);
    for report in [&lru, &lfu] {
        checked(report, &catalog, &events);
        assert!(report.policy_calls > 0);
    }
}

#[tokio::test]
async fn lru_adapts_to_popularity_shift_while_lfu_holds_stale_entry() {
    let (catalog, events) = fixture(&["a", "a", "a", "b", "c", "b", "c", "b", "c"], 400_000);
    let lru = replay(&catalog, &events, 1, lru()).await.unwrap();
    let lfu = replay(&catalog, &events, 1, lfu()).await.unwrap();
    assert_eq!(lru.hits, 6);
    assert_eq!(lfu.hits, 2);
    for report in [&lru, &lfu] {
        checked(report, &catalog, &events);
    }
}

#[tokio::test]
async fn lru_and_lfu_hold_invariants_on_team_datasets() {
    for (name, tight, unlimited) in [
        ("balanced", 1, 16),
        ("expensive_reuse", 1, 16),
        ("popularity_shift", 1, 16),
    ] {
        let dir = dataset(name);
        let summary: serde_json::Value =
            serde_json::from_reader(File::open(dir.join("summary.json")).unwrap()).unwrap();
        let (catalog, events) = load(&dir, |node| node);
        for make in [lru, lfu] {
            let first = replay(&catalog, &events, tight, make()).await.unwrap();
            let second = replay(&catalog, &events, tight, make()).await.unwrap();
            assert_eq!(first.requests, first.hits + first.misses);
            assert!(first.peak_bytes_after_enforcement <= first.capacity_bytes);
            checked(&first, &catalog, &events);
            assert_eq!(
                serde_json::to_string(&first.decisions).unwrap(),
                serde_json::to_string(&second.decisions).unwrap(),
                "{name}: nondeterministic"
            );
            // A cache larger than every entry hits on every repeat access.
            let all = replay(&catalog, &events, unlimited, make()).await.unwrap();
            assert_eq!(all.hits, summary["repeat_accesses"].as_u64().unwrap(), "{name}");
            assert_eq!(all.evicted_entries, 0);
        }
    }
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

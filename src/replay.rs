//! Result-only replay using the actual server graph and original enforcer.
use crate::{
    eviction::{ttl_and_watermark, EvictionPolicyConfig, EvictionPolicyName},
    policy::{synthetic::*, *},
    tvcache::TVCache,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    sync::Arc,
    time::Instant,
};
use tokio::sync::Mutex;

/// One decision per event, in tvcache_eval's `Decision` format so
/// `python3 -m tvcache_eval score` can check and rescore a run.
#[derive(Debug, Serialize)]
pub struct DecisionRecord {
    pub event_index: u64,
    pub node_id: String,
    pub hit: bool,
    pub admit: bool,
    pub evict: Vec<String>,
    pub store_snapshot: bool,
}

#[derive(Debug, Serialize)]
pub struct ReplayReport {
    pub policy: String,
    pub policy_parameters: Option<serde_json::Value>,
    pub mode: &'static str,
    pub capacity_bytes: u64,
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub gross_tool_ms_saved: f64,
    pub gross_tool_cost_usd_saved: Option<f64>,
    pub evicted_entries: u64,
    pub bypassed_oversized: u64,
    pub bypassed_missing_parent: u64,
    pub bypassed_by_policy: u64,
    pub peak_bytes_before_enforcement: u64,
    pub peak_bytes_after_enforcement: u64,
    pub final_bytes: u64,
    /// Time inside policy callbacks only; engine candidate building is excluded.
    pub policy_calls: u64,
    pub policy_ns: u64,
    #[serde(skip)]
    pub decisions: Vec<DecisionRecord>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn timed<T>(report: &mut ReplayReport, call: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = call();
    report.policy_ns += start.elapsed().as_nanos() as u64;
    report.policy_calls += 1;
    result
}

/// Independent check of a run: rebuild residency from the decision log alone and
/// confirm every hit and victim was resident and the budget held after each event.
pub fn check_decisions(
    report: &ReplayReport,
    catalog: &TraceCatalog,
    events: &[AccessRecord],
) -> Result<(), String> {
    if report.decisions.len() != events.len() {
        return Err(format!("{} decisions for {} events", report.decisions.len(), events.len()));
    }
    let mut resident: BTreeMap<&str, u64> = BTreeMap::new();
    let mut used = 0u64;
    for (d, event) in report.decisions.iter().zip(events) {
        let at = d.event_index;
        if at != event.event_index || d.node_id != event.node_id {
            return Err(format!("event {at}: decision does not match trace"));
        }
        if d.hit != resident.contains_key(d.node_id.as_str()) {
            return Err(format!("event {at}: hit disagrees with residency"));
        }
        if d.hit && d.admit {
            return Err(format!("event {at}: admission on a hit"));
        }
        for victim in &d.evict {
            used -= resident
                .remove(victim.as_str())
                .ok_or_else(|| format!("event {at}: victim {victim} not resident"))?;
        }
        if d.admit {
            let bytes = catalog.resolve(event).map_err(|e| e.to_string())?.0.cache_entry_bytes;
            resident.insert(&d.node_id, bytes);
            used += bytes;
        }
        if used > report.capacity_bytes {
            return Err(format!("event {at}: {used} bytes exceeds budget"));
        }
    }
    if report.decisions.iter().filter(|d| d.hit).count() as u64 != report.hits {
        return Err("hit count disagrees with report".into());
    }
    Ok(())
}
struct Resident {
    meta: EntryMeta,
}

/// Sees every policy eviction: the legal candidates and the chosen index, before
/// the removal is applied. Used to dump training data for learned policies.
pub trait EvictionObserver {
    fn observe(
        &mut self,
        context: &CacheContext,
        incoming: &EntryMeta,
        candidates: &[EvictionCandidate<'_>],
        chosen: usize,
    ) -> io::Result<()>;
}

/// None selects the original enforcer. Both modes enforce after insertion and
/// report transient overshoot. Snapshot costs and TTLs are intentionally disabled.
pub async fn replay(
    catalog: &TraceCatalog,
    events: &[AccessRecord],
    capacity_mib: u32,
    policy: Option<Box<dyn CachePolicy>>,
) -> io::Result<ReplayReport> {
    replay_observed(catalog, events, capacity_mib, policy, None).await
}

/// `replay`, also reporting each policy eviction to `observer`. The original
/// enforcer builds no candidates, so it is never observed.
pub async fn replay_observed(
    catalog: &TraceCatalog,
    events: &[AccessRecord],
    capacity_mib: u32,
    mut policy: Option<Box<dyn CachePolicy>>,
    mut observer: Option<&mut dyn EvictionObserver>,
) -> io::Result<ReplayReport> {
    let config = EvictionPolicyConfig::new(
        EvictionPolicyName::TtlAndWatermark,
        capacity_mib,
        capacity_mib,
    );
    let capacity = capacity_mib as u64 * 1024 * 1024;
    let cache = Arc::new(Mutex::new(TVCache::new()));
    let mut residents: BTreeMap<EntryId, Resident> = BTreeMap::new();
    let mut used = 0u64;
    let mut history = history::ObservedHistory::default();
    let mut report = ReplayReport {
        policy_parameters: policy.as_ref().and_then(|p| p.parameters()),
        policy: policy
            .as_ref()
            .map_or("original-ttl-and-watermark", |p| p.name())
            .into(),
        mode:
            "result-only; no TTL/snapshots; trace entry sizes; post-insert enforcement each event",
        capacity_bytes: capacity,
        requests: 0,
        hits: 0,
        misses: 0,
        hit_rate: 0.0,
        gross_tool_ms_saved: 0.0,
        gross_tool_cost_usd_saved: Some(0.0),
        evicted_entries: 0,
        bypassed_oversized: 0,
        bypassed_missing_parent: 0,
        bypassed_by_policy: 0,
        peak_bytes_before_enforcement: 0,
        peak_bytes_after_enforcement: 0,
        final_bytes: 0,
        policy_calls: 0,
        policy_ns: 0,
        decisions: Vec::new(),
    };
    let mut previous = None;
    for event in events {
        let (entry, access) = catalog.resolve(event)?;
        if let Some((index, time)) = previous {
            if access.event_index <= index || access.time_ms < time {
                return Err(invalid(
                    "events must have increasing indices and nondecreasing times",
                ));
            }
        }
        previous = Some((access.event_index, access.time_ms));
        if entry.path.is_empty()
            || entry.cache_entry_bytes == 0
            || entry.cache_entry_bytes > u32::MAX as u64
        {
            return Err(invalid(
                "replay requires nonempty paths and entry sizes in 1..=u32::MAX",
            ));
        }
        let handle = { cache.lock().await.get_tcg(entry.id.task_id.clone()) };
        let hit = handle.lock().await.find(entry.path.clone()).0.is_some();
        let outcome = if hit {
            AccessOutcome::ResultHit
        } else {
            AccessOutcome::Miss
        };
        let features = history.features(&entry, access);
        if let Some(p) = policy.as_mut() {
            timed(&mut report, || p.on_access(&entry, &features, outcome, &access));
        }
        history.observe(&entry, access);
        report.requests += 1;
        report.decisions.push(DecisionRecord {
            event_index: access.event_index,
            node_id: entry.id.node_id.clone(),
            hit,
            admit: false,
            evict: Vec::new(),
            store_snapshot: false,
        });
        if hit {
            report.hits += 1;
            report.gross_tool_ms_saved += entry.runtime_ms.unwrap_or(0.0);
            report.gross_tool_cost_usd_saved = report
                .gross_tool_cost_usd_saved
                .zip(entry.tool_cost_usd)
                .map(|(a, b)| a + b);
            continue;
        }
        report.misses += 1;
        if entry.cache_entry_bytes > capacity {
            report.bypassed_oversized += 1;
            continue;
        }
        // Server inserts fail if an ancestor has been evicted; do not invent it.
        {
            let tcg = handle.lock().await;
            let mut node = &tcg.root;
            let mut missing = false;
            for key in &entry.path[..entry.path.len() - 1] {
                if let Some(child) = node.children.get(key) {
                    node = child;
                } else {
                    missing = true;
                    break;
                }
            }
            if missing {
                report.bypassed_missing_parent += 1;
                continue;
            }
        }
        let context = |bytes| CacheContext {
            access,
            used: MemoryCharge {
                resident_bytes: bytes,
                external_snapshot_bytes: 0,
            },
            capacity: MemoryCharge {
                resident_bytes: capacity,
                external_snapshot_bytes: u64::MAX,
            },
        };
        if let Some(p) = policy.as_mut() {
            let charge = MemoryCharge {
                resident_bytes: entry.cache_entry_bytes,
                external_snapshot_bytes: 0,
            };
            if timed(&mut report, || p.admission(&entry, charge, &context(used)))
                == AdmissionDecision::Bypass
            {
                report.bypassed_by_policy += 1;
                continue;
            }
        }
        {
            let mut tcg = handle.lock().await;
            if !tcg
                .insert(
                    entry.path.clone(),
                    Some(String::new()),
                    None,
                    0,
                    None,
                    false,
                )
                .0
            {
                return Err(invalid("unexpected insertion failure"));
            }
            let mut node = &mut tcg.root;
            for key in &entry.path {
                node = node.children.get_mut(key).unwrap();
            }
            let old = node.size_in_bytes as u64;
            node.size_in_bytes = entry.cache_entry_bytes as u32;
            tcg.size_in_bytes = tcg.size_in_bytes - old + entry.cache_entry_bytes;
        }
        used += entry.cache_entry_bytes;
        if let Some(p) = policy.as_mut() {
            timed(&mut report, || p.on_insert(&entry, &context(used)));
        }
        report.decisions.last_mut().unwrap().admit = true;
        residents.insert(
            entry.id.clone(),
            Resident {
                meta: entry.clone(),
            },
        );
        report.peak_bytes_before_enforcement = report.peak_bytes_before_enforcement.max(used);
        if let Some(p) = policy.as_mut() {
            while used > capacity {
                let roots: Vec<_> = residents.values().collect();
                let features: Vec<_> = roots
                    .iter()
                    .map(|r| history.features(&r.meta, access))
                    .collect();
                let groups: Vec<Vec<EntryId>> = roots
                    .iter()
                    .map(|root| {
                        residents
                            .values()
                            .filter(|r| {
                                r.meta.id.task_id == root.meta.id.task_id
                                    && r.meta.path.starts_with(&root.meta.path)
                            })
                            .map(|r| r.meta.id.clone())
                            .collect()
                    })
                    .collect();
                let candidates: Vec<_> = roots
                    .iter()
                    .zip(&groups)
                    .zip(&features)
                    .map(|((root, ids), features)| EvictionCandidate {
                        root: &root.meta,
                        affected: ids,
                        reclaimable: MemoryCharge {
                            resident_bytes: ids
                                .iter()
                                .map(|id| residents[id].meta.cache_entry_bytes)
                                .sum(),
                            external_snapshot_bytes: 0,
                        },
                        features,
                    })
                    .collect();
                let index = timed(&mut report, || {
                    p.choose_victim(Some(&entry), &candidates, &context(used))
                })
                .ok_or_else(|| invalid("policy declined eviction while over capacity"))?;
                let candidate = candidates
                    .get(index)
                    .ok_or_else(|| invalid("invalid victim index"))?;
                if let Some(o) = observer.as_mut() {
                    o.observe(&context(used), &entry, &candidates, index)?;
                }
                let root = candidate.root.clone();
                let affected = candidate.affected.to_vec();
                let bytes = candidate.reclaimable.resident_bytes;
                let tcg_handle = { cache.lock().await.peek_tcg(&root.id.task_id).unwrap() };
                let mut tcg = tcg_handle.lock().await;
                let mut parent = &mut tcg.root;
                for key in &root.path[..root.path.len() - 1] {
                    parent = parent.children.get_mut(key).unwrap();
                }
                parent
                    .children
                    .remove(root.path.last().unwrap())
                    .ok_or_else(|| invalid("victim disappeared"))?;
                tcg.number_of_nodes -= affected.len() as u32;
                tcg.size_in_bytes -= bytes;
                used -= bytes;
                for id in &affected {
                    residents.remove(id);
                    timed(&mut report, || {
                        p.on_remove(id, RemovalReason::Capacity, &context(used))
                    });
                    let decision = report.decisions.last_mut().unwrap();
                    decision.evict.push(id.node_id.clone());
                }
                report.evicted_entries += affected.len() as u64;
            }
        } else {
            let removed = ttl_and_watermark::enforce(Arc::clone(&cache), &config).await;
            used -= removed.size_removed_in_bytes;
            report.evicted_entries += removed.nodes_removed as u64;
            if removed.nodes_removed > 0 {
                let handles = { cache.lock().await.entries_snapshot() };
                let mut live = BTreeSet::new();
                for (task, handle) in handles {
                    let tcg = handle.lock().await;
                    for (id, resident) in &residents {
                        if id.task_id != task {
                            continue;
                        }
                        let mut node = &tcg.root;
                        let exists = resident.meta.path.iter().all(|key| {
                            if let Some(child) = node.children.get(key) {
                                node = child;
                                true
                            } else {
                                false
                            }
                        });
                        if exists {
                            live.insert(id.clone());
                        }
                    }
                }
                let decision = report.decisions.last_mut().unwrap();
                for id in residents.keys().filter(|id| !live.contains(*id)) {
                    decision.evict.push(id.node_id.clone());
                }
                residents.retain(|id, _| live.contains(id));
            }
        }
        // Evicting the just-inserted entry in the same event is, net, a bypass.
        let decision = report.decisions.last_mut().unwrap();
        if let Some(i) = decision.evict.iter().position(|id| *id == decision.node_id) {
            decision.evict.remove(i);
            decision.admit = false;
        }
        if used > capacity {
            return Err(invalid("cache exceeds capacity after enforcement"));
        }
        report.peak_bytes_after_enforcement = report.peak_bytes_after_enforcement.max(used);
    }
    report.final_bytes = used;
    report.hit_rate = if report.requests == 0 {
        0.0
    } else {
        report.hits as f64 / report.requests as f64
    };
    Ok(report)
}

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
};
use tokio::sync::Mutex;

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
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
struct Resident {
    meta: EntryMeta,
}

/// None selects the original enforcer. Both modes enforce after insertion and
/// report transient overshoot. Snapshot costs and TTLs are intentionally disabled.
pub async fn replay(
    catalog: &TraceCatalog,
    events: &[AccessRecord],
    capacity_mib: u32,
    mut policy: Option<Box<dyn CachePolicy>>,
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
            p.on_access(&entry, &features, outcome, &access);
        }
        history.observe(&entry, access);
        report.requests += 1;
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
            if p.admission(
                &entry,
                MemoryCharge {
                    resident_bytes: entry.cache_entry_bytes,
                    external_snapshot_bytes: 0,
                },
                &context(used),
            ) == AdmissionDecision::Bypass
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
            p.on_insert(&entry, &context(used));
        }
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
                let index = p
                    .choose_victim(Some(&entry), &candidates, &context(used))
                    .ok_or_else(|| invalid("policy declined eviction while over capacity"))?;
                let candidate = candidates
                    .get(index)
                    .ok_or_else(|| invalid("invalid victim index"))?;
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
                    p.on_remove(id, RemovalReason::Capacity, &context(used));
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
                residents.retain(|id, _| live.contains(id));
            }
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

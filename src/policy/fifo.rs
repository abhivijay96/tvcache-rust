//! Small reference policy: admit fitting entries and evict oldest inserted roots.
//!
//! Hits do not refresh ordering. For subtree actions, age means the root's
//! insertion order, not the oldest member of the subtree. All legality and
//! resource-pressure filtering remains the engine's responsibility.
use super::*;
use std::collections::VecDeque;

/// An intentionally simple example with O(n) removal and O(n * candidates)
/// selection. Use a fresh instance per cache run. It tracks resident IDs only.
#[derive(Debug, Default)]
pub struct FifoPolicy {
    insertion_order: VecDeque<EntryId>,
}

impl CachePolicy for FifoPolicy {
    fn name(&self) -> &str {
        "fifo"
    }

    fn on_access(
        &mut self,
        _: &EntryMeta,
        _: &AccessFeatures,
        _: AccessOutcome,
        _: &AccessContext,
    ) {
        // FIFO ordering changes only when residency changes.
    }

    fn admission(
        &mut self,
        _: &EntryMeta,
        charge: MemoryCharge,
        context: &CacheContext,
    ) -> AdmissionDecision {
        if charge.resident_bytes <= context.capacity.resident_bytes
            && charge.external_snapshot_bytes <= context.capacity.external_snapshot_bytes
        {
            AdmissionDecision::Admit
        } else {
            AdmissionDecision::Bypass
        }
    }

    fn choose_victim(
        &mut self,
        _: Option<&EntryMeta>,
        candidates: &[EvictionCandidate<'_>],
        _: &CacheContext,
    ) -> Option<usize> {
        // Skip pinned/otherwise unavailable roots omitted by the engine.
        // Unknown roots are skipped: the engine must notify every insertion.
        self.insertion_order.iter().find_map(|id| {
            candidates
                .iter()
                .position(|candidate| &candidate.root.id == id)
        })
    }

    fn on_insert(&mut self, entry: &EntryMeta, _: &CacheContext) {
        // Defensive against duplicate notifications: retain one copy, newest age.
        self.insertion_order.retain(|id| id != &entry.id);
        self.insertion_order.push_back(entry.id.clone());
    }

    fn on_remove(&mut self, id: &EntryId, _: RemovalReason, _: &CacheContext) {
        self.insertion_order.retain(|resident| resident != id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(task: &str, node: &str) -> EntryMeta {
        EntryMeta {
            id: EntryId {
                task_id: task.into(),
                node_id: node.into(),
            },
            parent: None,
            path: vec![node.into()],
            depth: 1,
            mutates_state: false,
            tool: "example".into(),
            cache_entry_bytes: 64,
            value_bytes: 64,
            runtime_ms: None,
            tool_cost_usd: None,
            snapshot: None,
        }
    }

    fn context() -> CacheContext {
        CacheContext {
            access: AccessContext {
                event_index: 0,
                time_ms: 0.0,
            },
            used: MemoryCharge::default(),
            capacity: MemoryCharge {
                resident_bytes: 128,
                external_snapshot_bytes: 256,
            },
        }
    }

    fn victim(policy: &mut dyn CachePolicy, entries: &[&EntryMeta]) -> Option<usize> {
        let features = AccessFeatures::default();
        let affected: Vec<_> = entries.iter().map(|e| vec![e.id.clone()]).collect();
        let candidates: Vec<_> = entries
            .iter()
            .zip(&affected)
            .map(|(e, ids)| EvictionCandidate {
                root: e,
                affected: ids,
                reclaimable: MemoryCharge {
                    resident_bytes: 64,
                    external_snapshot_bytes: 0,
                },
                features: &features,
            })
            .collect();
        policy.choose_victim(None, &candidates, &context())
    }

    #[test]
    fn oldest_legal_root_wins_even_after_hits_and_reordered_candidates() {
        let (a, b, c) = (entry("t", "a"), entry("t", "b"), entry("t", "c"));
        let mut policy: Box<dyn CachePolicy> = Box::new(FifoPolicy::default());
        for e in [&a, &b, &c] {
            policy.on_insert(e, &context());
        }
        policy.on_access(
            &a,
            &AccessFeatures::default(),
            AccessOutcome::ResultHit,
            &context().access,
        );
        assert_eq!(victim(policy.as_mut(), &[&c, &b, &a]), Some(2));
        // The oldest root is pinned/omitted; choose the next legal root.
        assert_eq!(victim(policy.as_mut(), &[&c, &b]), Some(1));
        assert_eq!(victim(policy.as_mut(), &[]), None);
    }

    #[test]
    fn removal_reinsertion_and_task_namespaces_are_respected() {
        let (a, b) = (entry("task-a", "same"), entry("task-b", "same"));
        let mut policy = FifoPolicy::default();
        policy.on_insert(&a, &context());
        policy.on_insert(&b, &context());
        policy.on_remove(&a.id, RemovalReason::Capacity, &context());
        policy.on_insert(&a, &context());
        assert_eq!(victim(&mut policy, &[&a, &b]), Some(1));
        policy.on_insert(&b, &context());
        assert_eq!(victim(&mut policy, &[&b, &a]), Some(1));
        for e in [&a, &b] {
            policy.on_remove(&e.id, RemovalReason::Expired, &context());
        }
        assert_eq!(victim(&mut policy, &[&a, &b]), None);
    }

    #[test]
    fn misses_and_bypasses_do_not_create_resident_policy_state() {
        let a = entry("t", "a");
        let mut policy = FifoPolicy::default();
        policy.on_access(
            &a,
            &AccessFeatures::default(),
            AccessOutcome::Miss,
            &context().access,
        );
        assert_eq!(victim(&mut policy, &[&a]), None);
    }

    #[test]
    fn admission_checks_both_budgets_without_requiring_current_free_space() {
        let a = entry("t", "a");
        let mut policy = FifoPolicy::default();
        let mut ctx = context();
        ctx.used = ctx.capacity;
        for (resident_bytes, external_snapshot_bytes, expected) in [
            (128, 256, AdmissionDecision::Admit),
            (129, 0, AdmissionDecision::Bypass),
            (0, 257, AdmissionDecision::Bypass),
        ] {
            assert_eq!(
                policy.admission(
                    &a,
                    MemoryCharge {
                        resident_bytes,
                        external_snapshot_bytes
                    },
                    &ctx
                ),
                expected
            );
        }
        ctx.capacity = MemoryCharge::default();
        assert_eq!(
            policy.admission(
                &a,
                MemoryCharge {
                    resident_bytes: 1,
                    external_snapshot_bytes: 0
                },
                &ctx
            ),
            AdmissionDecision::Bypass
        );
    }
}

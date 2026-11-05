//! Classic LRU and LFU over the engine's legal eviction actions.
//!
//! State is kept per resident entry: `hits` (1 on insert, +1 per access while
//! resident) and `last` (event index of the latest insert or access). Counts
//! reset on eviction (in-cache LFU). Misses never create state, so accesses to
//! untracked entries are ignored. Persistent-count LFU is `LinearPolicy` with
//! only `frequency = 1`.
//!
//! An action may remove a subtree, so it is scored by the max over its tracked
//! `affected` entries: a hot descendant protects its ancestors. The lowest key
//! is evicted: LRU `(last, |affected|, root id)`, LFU `(hits, last, |affected|,
//! root id)`. Actions that would remove `incoming` are chosen only when nothing
//! else is left, so a newcomer is always admitted, as in textbook LRU/LFU.
//! Untracked roots are skipped.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Lru,
    Lfu,
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    hits: u64,
    last: u64,
}

/// Use a fresh instance per cache run. It tracks resident IDs only.
#[derive(Debug)]
pub struct ClassicPolicy {
    kind: Kind,
    stats: BTreeMap<EntryId, Stats>,
}

impl ClassicPolicy {
    pub fn lru() -> Self {
        Self {
            kind: Kind::Lru,
            stats: BTreeMap::new(),
        }
    }

    pub fn lfu() -> Self {
        Self {
            kind: Kind::Lfu,
            stats: BTreeMap::new(),
        }
    }
}

impl CachePolicy for ClassicPolicy {
    fn name(&self) -> &str {
        match self.kind {
            Kind::Lru => "lru",
            Kind::Lfu => "lfu",
        }
    }

    fn parameters(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"aggregate": "max", "lfu_counts": "in-cache"}))
    }

    fn on_access(
        &mut self,
        entry: &EntryMeta,
        _: &AccessFeatures,
        _: AccessOutcome,
        context: &AccessContext,
    ) {
        if let Some(stats) = self.stats.get_mut(&entry.id) {
            stats.hits += 1;
            stats.last = context.event_index;
        }
    }

    fn admission(
        &mut self,
        _: &EntryMeta,
        charge: MemoryCharge,
        context: &CacheContext,
    ) -> AdmissionDecision {
        admit_if_fits(charge, context)
    }

    fn choose_victim(
        &mut self,
        incoming: Option<&EntryMeta>,
        candidates: &[EvictionCandidate<'_>],
        _: &CacheContext,
    ) -> Option<usize> {
        candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let root = self.stats.get(&candidate.root.id)?;
                let (hits, last) = candidate
                    .affected
                    .iter()
                    .filter_map(|id| self.stats.get(id))
                    .fold((root.hits, root.last), |(h, l), s| {
                        (h.max(s.hits), l.max(s.last))
                    });
                let removes_incoming = incoming.is_some_and(|entry| {
                    candidate.root.id == entry.id || candidate.affected.contains(&entry.id)
                });
                let frequency = if self.kind == Kind::Lfu { hits } else { 0 };
                let key = (
                    removes_incoming,
                    frequency,
                    last,
                    candidate.affected.len(),
                    &candidate.root.id,
                );
                Some((key, index))
            })
            .min()
            .map(|(_, index)| index)
    }

    fn on_insert(&mut self, entry: &EntryMeta, context: &CacheContext) {
        self.stats.insert(
            entry.id.clone(),
            Stats {
                hits: 1,
                last: context.access.event_index,
            },
        );
    }

    fn on_remove(&mut self, id: &EntryId, _: RemovalReason, _: &CacheContext) {
        self.stats.remove(id);
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
            mutates_state: true,
            tool: "example".into(),
            cache_entry_bytes: 64,
            value_bytes: 64,
            runtime_ms: None,
            tool_cost_usd: None,
            snapshot: None,
        }
    }

    fn ctx(event_index: u64) -> CacheContext {
        CacheContext {
            access: AccessContext {
                event_index,
                time_ms: event_index as f64,
            },
            used: MemoryCharge::default(),
            capacity: MemoryCharge {
                resident_bytes: u64::MAX,
                external_snapshot_bytes: u64::MAX,
            },
        }
    }

    fn insert(policy: &mut ClassicPolicy, e: &EntryMeta, at: u64) {
        policy.on_insert(e, &ctx(at));
    }

    fn hit(policy: &mut ClassicPolicy, e: &EntryMeta, at: u64) {
        let features = AccessFeatures::default();
        policy.on_access(e, &features, AccessOutcome::ResultHit, &ctx(at).access);
    }

    fn single(e: &EntryMeta) -> (&EntryMeta, Vec<EntryId>) {
        (e, vec![e.id.clone()])
    }

    /// Each action is (root, affected ids including the root).
    fn victim(
        policy: &mut ClassicPolicy,
        incoming: Option<&EntryMeta>,
        actions: &[(&EntryMeta, Vec<EntryId>)],
    ) -> Option<usize> {
        let features = AccessFeatures::default();
        let candidates: Vec<_> = actions
            .iter()
            .map(|(root, affected)| EvictionCandidate {
                root,
                affected,
                reclaimable: MemoryCharge {
                    resident_bytes: 64,
                    external_snapshot_bytes: 0,
                },
                features: &features,
            })
            .collect();
        policy.choose_victim(incoming, &candidates, &ctx(1000))
    }

    fn both() -> [ClassicPolicy; 2] {
        [ClassicPolicy::lru(), ClassicPolicy::lfu()]
    }

    #[test]
    fn hit_refreshes_recency_and_equal_counts_evict_older() {
        let (a, b, c) = (entry("t", "a"), entry("t", "b"), entry("t", "c"));
        for mut policy in both() {
            for (i, e) in [&a, &b, &c].into_iter().enumerate() {
                insert(&mut policy, e, i as u64);
            }
            hit(&mut policy, &a, 3);
            // LRU: b is now oldest. LFU: b and c tie at one hit; b is older.
            let actions = [single(&a), single(&b), single(&c)];
            assert_eq!(victim(&mut policy, None, &actions), Some(1));
        }
    }

    #[test]
    fn lfu_keeps_frequent_entry_that_lru_evicts() {
        let (a, b) = (entry("t", "a"), entry("t", "b"));
        let [mut lru, mut lfu] = both();
        for policy in [&mut lru, &mut lfu] {
            insert(policy, &a, 0);
            hit(policy, &a, 1);
            hit(policy, &a, 2);
            insert(policy, &b, 3);
        }
        let actions = [single(&a), single(&b)];
        assert_eq!(victim(&mut lru, None, &actions), Some(0));
        assert_eq!(victim(&mut lfu, None, &actions), Some(1));
    }

    #[test]
    fn hot_descendant_protects_ancestor_and_ties_prefer_smaller_subtree() {
        let (parent, child, leaf) = (entry("t", "p"), entry("t", "c"), entry("t", "l"));
        for mut policy in both() {
            insert(&mut policy, &parent, 0);
            insert(&mut policy, &child, 1);
            insert(&mut policy, &leaf, 2);
            hit(&mut policy, &child, 5);
            let subtree = (&parent, vec![parent.id.clone(), child.id.clone()]);
            let actions = [subtree.clone(), single(&child), single(&leaf)];
            assert_eq!(victim(&mut policy, None, &actions), Some(2));
            // With the leaf gone, the parent's subtree ties the hot child: drop the child alone.
            let actions = [subtree, single(&child)];
            assert_eq!(victim(&mut policy, None, &actions), Some(1));
        }
    }

    #[test]
    fn incoming_is_protected_until_nothing_else_is_left() {
        let (a, b) = (entry("t", "a"), entry("t", "b"));
        for mut policy in both() {
            insert(&mut policy, &a, 0);
            for i in 1..4 {
                hit(&mut policy, &a, i);
            }
            insert(&mut policy, &b, 4);
            assert_eq!(victim(&mut policy, Some(&b), &[single(&a), single(&b)]), Some(0));
            // An ancestor whose subtree holds the incoming entry also removes it.
            let ancestor = (&a, vec![a.id.clone(), b.id.clone()]);
            assert_eq!(victim(&mut policy, Some(&b), &[ancestor, single(&b)]), Some(1));
            assert_eq!(victim(&mut policy, Some(&b), &[single(&b)]), Some(0));
        }
    }

    #[test]
    fn lfu_count_resets_after_eviction() {
        let (a, b) = (entry("t", "a"), entry("t", "b"));
        let mut lfu = ClassicPolicy::lfu();
        insert(&mut lfu, &a, 0);
        for i in 1..5 {
            hit(&mut lfu, &a, i);
        }
        insert(&mut lfu, &b, 5);
        hit(&mut lfu, &b, 6);
        lfu.on_remove(&a.id, RemovalReason::Capacity, &ctx(7));
        insert(&mut lfu, &a, 8);
        // a is newer but back to one hit; b has two.
        assert_eq!(victim(&mut lfu, None, &[single(&a), single(&b)]), Some(0));
    }

    #[test]
    fn tasks_untracked_roots_and_misses() {
        let (x1, x2, unknown) = (entry("t1", "x"), entry("t2", "x"), entry("t", "u"));
        for mut policy in both() {
            insert(&mut policy, &x1, 0);
            insert(&mut policy, &x2, 1);
            hit(&mut policy, &x1, 2);
            hit(&mut policy, &x1, 3);
            assert_eq!(victim(&mut policy, None, &[single(&x1), single(&x2)]), Some(1));
            let features = AccessFeatures::default();
            policy.on_access(&unknown, &features, AccessOutcome::Miss, &ctx(4).access);
            assert_eq!(victim(&mut policy, None, &[single(&unknown)]), None);
            assert_eq!(victim(&mut policy, None, &[single(&unknown), single(&x2)]), Some(1));
            assert_eq!(victim(&mut policy, None, &[]), None);
        }
    }
}

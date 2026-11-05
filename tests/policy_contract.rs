use std::{fs::File, io::BufReader, path::Path};
use tvcache::policy::{synthetic::*, *};

// Minimal contract example, deliberately not James's LRU/LFU implementation.
struct FirstEligible;
impl CachePolicy for FirstEligible {
    fn name(&self) -> &str {
        "first-eligible"
    }
    fn on_access(
        &mut self,
        _: &EntryMeta,
        _: &AccessFeatures,
        _: AccessOutcome,
        _: &AccessContext,
    ) {
    }
    fn admission(&mut self, _: &EntryMeta, _: MemoryCharge, _: &CacheContext) -> AdmissionDecision {
        AdmissionDecision::Admit
    }
    fn choose_victim(
        &mut self,
        _: Option<&EntryMeta>,
        c: &[EvictionCandidate<'_>],
        _: &CacheContext,
    ) -> Option<usize> {
        if c.is_empty() {
            None
        } else {
            Some(0)
        }
    }
    fn on_insert(&mut self, _: &EntryMeta, _: &CacheContext) {}
    fn on_remove(&mut self, _: &EntryId, _: RemovalReason, _: &CacheContext) {}
}

#[test]
fn adapter_resolves_every_event_in_team_datasets() {
    for name in [
        "balanced",
        "expensive_reuse",
        "memory_pressure",
        "popularity_shift",
    ] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data/synthetic")
            .join(name);
        let manifest: serde_json::Value =
            serde_json::from_reader(File::open(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["format"], "tvcache-synthetic-v1");
        let catalog =
            TraceCatalog::read(BufReader::new(File::open(dir.join("nodes.jsonl")).unwrap()))
                .unwrap();
        let events: Vec<AccessRecord> = read_jsonl(BufReader::new(
            File::open(dir.join("events.jsonl")).unwrap(),
        ))
        .unwrap();
        assert!(!events.is_empty());
        let mut previous = None;
        for event in events {
            let (entry, context) = catalog.resolve(&event).unwrap();
            assert_eq!(entry.id.task_id, event.task_id);
            assert_eq!(entry.id.node_id, event.node_id);
            assert_eq!(entry.depth as usize, entry.path.len());
            if let Some((index, time)) = previous {
                assert!(context.event_index > index);
                assert!(context.time_ms >= time);
            }
            previous = Some((context.event_index, context.time_ms));
        }
    }
}

#[test]
fn policy_is_object_safe_and_candidates_can_describe_subtrees() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/synthetic/balanced");
    let catalog =
        TraceCatalog::read(BufReader::new(File::open(dir.join("nodes.jsonl")).unwrap())).unwrap();
    let mut events: Vec<AccessRecord> = read_jsonl(BufReader::new(
        File::open(dir.join("events.jsonl")).unwrap(),
    ))
    .unwrap();
    let (entry, access) = catalog.resolve(&events[0]).unwrap();
    let (child, _) = catalog.resolve(&events[1]).unwrap();
    let ctx = CacheContext {
        access,
        used: MemoryCharge::default(),
        capacity: MemoryCharge {
            resident_bytes: 1024,
            external_snapshot_bytes: u64::MAX,
        },
    };
    let mut policy: Box<dyn CachePolicy> = Box::new(FirstEligible);
    assert_eq!(policy.name(), "first-eligible");
    policy.on_access(&entry, &events[0].features, AccessOutcome::Miss, &access);
    assert_eq!(
        policy.admission(&entry, MemoryCharge::default(), &ctx),
        AdmissionDecision::Admit
    );
    policy.on_insert(&entry, &ctx);
    assert_eq!(policy.choose_victim(None, &[], &ctx), None);
    let affected = vec![entry.id.clone(), child.id.clone()];
    let actions = [EvictionCandidate {
        root: &entry,
        affected: &affected,
        reclaimable: MemoryCharge {
            resident_bytes: entry.cache_entry_bytes + child.cache_entry_bytes,
            external_snapshot_bytes: 0,
        },
        features: &events[0].features,
    }];
    assert_eq!(policy.choose_victim(None, &actions, &ctx), Some(0));
    for id in &affected {
        policy.on_remove(id, RemovalReason::Capacity, &ctx);
    }
    events[0].node_id = "missing".into();
    assert!(catalog.resolve(&events[0]).is_err());
}

#[test]
fn adapter_reports_bad_json_line_and_duplicate_nodes() {
    assert!(read_jsonl::<AccessRecord>(&b"{}\n"[..])
        .unwrap_err()
        .to_string()
        .contains("line 1"));
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/synthetic/balanced/nodes.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let first = text.lines().next().unwrap();
    let duplicate = format!("{first}\n{first}\n");
    assert!(TraceCatalog::read(duplicate.as_bytes())
        .err()
        .unwrap()
        .to_string()
        .contains("duplicate node"));
}

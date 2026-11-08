//! JSONL dump of eviction candidates for training learned policies.
//!
//! The first line is a header with the feature names; every other line is one
//! sampled candidate at one eviction. Features come from `policy::features`, the
//! same code the serving policies run. Labels (future reuse) are not written
//! here: the engine must not see the future, so `tvcache_learn` adds them later.
use crate::{
    policy::{features, CacheContext, EntryMeta, EvictionCandidate},
    replay::EvictionObserver,
};
use std::io::{self, Write};

pub const FORMAT: &str = "tvcache-candidates-v1";

pub struct CandidateDump<W: Write> {
    out: W,
    per_eviction: usize,
    rng: u64,
    evictions: u64,
    with_affected: bool,
}

impl<W: Write> CandidateDump<W> {
    /// Writes the header immediately. `per_eviction` candidates are sampled
    /// without replacement at each eviction (all of them if fewer), and the
    /// chosen victim is always included.
    pub fn new(
        mut out: W,
        per_eviction: usize,
        seed: u64,
        header: serde_json::Value,
    ) -> io::Result<Self> {
        let mut header = header;
        header["format"] = FORMAT.into();
        header["feature_names"] = features::RICH_NAMES.to_vec().into();
        header["basic_features"] = features::BASIC_NAMES.len().into();
        header["per_eviction"] = per_eviction.into();
        header["seed"] = seed.into();
        serde_json::to_writer(&mut out, &header)?;
        out.write_all(b"\n")?;
        Ok(Self {
            out,
            per_eviction,
            // xorshift needs a nonzero state.
            rng: seed ^ 0x9e3779b97f4a7c15,
            evictions: 0,
            with_affected: false,
        })
    }

    /// Also write each candidate's affected node IDs (root included), so labels
    /// can count the cost of everything an eviction removes. Larger files.
    pub fn with_affected(mut self, enabled: bool) -> Self {
        self.with_affected = enabled;
        self
    }

    fn next(&mut self) -> u64 {
        // xorshift64*: deterministic and dependency-free.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545f4914f6cdd1d)
    }

    fn sample(&mut self, len: usize, chosen: usize) -> Vec<usize> {
        if len <= self.per_eviction {
            return (0..len).collect();
        }
        let mut pool: Vec<usize> = (0..len).filter(|&i| i != chosen).collect();
        let mut picked = vec![chosen];
        for k in 0..self.per_eviction.saturating_sub(1) {
            let j = k + (self.next() % (pool.len() - k) as u64) as usize;
            pool.swap(k, j);
            picked.push(pool[k]);
        }
        picked.sort_unstable();
        picked
    }

    pub fn into_inner(self) -> W {
        self.out
    }
}

impl<W: Write> EvictionObserver for CandidateDump<W> {
    fn observe(
        &mut self,
        context: &CacheContext,
        incoming: &EntryMeta,
        candidates: &[EvictionCandidate<'_>],
        chosen: usize,
    ) -> io::Result<()> {
        let eviction = self.evictions;
        self.evictions += 1;
        for i in self.sample(candidates.len(), chosen) {
            let c = &candidates[i];
            let mut row = serde_json::json!({
                "eviction": eviction,
                "event_index": context.access.event_index,
                "time_ms": context.access.time_ms,
                "capacity_bytes": context.capacity.resident_bytes,
                "used_bytes": context.used.resident_bytes,
                "num_candidates": candidates.len(),
                "task_id": c.root.id.task_id,
                "node_id": c.root.id.node_id,
                "affected_entries": c.affected.len(),
                "reclaimable_bytes": c.reclaimable.resident_bytes,
                "chosen": i == chosen,
                "removes_incoming": c.root.id == incoming.id || c.affected.contains(&incoming.id),
            });
            if self.with_affected {
                row["affected"] = c.affected.iter().map(|id| id.node_id.as_str()).collect::<Vec<_>>().into();
            }
            // Features are written as shortest-form f32 (training runs in f32),
            // which serde_json cannot do: it widens every number to f64.
            let mut line = row.to_string();
            line.pop();
            let x: Vec<String> = features::rich(c, context)
                .into_iter()
                .map(|v| (v as f32).to_string())
                .collect();
            writeln!(self.out, "{line},\"x\":[{}]}}", x.join(","))?;
        }
        Ok(())
    }
}

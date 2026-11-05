//! Budget sweep of every replay policy on the synthetic traces. Every run's decision
//! log is checked with `check_decisions` before its CSV row is written.
//! Run from the repository root: cargo run --release --bin sweep -- [OUTPUT_CSV]
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};
use tvcache::{
    policy::{
        classic::ClassicPolicy,
        fifo::FifoPolicy,
        linear::{LinearPolicy, LinearWeights},
        synthetic::*,
        CachePolicy,
    },
    replay::{check_decisions, replay},
};

// About 10/25/50/75/100% of each trace's unique_cache_entry_bytes.
const SWEEP: [(&str, [u32; 5]); 4] = [
    ("balanced", [1, 3, 5, 8, 10]),
    ("expensive_reuse", [1, 2, 4, 6, 7]),
    ("popularity_shift", [1, 3, 5, 8, 10]),
    ("memory_pressure", [133, 331, 661, 991, 1321]),
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "results/lru-lfu/summary.csv".into());
    let persistent: LinearWeights =
        serde_json::from_reader(File::open("configs/lfu-persistent.json")?)?;
    let mut csv = BufWriter::new(File::create(&out)?);
    writeln!(csv, "dataset,budget_mib,policy,hit_rate,gross_tool_ms_saved,gross_tool_cost_usd_saved,peak_bytes,evicted_entries,bypassed_missing_parent,policy_ns_per_request")?;
    for (dataset, budgets) in SWEEP {
        let dir = Path::new("data/synthetic").join(dataset);
        let catalog = TraceCatalog::read(BufReader::new(File::open(dir.join("nodes.jsonl"))?))?;
        let events: Vec<AccessRecord> =
            read_jsonl(BufReader::new(File::open(dir.join("events.jsonl"))?))?;
        for mib in budgets {
            let runs: [(&str, Option<Box<dyn CachePolicy>>); 6] = [
                ("original-ttl-and-watermark", None),
                ("fifo", Some(Box::new(FifoPolicy::default()))),
                ("linear", Some(Box::new(LinearPolicy::new(LinearWeights::default())?))),
                ("lru", Some(Box::new(ClassicPolicy::lru()))),
                ("lfu", Some(Box::new(ClassicPolicy::lfu()))),
                ("lfu-persistent", Some(Box::new(LinearPolicy::new(persistent.clone())?))),
            ];
            for (label, policy) in runs {
                let r = replay(&catalog, &events, mib, policy).await?;
                check_decisions(&r, &catalog, &events)
                    .map_err(|e| format!("{dataset} {mib} MiB {label}: {e}"))?;
                let usd = r.gross_tool_cost_usd_saved.map_or(String::new(), |v| v.to_string());
                let ns = r.policy_ns as f64 / r.requests as f64;
                writeln!(
                    csv,
                    "{dataset},{mib},{label},{},{},{usd},{},{},{},{ns}",
                    r.hit_rate,
                    r.gross_tool_ms_saved,
                    r.peak_bytes_after_enforcement,
                    r.evicted_entries,
                    r.bypassed_missing_parent
                )?;
                eprintln!("{dataset:<17} {mib:>5} MiB {label:<27} hit {:.4}", r.hit_rate);
            }
        }
    }
    csv.flush()?;
    eprintln!("wrote {out}");
    Ok(())
}

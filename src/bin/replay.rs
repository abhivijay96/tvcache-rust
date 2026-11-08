use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use tvcache::{
    candidates::CandidateDump,
    policy::{
        classic::ClassicPolicy,
        fifo::FifoPolicy,
        linear::{LinearPolicy, LinearWeights},
        mlp::MlpPolicy,
        synthetic::*,
        CachePolicy,
    },
    replay::{replay, replay_observed, ReplayReport},
};

const USAGE: &str = "Usage: cargo run --bin replay -- DATASET_DIR original|fifo|both|linear|lru|lfu|mlp|all CAPACITY_MIB [LINEAR_WEIGHTS_JSON|MLP_MODEL_JSON] [--decisions=DIR] [--dump-candidates=FILE [--dump-per-eviction=K] [--dump-seed=N] [--dump-affected]]";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut decisions_dir = None;
    let mut dump_path = None;
    let mut per_eviction = 32usize;
    let mut dump_seed = 0u64;
    let mut dump_affected = false;
    let mut args = Vec::new();
    for arg in std::env::args().skip(1) {
        if let Some(dir) = arg.strip_prefix("--decisions=") {
            decisions_dir = Some(PathBuf::from(dir));
        } else if let Some(file) = arg.strip_prefix("--dump-candidates=") {
            dump_path = Some(PathBuf::from(file));
        } else if let Some(k) = arg.strip_prefix("--dump-per-eviction=") {
            per_eviction = k.parse()?;
        } else if let Some(n) = arg.strip_prefix("--dump-seed=") {
            dump_seed = n.parse()?;
        } else if arg == "--dump-affected" {
            dump_affected = true;
        } else {
            args.push(arg);
        }
    }
    if !(args.len() == 3 || args.len() == 4)
        || !matches!(
            args[1].as_str(),
            "original" | "fifo" | "both" | "linear" | "lru" | "lfu" | "mlp" | "all"
        )
        || (args.len() == 4 && !matches!(args[1].as_str(), "linear" | "mlp" | "all"))
        || (args[1] == "mlp" && args.len() != 4)
    {
        return Err(USAGE.into());
    }
    let mode = args[1].as_str();
    if dump_path.is_some() && matches!(mode, "original" | "both" | "all") {
        return Err("--dump-candidates needs exactly one non-original policy".into());
    }
    let dir = PathBuf::from(&args[0]);
    let capacity: u32 = args[2].parse()?;
    let weights: LinearWeights = match args.get(3) {
        Some(path) if mode != "mlp" => serde_json::from_reader(File::open(path)?)?,
        _ => LinearWeights::default(),
    };
    let linear = LinearPolicy::new(weights)?;
    let manifest: serde_json::Value =
        serde_json::from_reader(File::open(dir.join("manifest.json"))?)?;
    if manifest["format"] != "tvcache-synthetic-v1" {
        return Err("unsupported trace format".into());
    }
    let catalog = TraceCatalog::read(BufReader::new(File::open(dir.join("nodes.jsonl"))?))?;
    let events: Vec<AccessRecord> =
        read_jsonl(BufReader::new(File::open(dir.join("events.jsonl"))?))?;
    if let Some(path) = dump_path {
        let policy: Box<dyn CachePolicy> = match mode {
            "fifo" => Box::new(FifoPolicy::default()),
            "linear" => Box::new(linear),
            "lru" => Box::new(ClassicPolicy::lru()),
            "lfu" => Box::new(ClassicPolicy::lfu()),
            "mlp" => Box::new(MlpPolicy::from_json(BufReader::new(File::open(&args[3])?))?),
            _ => unreachable!(),
        };
        let header = serde_json::json!({
            "trace": dir.file_name().map(|n| n.to_string_lossy().into_owned()),
            "trace_dir": dir,
            "capacity_mib": capacity,
            "behavior_policy": policy.name(),
        });
        let out = BufWriter::new(File::create(&path)?);
        let mut dump =
            CandidateDump::new(out, per_eviction, dump_seed, header)?.with_affected(dump_affected);
        let report = replay_observed(&catalog, &events, capacity, Some(policy), Some(&mut dump)).await?;
        dump.into_inner().flush()?;
        write_decisions(decisions_dir.as_deref(), std::slice::from_ref(&report))?;
        serde_json::to_writer_pretty(io::stdout().lock(), &[report])?;
        println!();
        return Ok(());
    }
    let mut reports = Vec::new();
    if matches!(mode, "original" | "both" | "all") {
        reports.push(replay(&catalog, &events, capacity, None).await?);
    }
    if matches!(mode, "fifo" | "both" | "all") {
        reports.push(
            replay(
                &catalog,
                &events,
                capacity,
                Some(Box::new(FifoPolicy::default())),
            )
            .await?,
        );
    }
    if matches!(mode, "linear" | "all") {
        reports.push(replay(&catalog, &events, capacity, Some(Box::new(linear))).await?);
    }
    if matches!(mode, "lru" | "all") {
        reports.push(replay(&catalog, &events, capacity, Some(Box::new(ClassicPolicy::lru()))).await?);
    }
    if matches!(mode, "lfu" | "all") {
        reports.push(replay(&catalog, &events, capacity, Some(Box::new(ClassicPolicy::lfu()))).await?);
    }
    if mode == "mlp" {
        let mlp = MlpPolicy::from_json(BufReader::new(File::open(&args[3])?))?;
        reports.push(replay(&catalog, &events, capacity, Some(Box::new(mlp))).await?);
    }
    write_decisions(decisions_dir.as_deref(), &reports)?;
    serde_json::to_writer_pretty(io::stdout().lock(), &reports)?;
    println!();
    Ok(())
}

fn write_decisions(dir: Option<&Path>, reports: &[ReplayReport]) -> io::Result<()> {
    let Some(out) = dir else { return Ok(()) };
    std::fs::create_dir_all(out)?;
    for report in reports {
        let mut file = BufWriter::new(File::create(
            out.join(format!("{}.decisions.jsonl", report.policy)),
        )?);
        for decision in &report.decisions {
            serde_json::to_writer(&mut file, decision)?;
            file.write_all(b"\n")?;
        }
        file.flush()?;
    }
    Ok(())
}

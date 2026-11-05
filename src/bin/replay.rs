use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    path::PathBuf,
};
use tvcache::{
    policy::{
        classic::ClassicPolicy,
        fifo::FifoPolicy,
        linear::{LinearPolicy, LinearWeights},
        synthetic::*,
    },
    replay::replay,
};

const USAGE: &str = "Usage: cargo run --bin replay -- DATASET_DIR original|fifo|both|linear|lru|lfu|all CAPACITY_MIB [LINEAR_WEIGHTS_JSON] [--decisions=DIR]";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut decisions_dir = None;
    let mut args = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.strip_prefix("--decisions=") {
            Some(dir) => decisions_dir = Some(PathBuf::from(dir)),
            None => args.push(arg),
        }
    }
    if !(args.len() == 3 || args.len() == 4)
        || !matches!(
            args[1].as_str(),
            "original" | "fifo" | "both" | "linear" | "lru" | "lfu" | "all"
        )
        || (args.len() == 4 && !matches!(args[1].as_str(), "linear" | "all"))
    {
        return Err(USAGE.into());
    }
    let dir = PathBuf::from(&args[0]);
    let capacity: u32 = args[2].parse()?;
    let weights: LinearWeights = if let Some(path) = args.get(3) {
        serde_json::from_reader(File::open(path)?)?
    } else {
        LinearWeights::default()
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
    let mode = args[1].as_str();
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
    if let Some(out) = decisions_dir {
        std::fs::create_dir_all(&out)?;
        for report in &reports {
            let mut file = BufWriter::new(File::create(
                out.join(format!("{}.decisions.jsonl", report.policy)),
            )?);
            for decision in &report.decisions {
                serde_json::to_writer(&mut file, decision)?;
                file.write_all(b"\n")?;
            }
            file.flush()?;
        }
    }
    serde_json::to_writer_pretty(io::stdout().lock(), &reports)?;
    println!();
    Ok(())
}

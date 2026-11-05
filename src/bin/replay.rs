use std::{
    fs::File,
    io::{self, BufReader},
    path::PathBuf,
};
use tvcache::{
    policy::{
        fifo::FifoPolicy,
        linear::{LinearPolicy, LinearWeights},
        synthetic::*,
    },
    replay::replay,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(args.len() == 3 || args.len() == 4)
        || !matches!(
            args[1].as_str(),
            "original" | "fifo" | "both" | "linear" | "all"
        )
        || (args.len() == 4 && !matches!(args[1].as_str(), "linear" | "all"))
    {
        return Err(
            "Usage: cargo run --bin replay -- DATASET_DIR original|fifo|both|linear|all CAPACITY_MIB [LINEAR_WEIGHTS_JSON]".into(),
        );
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
    let mut reports = Vec::new();
    if matches!(args[1].as_str(), "original" | "both" | "all") {
        reports.push(replay(&catalog, &events, capacity, None).await?);
    }
    if matches!(args[1].as_str(), "fifo" | "both" | "all") {
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
    if matches!(args[1].as_str(), "linear" | "all") {
        reports.push(replay(&catalog, &events, capacity, Some(Box::new(linear))).await?);
    }
    serde_json::to_writer_pretty(io::stdout().lock(), &reports)?;
    println!();
    Ok(())
}

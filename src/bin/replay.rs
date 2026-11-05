use std::{
    fs::File,
    io::{self, BufReader},
    path::PathBuf,
};
use tvcache::{
    policy::{fifo::FifoPolicy, synthetic::*},
    replay::replay,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 || !matches!(args[1].as_str(), "original" | "fifo" | "both") {
        return Err(
            "Usage: cargo run --bin replay -- DATASET_DIR original|fifo|both CAPACITY_MIB".into(),
        );
    }
    let dir = PathBuf::from(&args[0]);
    let capacity: u32 = args[2].parse()?;
    let manifest: serde_json::Value =
        serde_json::from_reader(File::open(dir.join("manifest.json"))?)?;
    if manifest["format"] != "tvcache-synthetic-v1" {
        return Err("unsupported trace format".into());
    }
    let catalog = TraceCatalog::read(BufReader::new(File::open(dir.join("nodes.jsonl"))?))?;
    let events: Vec<AccessRecord> =
        read_jsonl(BufReader::new(File::open(dir.join("events.jsonl"))?))?;
    let mut reports = Vec::new();
    if args[1] != "fifo" {
        reports.push(replay(&catalog, &events, capacity, None).await?);
    }
    if args[1] != "original" {
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
    serde_json::to_writer_pretty(io::stdout().lock(), &reports)?;
    println!();
    Ok(())
}

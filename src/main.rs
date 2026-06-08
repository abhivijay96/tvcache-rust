mod enforcer;
mod eviction;
mod protocol_primitives;
mod request_handler;
mod tcg;
mod tvcache;
mod types;
mod utils;

use std::sync::Arc;
use std::{env, io};

use eviction::{sandbox_tracker::SandboxTracker, EvictionPolicyConfig, EvictionPolicyName};
use request_handler::RequestHandler;
use tokio::{net::TcpListener, sync::Mutex};
use tvcache::TVCache;
use types::{CacheSharedObject, SandboxTrackerSharedObject};

#[tokio::main]
async fn main() -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:7878").await?;
    let eviction_policy_config = parse_eviction_policy_config();

    let cache_shared_object: CacheSharedObject = Arc::new(Mutex::new(TVCache::new()));
    let sandbox_tracker_shared_object: SandboxTrackerSharedObject =
        Arc::new(Mutex::new(SandboxTracker::new()));
    let eviction_cache = Arc::clone(&cache_shared_object);
    let eviction_sandbox_tracker = Arc::clone(&sandbox_tracker_shared_object);

    tokio::spawn(async move {
        enforcer::run_eviction_loop(
            eviction_cache,
            eviction_policy_config,
            eviction_sandbox_tracker,
        )
        .await;
    });

    println!("Listening on 127.0.0.1:7878 with tokio");

    loop {
        match listener.accept().await {
            Ok((tcp_stream, _ /*this is socket attress but I am nto usin this */)) => {
                let cache_shared_object = Arc::clone(&cache_shared_object);
                let stream = tcp_stream;

                tokio::spawn(async move {
                    let request_handler = RequestHandler::new(cache_shared_object);
                    request_handler.handle_stream(stream).await;
                });
            }
            Err(error) => {
                eprintln!("Failed to accept connection: {error}");
            }
        }
    }
}

fn parse_eviction_policy_config() -> EvictionPolicyConfig {
    let mut eviction_policy = String::from("ttl_and_watermark");
    let mut high_watermark_mb = 1024;
    let mut low_watermark_mb = 768;

    for arg in env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--eviction-policy=") {
            eviction_policy = value.to_string();
        } else if let Some(value) = arg.strip_prefix("--high-watermark-mb=") {
            high_watermark_mb = value.parse().unwrap_or(high_watermark_mb);
        } else if let Some(value) = arg.strip_prefix("--low-watermark-mb=") {
            low_watermark_mb = value.parse().unwrap_or(low_watermark_mb);
        }
    }

    let policy_name = match eviction_policy.as_str() {
        "none" => EvictionPolicyName::None,
        "ttl_and_watermark" => EvictionPolicyName::TtlAndWatermark,
        _ => EvictionPolicyName::TtlAndWatermark,
    };

    EvictionPolicyConfig::new(policy_name, high_watermark_mb, low_watermark_mb)
}

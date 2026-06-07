mod protocol_primitives;
mod request_handler;
mod tcg;
mod tvcache;
mod types;
mod utils;

use std::io;
use std::sync::Arc;

use request_handler::RequestHandler;
use tokio::{net::TcpListener, sync::Mutex};
use tvcache::TVCache;
use types::CacheSharedObject;

#[tokio::main]
async fn main() -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:7878").await?;
    let cache_shared_object: CacheSharedObject = Arc::new(Mutex::new(TVCache::new()));

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

// class tool_call_graph, properties: Root: Node, number of nodes, number of envs
// methods: insert (list of string) -> returns bool, find (list of string) -> return a tuple (string, list(string))

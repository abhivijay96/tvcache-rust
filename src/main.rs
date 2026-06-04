mod request_handler;
mod tcg;
mod tvcache;
mod types;

use std::collections::VecDeque;
use std::env;
use std::io;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use request_handler::RequestHandler;
use tvcache::TVCache;
use types::{CacheSharedObject, StreamsSharedObject};

fn handle_incoming_requests(
    streams_shared_object: StreamsSharedObject,
    cache_shared_object: CacheSharedObject,
) {
    let mut request_handler = RequestHandler::new(cache_shared_object);

    loop {
        let stream = {
            let mut streams_object = streams_shared_object.lock().unwrap();

            streams_object.pop_front()
        };

        if let Some(stream) = stream {
            // println!("Worker {:?} dequeued a stream", thread::current().id());
            request_handler.handle_stream(stream);
        } else {
            thread::sleep(Duration::from_millis(1));
        }

        // Stream gets dropped because it has been moved to this block's scope and it will be dropped after this
    }
}

fn main() -> io::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:7878")?;
    let num_threads = env::var("NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4);

    // Objects related to tvcache
    let streams_shared_object: StreamsSharedObject = Arc::new(Mutex::new(VecDeque::new()));
    let cache_shared_object: CacheSharedObject = Arc::new(Mutex::new(TVCache::new()));

    for _ in 0..num_threads {
        let streams_shared_object = Arc::clone(&streams_shared_object);
        let cache_shared_object = Arc::clone(&cache_shared_object);

        thread::spawn(move || handle_incoming_requests(streams_shared_object, cache_shared_object));
    }

    println!("Listening on 127.0.0.1:7878 with {num_threads} worker threads");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let mut streams_object = streams_shared_object.lock().unwrap();
                streams_object.push_back(stream);
                // println!("Connection queued. Queue length: {}", streams_object.len());
            }
            Err(error) => {
                eprintln!("Failed to accept connection: {error}");
            }
        }
    }

    Ok(())
}

// class tool_call_graph, properties: Root: Node, number of nodes, number of envs
// methods: insert (list of string) -> returns bool, find (list of string) -> return a tuple (string, list(string))

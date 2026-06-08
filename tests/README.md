# Python Protocol Smoke Test

Start the Rust server in one terminal:

```sh
cargo run
```

Run the Python client in another terminal:

```sh
cd tests
uv run test_client.py
```

Run the load test:

```sh
cd tests
uv run load_test.py
```

Run the eviction policy test against a fresh Rust server with watermark eviction enabled:

```sh
cargo run -- --eviction-policy=ttl_and_watermark --high-watermark-mb=2 --low-watermark-mb=1
```

Then, in another terminal:

```sh
cd tests
uv run test_eviction.py
```

Run the TTL eviction test against a fresh Rust server with the same eviction policy enabled:

```sh
cd tests
uv run test_ttl_eviction.py
```

Open the TCG visualizer in another terminal:

```sh
cd tests
uv run visualize.py
```

Then visit:

```text
http://127.0.0.1:8080
```

The client sends six requests:

1. `INSERT` for a one-node parent path with a sandbox id.
2. `INSERT` for a child path that extends the parent by exactly one node.
3. `INSERT` for a child path whose parent is missing, which should fail gracefully.
4. `FIND` for the inserted child path.
5. `FIND` for a missing path.
6. `DUMP_ALL` to confirm the JSON endpoint returns the current `TVCache`.

The script prints each decoded response and exits with an assertion error if the protocol response does not match the expected value.

`cache_client.py` contains the reusable TCP protocol client used by `test_client.py`, `load_test.py`, and `visualize.py`.

The load test creates `--num-tasks` TCGs, depth 5 by default. For each task it builds a valid TCG by inserting parents before children, with random width 1-8 at each parent. Dataset creation reuses one socket per task. After the dataset is created, it runs one find phase for each value in `--num-rollouts`. Each phase starts `num_tasks * num_rollouts` parallel find workers. Each find worker opens one socket and performs one random valid `FIND` for its designated task. The script logs mean, median, and p95 response completion times.

Useful load-test options:

```sh
uv run load_test.py --num-tasks 32 --num-rollouts 4
uv run load_test.py --num-tasks 32 --num-rollouts 1,2,4,8
uv run load_test.py --num-tasks 8 --num-rollouts 2 --depth 4
uv run load_test.py --build-workers 4
uv run load_test.py --timeout 30 --retries 5 --retry-backoff 0.1
```

The eviction test creates enough data to exceed the 2 MB high watermark, warms a subset of tasks with repeated exact `FIND` calls, waits for the eviction loop, and checks that hot tasks are retained at least as well as cold tasks. It also runs a post-eviction exact `FIND` on every hot path and checks that each node's cache-hit counter increases. Its test settings are top-level constants in `test_eviction.py`. Run it against a fresh server because eviction operates on the whole cache, not only the test-created task prefix.

The TTL eviction test inserts a root, an expiring child with a four-node subtree, and a non-expiring sibling. It waits for the eviction loop and checks that the expired subtree is removed while the sibling remains readable. Its settings are top-level constants in `test_ttl_eviction.py`.

The visualizer calls the same `DUMP_ALL` endpoint and renders one TCG at a time. Use the task selector to switch between TCGs and the refresh button to pull the latest state from the Rust server.

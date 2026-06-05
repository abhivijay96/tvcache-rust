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

The visualizer calls the same `DUMP_ALL` endpoint and renders one TCG at a time. Use the task selector to switch between TCGs and the refresh button to pull the latest state from the Rust server.

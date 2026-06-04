# Python Protocol Smoke Test

Start the Rust server in one terminal:

```sh
NUM_THREADS=4 cargo run
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

The load test creates 32 task ids by default. For each task it builds a valid TCG with depth 5, inserting parents before children, and randomly creates 1-8 children per node at each depth. It also samples leaf paths and calls `FIND` to verify inserted values. After the dataset is created, it runs parallel `FIND` batches at concurrency 32, 64, and 128 and logs mean, median, and p95 response completion times.

Dataset creation uses fewer workers than the find benchmark by default. Each request opens a short-lived TCP connection, so building the dataset with too many workers can exhaust local ephemeral ports before the benchmark starts.

Useful load-test options:

```sh
uv run load_test.py --seed 99 --workers 8
uv run load_test.py --tasks 8 --depth 4 --children-max 4
uv run load_test.py --find-concurrency 32,64,128
uv run load_test.py --workers 4 --insert-delay 0.001
uv run load_test.py --timeout 30 --retries 5 --retry-backoff 0.1
```

The visualizer calls the same `DUMP_ALL` endpoint and renders one TCG at a time. Use the task selector to switch between TCGs and the refresh button to pull the latest state from the Rust server.

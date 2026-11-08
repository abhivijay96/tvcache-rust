#!/usr/bin/env python3
"""Run live checks and the full load matrix on fresh local servers.

Build release binaries first. The sibling tvclient checkout is used via
PYTHONPATH. Requires port 7878 to be free. No real sandbox service is used.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-workers", type=int, default=32)
    parser.add_argument("--checks", nargs="+", help="optional subset of check names")
    parser.add_argument("--include-legacy-client", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        ("protocol", "none", ["test_client.py"]),
        ("sandbox-lifecycle", "ttl_and_watermark", ["test_sandbox_lifecycle.py"]),
        ("tvclient-current", "none", ["test_tvclient_current.py"]),
        ("ttl", "ttl_and_watermark", ["test_ttl_eviction.py"]),
        ("watermark", "ttl_and_watermark", ["test_eviction.py"]),
    ]
    if args.include_legacy_client:
        cases.append(("tvclient-legacy", "none", ["test_tvclient.py"]))
    for tasks in (32, 64, 128):
        cases.append((f"load-{tasks}", "none", ["load_test.py", "--num-tasks", str(tasks),
                      "--num-rollouts", "2,4,8,16", "--timeout", "60", "--build-workers",
                      str(args.build_workers), "--task-prefix", f"compatibility-{time.time_ns()}-{tasks}"]))
    names = {name for name, _, _ in cases}
    if args.checks and set(args.checks) - names:
        parser.error(f"unknown checks: {sorted(set(args.checks) - names)}")
    checks = []
    for name, policy, command in cases:
        if args.checks and name not in args.checks:
            continue
        # Binding checks availability without a potentially slow failed connect.
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", 7878))
        start = time.monotonic()
        with (out / f"{name}.server.log").open("w") as server_log:
            server = subprocess.Popen([str(ROOT / "target/release/tvcache"),
                f"--eviction-policy={policy}", "--high-watermark-mb=2", "--low-watermark-mb=1"],
                cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT,
                env={**os.environ, "RUST_LOG": "warn"})
            try:
                for attempt in range(100):
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited: {server.returncode}")
                    with socket.socket() as sock:
                        sock.settimeout(0.5)
                        if sock.connect_ex(("127.0.0.1", 7878)) == 0:
                            break
                    time.sleep(0.05)
                else:
                    raise RuntimeError("Server did not become ready")
                with (out / f"{name}.log").open("w") as log:
                    result = subprocess.run([sys.executable, "-u", *command], cwd=ROOT / "tests",
                        stdout=log, stderr=subprocess.STDOUT, timeout=1200,
                        env={**os.environ, "PYTHONPATH": str(ROOT.parent / "tvclient")})
                checks.append({"check": name, "exit_code": result.returncode,
                    "elapsed_seconds": time.monotonic() - start, "command": command,
                    "eviction_policy": policy, "build_workers": args.build_workers if name.startswith("load-") else None})
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        (out / "server-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(f"{name}: exit {result.returncode}", flush=True)
    return int(any(c["exit_code"] != 0 for c in checks))


if __name__ == "__main__":
    sys.exit(main())

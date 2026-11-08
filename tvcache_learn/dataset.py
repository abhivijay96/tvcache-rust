"""Join candidate dumps with each trace's future to build labelled arrays.

python -m tvcache_learn.dataset DUMP_DIR OUT_DIR

Writes OUT_DIR/<split>.npz for each split subdirectory of DUMP_DIR. Only raw
quantities are stored (feature vectors, next-access distance, lost time, group
ids), so training targets are chosen at train time without rebuilding.

Labels:
- distance: events from the eviction to the candidate root's next access. A
  rollout reaches any node through its whole stateful prefix, so the root's
  next access is also the first future use of anything in the removed subtree.
  Roots never accessed again are censored at the end of the trace.
- loss (dumps made with --affected): for each horizon H in HORIZONS, the tool
  runtime (ms) of every removed entry that is requested again within H events,
  i.e. the work this eviction would cost if nothing were re-admitted. NaN when
  the dump has no subtree IDs.
"""

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np

HORIZONS = (250, 1000, 4000, 16000)
SCALARS = ("chosen", "removes_incoming", "event_index", "num_candidates",
           "reclaimable_bytes", "capacity_bytes")


def trace_index(trace_dir):
    """node_id -> sorted access indices, node_id -> runtime_ms, event count."""
    positions, count = {}, 0
    with (Path(trace_dir) / "events.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            positions.setdefault(event["node_id"], []).append(event["event_index"])
            count += 1
    runtime = {}
    with (Path(trace_dir) / "nodes.jsonl").open() as stream:
        for line in stream:
            node = json.loads(line)
            runtime[node["node_id"]] = node["runtime_ms"]
    return positions, runtime, count


def read_dump(path, index, group_offset):
    """One dump file as numpy arrays (converted per file to bound memory)."""
    positions, runtime, num_events = index
    horizons = np.asarray(HORIZONS)
    x, scalars = [], {k: [] for k in SCALARS}
    group, distance, censored, loss = [], [], [], []
    with path.open() as stream:
        header = json.loads(next(stream))
        if header.get("format") != "tvcache-candidates-v1":
            raise ValueError(f"{path}: not a candidate dump")
        for line in stream:
            row = json.loads(line)
            now = row["event_index"]
            seen = positions.get(row["node_id"], [])
            k = bisect_right(seen, now)
            gone = k == len(seen)
            # Censored roots are placed beyond the end of the trace.
            distance.append((2 * num_events - now) if gone else seen[k] - now)
            censored.append(gone)
            if "affected" in row:
                lost = np.zeros(len(HORIZONS))
                for node in row["affected"]:
                    hits = positions.get(node, [])
                    j = bisect_right(hits, now)
                    if j < len(hits):
                        lost += runtime[node] * (hits[j] - now <= horizons)
                loss.append(lost)
            else:
                loss.append(np.full(len(HORIZONS), np.nan))
            x.append(row["x"])
            group.append(group_offset + row["eviction"])
            for key in SCALARS:
                scalars[key].append(row[key])
    if not x:
        return header, None
    arrays = {"x": np.asarray(x, dtype=np.float32), "group": np.asarray(group),
              "distance": np.asarray(distance), "censored": np.asarray(censored),
              "loss": np.asarray(loss, dtype=np.float32),
              **{k: np.asarray(v) for k, v in scalars.items()}}
    arrays["num_events"] = np.full(len(x), num_events)
    return header, arrays


def build(dump_dir, out_dir):
    dump_dir, out_dir = Path(dump_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indexes = {}
    feature_names = None
    for split_dir in sorted(p for p in dump_dir.iterdir() if p.is_dir()):
        parts, sources, offset = [], [], 0
        for path in sorted(split_dir.glob("*.jsonl")):
            with path.open() as stream:
                trace_dir = json.loads(next(stream))["trace_dir"]
            if trace_dir not in indexes:
                indexes[trace_dir] = trace_index(trace_dir)
            header, arrays = read_dump(path, indexes[trace_dir], offset)
            if arrays is None:
                continue
            if feature_names is None:
                feature_names = header["feature_names"]
            elif header["feature_names"] != feature_names:
                raise ValueError(f"{path}: feature names differ from earlier dumps")
            offset = int(arrays["group"].max()) + 1
            arrays["source"] = np.full(len(arrays["x"]), len(sources))
            sources.append({"dump": str(path), "trace_dir": trace_dir,
                            "policy": header["behavior_policy"],
                            "capacity_mib": header["capacity_mib"]})
            parts.append(arrays)
        if not parts:
            continue
        arrays = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        np.savez_compressed(out_dir / f"{split_dir.name}.npz", **arrays)
        (out_dir / f"{split_dir.name}.sources.json").write_text(json.dumps(sources, indent=2))
        groups = len(np.unique(arrays["group"]))
        labelled = float(np.mean(~np.isnan(arrays["loss"][:, 0])))
        print(f"{split_dir.name:<5} {len(arrays['x']):>9} rows  {groups:>7} evictions  "
              f"{len(sources)} dumps  censored {arrays['censored'].mean():.1%}  "
              f"loss-labelled {labelled:.0%}", flush=True)
    if feature_names is None:
        raise ValueError(f"no candidate rows found under {dump_dir}")
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names))
    (out_dir / "horizons.json").write_text(json.dumps(HORIZONS))


def load(data_dir, split):
    data = np.load(Path(data_dir) / f"{split}.npz")
    names = json.loads((Path(data_dir) / "feature_names.json").read_text())
    return {k: data[k] for k in data.files}, names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dumps", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    build(args.dumps, args.out)


if __name__ == "__main__":
    main()

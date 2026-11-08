"""Train a retention scorer and export it for the Rust MlpPolicy.

python -m tvcache_learn.train DATA_DIR --model small --target reuse:256 --out models/small.json

Models: linear (no hidden layer), small (16), big (128-128-64), or --hidden.
Higher scores mean "keep"; the policy evicts the lowest score.

Targets from next-access distance d (events):
  reuse:H   binary "reused within H events" (BCE on a logit)
  logdist   regress -log1p(d), so sooner reuse scores higher
  belady    per eviction, softmax over -score picks the candidate with the
            largest d (imitating Belady's farthest-next-use victim)
Cost-aware targets from lost time L_H (ms of tool work the eviction removes and
that is requested again within H events; dumps made with --affected):
  loss:H    regress log1p(L_H); pair with --output exp_per_byte for time lost
            per byte freed
  minloss:H per eviction, softmax over -score picks the candidate(s) with the
            smallest L_H (evict what costs least)
H for loss targets must be one of the dataset's horizons (250/1000/4000/16000).

Validation reports victim accuracy (the lowest-scored candidate has the largest
d), victim quality (mean log1p(d_victim) / log1p(d_max); 1.0 is Belady on the
sampled candidates) and, when lost-time labels exist, victim_loss_ms (mean L_H
of the evicted candidate at the target's H, or 1000). Cost targets keep the
epoch with the lowest victim_loss_ms; the others the highest victim quality.
A linear model on basic features is also written as LinearPolicy weights.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import HORIZONS, load

PRESETS = {"linear": [], "small": [16], "big": [128, 128, 64]}
BASIC = 9
LINEAR_WEIGHT_NAMES = ["frequency", "recency", "runtime", "dollar_cost", "inverse_size",
                       "depth", "branching", "subtree_frequency", "subtree_recency"]


def make_model(inputs, hidden):
    layers, width = [], inputs
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def grouped(data, columns):
    """Pad rows into [groups, max candidates, ...] tensors with a mask."""
    order = np.argsort(data["group"], kind="stable")
    group = data["group"][order]
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    sizes = np.diff(np.r_[starts, len(group)])
    width = int(sizes.max())
    slot = np.arange(len(group)) - np.repeat(starts, sizes)
    row = np.repeat(np.arange(len(starts)), sizes)
    mask = np.zeros((len(starts), width), dtype=bool)
    mask[row, slot] = True
    out = {"mask": torch.from_numpy(mask)}
    for name, values in columns.items():
        values = values[order]
        padded = np.zeros((len(starts), width, *values.shape[1:]), dtype=values.dtype)
        padded[row, slot] = values
        out[name] = torch.from_numpy(padded)
    return out


def victim_metrics(scores, distance, mask, loss=None):
    """Lowest score is evicted; compare with the farthest next use in each group.

    `loss` is the [groups, candidates] lost time at one horizon, if labelled.
    """
    scores = scores.masked_fill(~mask, math.inf)
    victim = scores.argmin(dim=1)
    d = distance.double().masked_fill(~mask, -1)
    best = d.max(dim=1).values
    chosen = d.gather(1, victim[:, None]).squeeze(1)
    quality = torch.log1p(chosen) / torch.log1p(best).clamp(min=1e-12)
    out = {"victim_accuracy": float((chosen == best).double().mean()),
           "victim_quality": float(quality.mean())}
    if loss is not None and not torch.isnan(loss[mask]).any():
        lost = loss.double().gather(1, victim[:, None]).squeeze(1)
        least = loss.double().masked_fill(~mask, math.inf).min(dim=1).values
        out["victim_loss_ms"] = float(lost.mean())
        out["least_loss_ms"] = float(least.mean())
    return out


def horizon_of(target):
    """Index into HORIZONS used for lost-time labels and the cost metric."""
    h = int(target.split(":", 1)[1]) if target.startswith(("loss:", "minloss:")) else 1000
    if h not in HORIZONS:
        raise ValueError(f"loss horizon must be one of {HORIZONS}")
    return HORIZONS.index(h)


def target_loss(target, scores, batch):
    mask, d = batch["mask"], batch["distance"]
    if target.startswith("loss:"):
        y = torch.log1p(batch["loss"][..., horizon_of(target)])
        return ((scores - y) ** 2)[mask].mean()
    if target.startswith("minloss:"):
        logits = (-scores).masked_fill(~mask, -math.inf)
        lost = batch["loss"][..., horizon_of(target)].masked_fill(~mask, math.inf)
        best = (lost == lost.min(dim=1, keepdim=True).values) & mask
        return -(torch.logsumexp(logits.masked_fill(~best, -math.inf), 1)
                 - torch.logsumexp(logits, 1)).mean()
    if target == "belady":
        logits = (-scores).masked_fill(~mask, -math.inf)
        far = d.masked_fill(~mask, -1)
        best = far == far.max(dim=1, keepdim=True).values
        return -(torch.logsumexp(logits.masked_fill(~best, -math.inf), 1)
                 - torch.logsumexp(logits, 1)).mean()
    if target == "logdist":
        y = -torch.log1p(d.float())
        return ((scores - y) ** 2)[mask].mean()
    horizon = int(target.split(":", 1)[1])
    y = ((d <= horizon) & ~batch["censored"]).float()
    return nn.functional.binary_cross_entropy_with_logits(scores[mask], y[mask])


def export(model, mean, std, names, features, args, metrics):
    linears = [m for m in model if isinstance(m, nn.Linear)]
    layers = [{"weight": m.weight.detach().double().tolist(),
               "bias": m.bias.detach().double().tolist(),
               "activation": "none" if i == len(linears) - 1 else "relu"}
              for i, m in enumerate(linears)]
    return {"format": "tvcache-mlp-v1", "name": args.name or f"mlp-{args.model}",
            "features": features, "feature_names": names,
            "input_mean": mean.tolist(), "input_std": std.tolist(), "layers": layers,
            "output": args.output, "protect_incoming": args.protect_incoming,
            "max_candidates": args.max_candidates,
            "metadata": {"target": args.target, "hidden": [m.out_features for m in linears[:-1]],
                         "data": str(args.data), "epochs_run": metrics.pop("epochs_run"),
                         "seed": args.seed, "validation": metrics}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", choices=PRESETS, default="small")
    parser.add_argument("--hidden", type=int, nargs="*", help="override the preset's hidden widths")
    parser.add_argument("--features", choices=["basic", "rich"],
                        help="default: basic for linear/small, rich for big")
    parser.add_argument("--target", default="belady",
                        help="reuse:H | logdist | belady | loss:H | minloss:H")
    parser.add_argument("--output", choices=["score", "prob_benefit_per_byte", "exp_per_byte"],
                        default="score")
    parser.add_argument("--protect-incoming", action="store_true")
    parser.add_argument("--max-candidates", type=int,
                        help="policy scores a deterministic sample of this many candidates per eviction")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256, help="evictions per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    kind, _, h = args.target.partition(":")
    if not (args.target in ("belady", "logdist") or
            (kind in ("reuse", "loss", "minloss") and h.isdigit())):
        parser.error("--target must be reuse:H, logdist, belady, loss:H or minloss:H")
    cost_target = kind in ("loss", "minloss")
    try:
        horizon = horizon_of(args.target)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output == "prob_benefit_per_byte" and kind != "reuse":
        parser.error("prob_benefit_per_byte expects a reuse:H logit")
    if args.output == "exp_per_byte" and kind != "loss":
        parser.error("exp_per_byte expects a loss:H regression")
    features = args.features or ("rich" if args.model == "big" else "basic")
    hidden = PRESETS[args.model] if args.hidden is None else args.hidden
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train, all_names = load(args.data, "train")
    val, _ = load(args.data, "val")
    if cost_target and (np.isnan(train["loss"]).any() or np.isnan(val["loss"]).any()):
        parser.error("loss targets need every dump collected with --affected")
    width = BASIC if features == "basic" else len(all_names)
    names = all_names[:width]
    mean = train["x"][:, :width].astype(np.float64).mean(0)
    std = train["x"][:, :width].astype(np.float64).std(0)
    std[std < 1e-8] = 1.0  # constant features pass through unscaled

    def tensors(data):
        x = ((data["x"][:, :width] - mean) / std).astype(np.float32)
        return grouped(data, {"x": x, "distance": data["distance"], "censored": data["censored"],
                              "chosen": data["chosen"], "loss": data["loss"]})

    train_t, val_t = tensors(train), tensors(val)
    model = make_model(width, hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def evaluate(t, chunk=20000):
        # Chunked so large validation sets fit in memory with the big model.
        model.eval()
        with torch.no_grad():
            scores = torch.cat([model(x).squeeze(-1) for x in t["x"].split(chunk)])
            losses = [float(target_loss(args.target, scores[i:i + chunk],
                                        {k: v[i:i + chunk] for k, v in t.items()}))
                      for i in range(0, len(scores), chunk)]
        return {"loss": float(np.mean(losses)),
                **victim_metrics(scores, t["distance"], t["mask"], t["loss"][..., horizon])}

    def selection(metrics):
        # Higher is better: least lost time for cost targets, else Belady quality.
        return -metrics["victim_loss_ms"] if cost_target else metrics["victim_quality"]

    def describe(m):
        text = f"victim_accuracy {m['victim_accuracy']:.3f}  victim_quality {m['victim_quality']:.3f}"
        if "victim_loss_ms" in m:
            text += f"  victim_loss_ms {m['victim_loss_ms']:.1f} (least {m['least_loss_ms']:.1f})"
        return text

    behavior = victim_metrics((~val_t["chosen"]).float(), val_t["distance"], val_t["mask"],
                              val_t["loss"][..., horizon])
    print(f"model {args.model} hidden={hidden} features={features} ({width}) target={args.target}")
    print(f"train {train_t['mask'].shape[0]} evictions, val {val_t['mask'].shape[0]} evictions")
    print(f"behavior policy on val: {describe(behavior)}")
    best, best_state, stale, epochs = -math.inf, None, 0, 0
    groups = train_t["mask"].shape[0]
    for epoch in range(args.epochs):
        model.train()
        for idx in torch.randperm(groups).split(args.batch):
            batch = {k: v[idx] for k, v in train_t.items()}
            loss = target_loss(args.target, model(batch["x"]).squeeze(-1), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = evaluate(val_t)
        epochs = epoch + 1
        print(f"epoch {epochs:>3}  val loss {metrics['loss']:.4f}  {describe(metrics)}", flush=True)
        if selection(metrics) > best:
            best, best_state, stale = selection(metrics), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    metrics = {**evaluate(val_t), "behavior": behavior, "epochs_run": epochs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(export(model, mean, std, names, features, args, metrics), indent=1) + "\n")
    print(f"best val: {describe(metrics)}; wrote {args.out}")
    if not hidden and features == "basic" and args.output == "score":
        # Undo standardization: the mean shift is a constant, which ranking ignores.
        w = model[0].weight.detach().double().numpy()[0] / std
        weights = dict(zip(LINEAR_WEIGHT_NAMES, w.tolist()))
        path = args.out.with_suffix(".linear-weights.json")
        path.write_text(json.dumps(weights, indent=2) + "\n")
        print(f"also wrote LinearPolicy weights {path}")


if __name__ == "__main__":
    main()

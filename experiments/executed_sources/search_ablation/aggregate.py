"""Fixed-cohort ablation tables and all paired differences, without score selection."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from runner import ROOT, read, resolve, sha, subset, validate, verify_seal, write


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def run_cost(data, result_path):
    result = {key: data.get(key) for key in ("train_seconds", "eval_seconds", "seconds", "peak_train_allocated_GiB",
              "peak_allocated_GiB", "updates", "global_batches", "groups", "skipped", "local_ties_total", "seed")}
    logs = sorted(Path(result_path).parent.glob("rank*/train.jsonl"))
    if logs:
        totals = {key: 0.0 for key in ("judge_seconds", "metric_seconds", "candidate_seconds")}
        available = set()
        count = 0
        for log in logs:
            for line in log.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                for key in totals:
                    if row.get(key) is not None:
                        value = float(row[key])
                        assert math.isfinite(value) and value >= 0
                        totals[key] += value
                        available.add(key)
        result.update(actual_logged_groups=count, **{key + "_sum": totals[key] if key in available else None for key in totals})
    result["cost_note"] = "Worker-time sums are not wall time. Allocator peaks exclude separate judge processes. Missing measurements remain null."
    return result


def aggregate(root=ROOT):
    root = Path(root).resolve()
    started = time.monotonic()
    plan = read(root / "plan.json")
    seal = verify_seal(root)
    destination = root / plan.get("summary_filename", "summary.json")
    if destination.exists():
        assert validate(root, {"kind": "aggregate", "result": str(destination)})
        existing = read(destination)
        assert existing["plan_sha256"] == sha(root / "plan.json")
        print("SUMMARY_ALREADY_VERIFIED", destination, flush=True)
        return existing
    groups = plan["comparisons"]
    assert isinstance(groups, list) and groups
    assert len({group["name"] for group in groups}) == len(groups)
    horizons = (10, 20, 32)
    draws, seed = 5000, 20261015
    # Same resampled trajectory indices for every arm/horizon/metric, preserving pairing.
    bootstrap_indices = np.random.default_rng(seed).integers(0, 128, (draws, 128))
    report = dict(state="SUCCEEDED", scope=plan.get("scope"), groups={}, input_results=[],
                  plan_sha256=sha(root / "plan.json"), source_manifest=seal,
                  bootstrap=dict(draws=draws, seed=seed, unit="paired trajectory", interval="percentile 95%",
                                 limitation="One training seed; trajectory intervals do not estimate training-seed variability."),
                  limitations=plan.get("limitations", []))
    loaded = {}
    mean_rows, delta_rows = [], []
    for group in groups:
        arms = group["arms"]
        labels = [arm["label"] for arm in arms]
        assert len(arms) >= 2 and len(set(labels)) == len(labels)
        matrices, metadata, costs, identity = {}, {}, {}, None
        for arm in arms:
            path = resolve(root, arm["result"])
            if str(path) not in loaded:
                stage = dict(arm, kind="existing", result=str(path))
                assert validate(root, stage), f"Missing comparison arm: {path}"
                loaded[str(path)] = read(path)
                report["input_results"].append({"path": str(path), "sha256": sha(path)})
            data = loaded[str(path)]
            subset(data, arm.get("expected", {}), arm["label"])
            subset(data.get("protocol", {}), arm.get("expected_protocol", {}), arm["label"] + ".protocol")
            rows = data["evaluation"]
            pairs = [(row["id"], row["prepared_input_sha256"]) for row in rows]
            if identity is None:
                identity = pairs
            assert pairs == identity, f"Cohort/input mismatch in {group['name']} / {arm['label']}"
            label = arm["label"]
            metric_names = set(rows[0]["per_frame"])
            assert all(set(row["per_frame"]) == metric_names for row in rows)
            matrices[label] = {str(h): {m: np.array([np.mean(row["per_frame"][m][:h]) for row in rows], dtype=np.float64)
                                     for m in sorted(metric_names)} for h in horizons}
            metadata[label] = {**arm, "result": str(path), "result_sha256": sha(path),
                               "initial_digest": data.get("initial_digest"), "final_digest": data.get("final_digest"),
                               "protocol": data.get("protocol"), "updates": data.get("updates", data.get("rl_updates")),
                               "seed": data.get("seed")}
            costs[label] = run_cost(data, path)
            for h in horizons:
                for metric, values in matrices[label][str(h)].items():
                    mean_rows.append([group["name"], label, arm.get("family", ""), h, metric, float(values.mean()), 128])
        for field in group.get("match_fields", []):
            values = []
            for arm in arms:
                value = loaded[str(resolve(root, arm["result"]))]
                for key in field.split("."):
                    assert isinstance(value, dict) and key in value, (group["name"], field, "missing")
                    value = value[key]
                values.append(value)
            assert all(value == values[0] for value in values), (group["name"], field, "not matched")
        for h in horizons:
            assert len({tuple(matrices[label][str(h)]) for label in labels}) == 1, "Metrics differ across arms"
        means = {label: {h: {metric: float(values.mean()) for metric, values in values_by_metric.items()}
                        for h, values_by_metric in by_horizon.items()} for label, by_horizon in matrices.items()}
        comparisons = []
        # Every pair follows explicit arm order. Positive delta means later-listed minus earlier-listed.
        for baseline, treatment in itertools.combinations(labels, 2):
            item = dict(baseline=baseline, treatment=treatment, n=128, delta_definition="treatment minus baseline", horizons={})
            for h in horizons:
                values = {}
                for metric in matrices[treatment][str(h)]:
                    delta = matrices[treatment][str(h)][metric] - matrices[baseline][str(h)][metric]
                    samples = delta[bootstrap_indices].mean(axis=1)
                    interval = np.quantile(samples, [.025, .975]).tolist()
                    tolerance = 1e-12
                    stat = dict(delta=float(delta.mean()), paired_trajectory_ci95=interval,
                                positive=int(np.sum(delta > tolerance)), tied=int(np.sum(np.abs(delta) <= tolerance)),
                                negative=int(np.sum(delta < -tolerance)))
                    values[metric] = stat
                    delta_rows.append([group["name"], treatment, baseline, h, metric, stat["delta"], *interval,
                                       stat["positive"], stat["tied"], stat["negative"], 128])
                item["horizons"][str(h)] = values
            comparisons.append(item)
        report["groups"][group["name"]] = dict(description=group.get("description"), semantics=group.get("semantics"),
            arm_order=labels, arms=metadata, means=means, cost=costs, comparisons=comparisons,
            cohort={"n": 128, "ordered_ids": [v[0] for v in identity],
                    "ordered_id_input_sha256": hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()})
    report["stage_receipts"] = [read(path) for path in sorted((root / "stage_receipts").glob("*.json"))]
    status_path = root / "status.json"
    current_status = read(status_path) if status_path.exists() else {}
    report["runtime"] = dict(aggregation_seconds=time.monotonic()-started,
        queue_elapsed_at_aggregation_start_seconds=current_status.get("elapsed_seconds"),
        queue_started_at_unix=current_status.get("started_at_unix"),
        completed_stage_runtime_seconds=sum(r.get("elapsed_seconds", 0) for r in report["stage_receipts"]),
        final_runtime_receipt="runner_complete.json",
        note="The runner records total queue wall time and final aggregation-stage receipt after this process returns.")
    outputs = {}
    for filename, header, rows in [
        ("summary.csv", ["experiment", "arm", "family", "horizon", "metric", "mean", "n"], mean_rows),
        ("paired_deltas.csv", ["experiment", "treatment", "baseline", "horizon", "metric", "delta", "ci95_low", "ci95_high", "positive", "tied", "negative", "n"], delta_rows),
    ]:
        path = root / filename
        assert not path.exists(), f"Partial output preserved; refusing overwrite: {path}"
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
        atomic_text(path, stream.getvalue())
        outputs[filename] = sha(path)
    report["output_files"] = outputs
    verify_seal(root)
    for item in report["input_results"]:
        assert sha(item["path"]) == item["sha256"], "Input result changed during aggregation"
    write(destination, report)
    print("SUMMARY_COMPLETE", destination, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    aggregate(args.root)

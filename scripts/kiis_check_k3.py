"""Check synced K3 weights, validation gates, and dev policy pairing.

Run after kiis_validate_wm.py and the B0_typed/B_typed/D_gen dev smokes.
Only reads existing artifacts; writes validation/gates.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "artifacts/kiis_k3"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    validation = BASE / "validation"
    sync = read(BASE / "sync_manifest.json")
    matches = {name: (BASE / name).stat().st_size == spec["bytes"]
               and sha(BASE / name) == spec["sha256"]
               for name, spec in sync["files"].items()}
    source_hashes = read(validation / "source_hashes.json")
    source_matches = {name: sha(ROOT / name) == expected
                      for name, expected in source_hashes.items()}
    reports = {arm: read(validation / f"{arm}.json") for arm in ("B_typed", "D_gen")}
    meta = {arm: read(BASE / arm / "train_meta.json") for arm in reports}
    manifests, episodes, initial = {}, {}, {}
    for arm in ("B0_typed", "B_typed", "D_gen"):
        path = validation / f"smoke_{arm}"
        manifests[arm] = read(path / "run_manifest.json")
        eps = rows(path / "episodes.jsonl")
        assert len(eps) == 1, (arm, len(eps))
        episodes[arm] = eps[0]
        starts = [r for r in rows(path / "steps.jsonl") if r["step"] == 0]
        assert len(starts) == 1, (arm, len(starts))
        initial[arm] = starts[0]
    b, d = (meta[arm] for arm in ("B_typed", "D_gen"))
    shared = ("n", "seed", "train", "val", "eval_n", "epochs", "lr", "micro_tokens", "accum", "model")
    gates = {
        "training_files_synced": all(matches.values()),
        "local_validation_inputs_match_server": all(source_matches.values()),
        "validation_report_inputs": all(
            r["val_sha256"] == source_hashes["data/kiis/transitions/val.jsonl"]
            and r["script_sha256"] == source_hashes["scripts/kiis_validate_wm.py"]
            for r in reports.values()),
        "same_training_data_and_settings": b["data_sha"] == d["data_sha"]
            and b["lora"] == d["lora"]
            and all(b["args"][k] == d["args"][k] for k in shared),
        "planned_training_size": all(m["n_transitions"] == 6000 and len(m["epochs"]) == 2 for m in meta.values()),
        "same_validation_subset": reports["B_typed"]["subset_sha256"] == reports["D_gen"]["subset_sha256"]
            and all(r["transitions"] == 200 for r in reports.values()),
        "validation_gates": all(all(r["gates"].values()) for r in reports.values()),
        "dev_manifests_finished": all(m["run"].get("finished") and m["counts"]["episodes"] == 1
                                      for m in manifests.values()),
        "dev_episodes_completed": all(e["status"] in ("success", "cap") for e in episodes.values()),
        "same_frozen_condition": {m["config_hash"] for m in manifests.values()} == {"0fe479090329"},
        "same_dev_world": all(m["world_fingerprints"] == manifests["B0_typed"]["world_fingerprints"] for m in manifests.values())
            and all(m["run"]["split"] == "dev" for m in manifests.values()),
        "same_policy_seed": {m["run"]["policy_seed"] for m in manifests.values()} == {20260921},
        "same_step0_candidates": all(r["plans"] == initial["B0_typed"]["plans"] for r in initial.values()),
        "selected_adapters_used": all(manifests[arm]["run"]["arm_configs"][arm]["adapter"]["weights_sha256"]
             == reports[arm]["results"]["selected"]["weights_sha256"] for arm in reports),
    }
    summary = {"gates": gates, "training_files": matches, "source_files": source_matches,
               "validation": reports, "smoke_episodes": episodes,
               "step0_plans": initial["B0_typed"]["plans"],
               "gpu_hours": {arm: m["gpu_hours"] for arm, m in meta.items()}}
    (validation / "gates.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(gates, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

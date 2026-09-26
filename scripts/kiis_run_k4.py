"""Launch the frozen K4 matrix: four concurrent jobs per seed, then next seed.

The smoke run uses one world/root only to check execution contracts. It writes
to a separate directory and never contributes to the full results. v1's smoke
used a test world; v2's uses dev (kiis2026f/실험계획.md §11).

--version v2 runs condition F2 (kiis2026f/실험계획.md §4 V2): the calibrated
task and policy, plus C_fm, with the adapters retrained in V3. A v2 smoke may
borrow other adapters (--smoke-adapters) to check the pipeline before V3; its
B/D numbers then mean nothing and are labelled as such.

--jobs runs a subset of the four jobs (and --gpu puts them all on one GPU), so
jobs that do not need a model still training can start early, e.g. v2's
zero_shot and generative on GPU1 while B trains on GPU0 (kiis2026f/실험계획.md
§4 V4). Jobs are independent and seeded, so the order they run in does not
change their results. A later subset adds its job directories next to the
earlier ones; an existing job directory is never overwritten. --dry-run prints
the plan and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEEDS = (20260921, 777, 1234)
CONFIG_HASH = "0fe479090329"
JOBS = (
    ("core", 0, "C,validity,oracle,A_jev", None),
    ("generative", 0, "D_gen", "artifacts/kiis_k3/D_gen"),
    ("zero_shot", 1, "B0_typed,D0_gen", None),
    ("typed", 1, "B_typed", "artifacts/kiis_k3/B_typed"),
)
V2 = {
    "config_hash": "c7a604152a04",
    "args": ["--v2", "--levers", "T3", "--policy-hint-carry-key", "--policy-samples", "2"],
    "jobs": (
        ("core", 0, "C,C_fm,validity,oracle,A_jev", None),
        ("generative", 0, "D_gen", "artifacts/kiis_k3v2/D_gen"),
        ("zero_shot", 1, "B0_typed,D0_gen", None),
        ("typed", 1, "B_typed", "artifacts/kiis_k3v2/B_typed"),
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, status: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--version", choices=("v1", "v2"), default="v1")
    ap.add_argument("--smoke-adapters", default=None,
                    help="v2 smoke only: directory with B_typed/ and D_gen/ to borrow")
    ap.add_argument("--jobs", default=None, help="comma list of job names (default: all four)")
    ap.add_argument("--gpu", type=int, default=None, help="run every selected job on this GPU")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    v2 = args.version == "v2"
    if args.smoke_adapters and not (v2 and smoke):
        raise SystemExit("--smoke-adapters is for a v2 smoke only")
    config_hash = V2["config_hash"] if v2 else CONFIG_HASH
    jobs = V2["jobs"] if v2 else JOBS
    if args.smoke_adapters:
        jobs = tuple((n, g, a, f"{args.smoke_adapters}/{a}" if ad else None)
                     for n, g, a, ad in jobs)
    names = [j[0] for j in jobs]
    selected = names if args.jobs is None else [x.strip() for x in args.jobs.split(",") if x.strip()]
    unknown = set(selected) - set(names)
    if unknown or not selected:
        raise SystemExit(f"unknown or empty --jobs: {sorted(unknown)}; choose from {names}")
    jobs = tuple(j for j in jobs if j[0] in selected)
    if args.gpu is not None:
        jobs = tuple((n, args.gpu, a, ad) for n, g, a, ad in jobs)
    split = "dev" if (v2 and smoke) else "test"
    name = "kiis_k4v2" if v2 else "kiis_k4"
    out = ROOT / "artifacts" / (f"{name}_smoke" if smoke else name)
    seeds = SEEDS[:1] if smoke else SEEDS
    label = "" if len(jobs) == len(names) else "_" + "+".join(j[0] for j in jobs)
    status_path = out / f"launch_status{label}.json"
    taken = [str(out / str(sd) / j[0]) for sd in seeds for j in jobs if (out / str(sd) / j[0]).exists()]
    if taken or status_path.exists():
        raise SystemExit(f"refusing to overwrite existing K4 output: {taken or status_path}")
    if args.dry_run:
        print(f"{out}  status {status_path.name}  split {split}  hash {config_hash}")
        for sd in seeds:
            for n, g, a, ad in jobs:
                print(f"  seed {sd}  {n:<10} GPU{g}  arms {a}  adapter {ad}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    status = {"mode": args.mode, "version": args.version, "split": split,
              "smoke_adapters": args.smoke_adapters, "jobs": [j[0] for j in jobs],
              "gpu_override": args.gpu, "phase": "running", "started": now(), "seeds": {}}
    write_status(status_path, status)

    for seed in seeds:
        seed_dir = out / str(seed)
        seed_dir.mkdir(exist_ok=True)
        processes: dict[str, tuple[subprocess.Popen, object, Path, int]] = {}
        status["seeds"][str(seed)] = {}
        for name, gpu, arms, adapter in jobs:
            job_dir = seed_dir / name
            log_path = seed_dir / f"{name}.log"
            cmd = [sys.executable, str(ROOT / "scripts" / "mvp_c_headroom.py"),
                   "--device", "cuda:0", "--trap", "--split", split,
                   "--worlds", "1" if smoke else "24",
                   "--roots", "1" if smoke else "4", "--cap", "30",
                   "--horizon", "2", "--policy-seed", str(seed),
                   "--budget", "0.1" if smoke else "3",
                   "--arms", arms, "--out", str(job_dir)] + (V2["args"] if v2 else [])
            if adapter:
                cmd += ["--adapter", str(ROOT / adapter)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = log_path.open("w")
            try:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            except Exception:
                log.close()
                for running, handle, _, _ in processes.values():
                    running.terminate()
                    handle.close()
                raise
            processes[name] = (proc, log, job_dir, len(arms.split(",")))
            status["seeds"][str(seed)][name] = {"pid": proc.pid, "gpu": gpu,
                                                  "started": now(), "phase": "running"}
            write_status(status_path, status)

        failed = False
        for name, (proc, log, job_dir, n_arms) in processes.items():
            code = proc.wait()
            log.close()
            job = status["seeds"][str(seed)][name]
            job.update({"finished": now(), "exit_code": code})
            manifest_path = job_dir / "run_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text())
                run = manifest["run"]
                expected = (1 if smoke else 24) * (1 if smoke else 4) * n_arms
                valid = (code == 0 and bool(run.get("finished"))
                         and manifest["config_hash"] == config_hash
                         and run["split"] == split and run["policy_seed"] == seed
                         and run["worlds"] == (1 if smoke else 24)
                         and run["roots"] == (1 if smoke else 4)
                         and manifest["counts"]["episodes"] == expected)
                job["episodes"] = manifest["counts"]["episodes"]
                job["config_hash"] = manifest["config_hash"]
                job["revision"] = manifest["revision"]["line"]
            except (OSError, KeyError, TypeError, ValueError) as exc:
                valid = False
                job["error"] = str(exc)
            job["phase"] = "complete" if valid else "failed"
            failed |= not valid
            write_status(status_path, status)
        if failed:
            status["phase"] = "failed"
            status["finished"] = now()
            write_status(status_path, status)
            return 1

    status["phase"] = "complete"
    status["finished"] = now()
    write_status(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

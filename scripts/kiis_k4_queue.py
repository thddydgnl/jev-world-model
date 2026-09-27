"""K4v2/K4v3 as a queue of (seed, job) units over GPU slots (kiis2026f/실험계획.md §4 V4, W3).

kiis_run_k4.py starts all of a seed's jobs together and waits for the slowest,
so a GPU slot sits idle while core (five arms, the slowest job) finishes. Here
every (seed, job) unit is queued on its own and a free slot takes the next one.
core is split into core_a (C, C_fm) and core_b (validity, oracle, A_jev): an
arm's episodes depend only on (world, root, step) seeds, never on which process
or which other arms ran them, so the split does not change any result
(checked in K1: step-0 candidates identical across processes and GPUs).

- slots: two per GPU (policy + world model share weights; about 8–12 GB each);
  --per-gpu sets the start value and a PER_GPU file in the output directory
  overrides it while running (v3: 1 while K5v3 shares the GPUs, then 2).
- --adopt seed:job:pid:gpu takes over a unit already running (started by
  kiis_run_k4.py): it holds a slot until the pid exits, then is validated.
- a GPU listed in --gate-gpu is used only once the gate file exists. GPU0 is
  opened by hand after B is trained and validated (and typed units need B too),
  so the next stage starts only after its inputs have been checked.
- every finished unit is validated like kiis_run_k4.py does (exit, finished
  manifest, condition hash, split, seed, 24 worlds x 4 roots x arms).
- a failed unit is recorded and the queue goes on; nothing is overwritten.

Status: artifacts/kiis_k4v2/queue_status.json. --dry-run prints the plan.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from kiis_run_k4 import CONFIGS, SEEDS  # noqa: E402

PER_GPU = 2
# per version: output, gate file, jobs that wait for the gate, units heaviest
# first. v2: typed needed the validated B adapter and GPU0 opened with it.
# v3 (§4 W3): nothing waits for a GPU; generative and typed wait until the v1
# adapters have passed the reuse check (ADAPTERS_READY).
VERSIONS = {
    "v2": {"gate": "GPU0_READY", "needs_gate": {"typed"}, "adapters": "artifacts/kiis_k3v2"},
    "v3": {"gate": "ADAPTERS_READY", "needs_gate": {"typed", "generative"},
           "adapters": "artifacts/kiis_k3"},
}
OUT = GATE = CFG = None
NEEDS_GATE: set = set()
UNITS: tuple = ()


def configure(version: str) -> None:
    global OUT, GATE, CFG, NEEDS_GATE, UNITS
    v = VERSIONS[version]
    CFG = CONFIGS[version]
    if CFG["config_hash"] is None:
        raise SystemExit(f"{version} is not frozen yet (kiis2026f/실험계획.md §4 W2)")
    OUT = ROOT / "artifacts" / f"kiis_k4{version}"
    GATE = OUT / v["gate"]
    NEEDS_GATE = set(v["needs_gate"])
    UNITS = (
        ("core_a", "C,C_fm", None),
        ("zero_shot", "B0_typed,D0_gen", None),
        ("core_b", "validity,oracle,A_jev", None),
        ("generative", "D_gen", f"{v['adapters']}/D_gen"),
        ("typed", "B_typed", f"{v['adapters']}/B_typed"),
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def alive(pid: int) -> bool:
    """Running and not a zombie. A process that has exited but was not yet
    reaped still answers kill(pid, 0), so its state is read as well."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                          capture_output=True, text=True).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def command(seed: int, arms: str, adapter: str | None, out: Path) -> list[str]:
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "mvp_c_headroom.py"),
           "--device", "cuda:0", "--trap", "--split", CFG.get("split", "test"),
           "--worlds", "24", "--roots", "4",
           "--cap", "30", "--horizon", "2", "--policy-seed", str(seed), "--budget", "3",
           "--arms", arms, "--out", str(out)] + CFG["args"]
    if adapter:
        cmd += ["--adapter", str(ROOT / adapter)]
    return cmd


def validate(unit: dict, code: int | None) -> tuple[bool, dict]:
    info: dict = {"exit_code": code}
    try:
        m = json.loads((Path(unit["out"]) / "run_manifest.json").read_text())
        run = m["run"]
        expected = 24 * 4 * len(unit["arms"].split(","))
        ok = ((code in (0, None)) and bool(run.get("finished"))
              and m["config_hash"] == CFG["config_hash"] and run["split"] == CFG.get("split", "test")
              and run["policy_seed"] == unit["seed"] and run["worlds"] == 24 and run["roots"] == 4
              and m["counts"]["episodes"] == expected)
        info.update(episodes=m["counts"]["episodes"], expected=expected,
                    config_hash=m["config_hash"], revision=m["revision"].get("line", ""),
                    jev_usd=m["counts"].get("jev_usd"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        ok, info["error"] = False, str(exc)
    return ok, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=tuple(VERSIONS), default="v2")
    ap.add_argument("--per-gpu", type=int, default=PER_GPU,
                    help="slots per GPU at start; a file PER_GPU in the output directory, "
                         "if present, overrides it while running (e.g. 1 while K5 shares the "
                         "GPUs, then 2 — written by hand, read every minute)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--gate-gpu", default=None,
                    help="GPUs used only once the gate file exists (default: 0 for v2, none for v3)")
    ap.add_argument("--adopt", action="append", default=[], help="seed:job:pid:gpu of a running unit")
    ap.add_argument("--skip", action="append", default=[], help="seed:job not to queue (already run)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configure(args.version)
    gpus = [int(g) for g in args.gpus.split(",")]
    gate_gpu = args.gate_gpu if args.gate_gpu is not None else ("0" if args.version == "v2" else "")
    gated = {int(g) for g in gate_gpu.split(",") if g != ""}

    adopted = []
    for spec in args.adopt:
        seed, job, pid, gpu = spec.split(":")
        adopted.append({"seed": int(seed), "job": job, "pid": int(pid), "gpu": int(gpu),
                        "out": str(OUT / seed / job), "adopted": True,
                        "arms": {"core": "C,C_fm,validity,oracle,A_jev",
                                 **{u[0]: u[1] for u in UNITS}}[job]})
    skip = {(int(s.split(":")[0]), s.split(":")[1]) for s in args.skip}
    skip |= {(a["seed"], a["job"]) for a in adopted}
    # a seed whose unsplit core is adopted or skipped needs no core_a/core_b
    whole_core = {s for s, j in skip if j == "core"}
    queue = []
    for name, arms, adapter in UNITS:
        for seed in SEEDS:
            if (seed, name) in skip or (name.startswith("core_") and seed in whole_core):
                continue
            out = OUT / str(seed) / name
            if out.exists():
                raise SystemExit(f"refusing to overwrite {out}")
            queue.append({"seed": seed, "job": name, "arms": arms, "adapter": adapter,
                          "out": str(out)})
    if args.dry_run:
        print(f"GPUs {gpus}, {args.per_gpu} slots each (override file {OUT.name}/PER_GPU), "
              f"gated {sorted(gated)} by {GATE.name}")
        for a in adopted:
            print(f"  adopt  seed {a['seed']:<9} {a['job']:<10} pid {a['pid']} GPU{a['gpu']} alive={alive(a['pid'])}")
        for u in queue:
            print(f"  queue  seed {u['seed']:<9} {u['job']:<10} arms {u['arms']}"
                  + (f"  adapter {u['adapter']}" if u["adapter"] else "")
                  + ("  (after gate)" if u["job"] in NEEDS_GATE else ""))
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "queue_status.json"
    status = {"started": now(), "phase": "running", "gpus": gpus, "gated": sorted(gated),
              "gate": str(GATE), "units": {}}
    def key(u):
        return f"{u['seed']}/{u['job']}"
    def save():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=1) + "\n")
        tmp.replace(status_path)
    for u in queue:
        status["units"][key(u)] = {"phase": "queued", "arms": u["arms"]}
    running = []                      # dicts with pid, gpu, proc (None if adopted), log
    for a in adopted:
        running.append({**a, "proc": None, "log": None, "started": "adopted"})
        status["units"][key(a)] = {"phase": "running (adopted)", "arms": a["arms"],
                                   "pid": a["pid"], "gpu": a["gpu"]}
    save()

    while queue or running:
        for r in list(running):
            code = r["proc"].poll() if r["proc"] else (None if alive(r["pid"]) else "exited")
            if code is None:
                continue
            if r["log"]:
                r["log"].close()
            running.remove(r)
            ok, info = validate(r, code if r["proc"] else None)
            status["units"][key(r)].update(phase="complete" if ok else "failed", finished=now(), **info)
            save()
        gate_open = GATE.exists()
        per_gpu = args.per_gpu
        override = OUT / "PER_GPU"
        if override.exists():
            try:
                per_gpu = int(override.read_text().strip())
            except ValueError:
                pass
        if status.get("per_gpu") != per_gpu:
            status["per_gpu"] = per_gpu
            save()
        for gpu in gpus:
            while sum(r["gpu"] == gpu for r in running) < per_gpu:
                if gpu in gated and not gate_open:
                    break
                pick = next((u for u in queue if u["job"] not in NEEDS_GATE or gate_open), None)
                if pick is None:
                    break
                queue.remove(pick)
                Path(pick["out"]).parent.mkdir(parents=True, exist_ok=True)
                log = (Path(pick["out"]).parent / f"{pick['job']}.log").open("w")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                env.setdefault("HF_HOME", str(Path.home() / "hf"))
                proc = subprocess.Popen(command(pick["seed"], pick["arms"], pick["adapter"], Path(pick["out"])),
                                        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
                running.append({**pick, "gpu": gpu, "pid": proc.pid, "proc": proc, "log": log})
                status["units"][key(pick)].update(phase="running", gpu=gpu, pid=proc.pid, started=now())
                save()
        time.sleep(60)

    status["phase"] = "complete" if all(u["phase"] == "complete" for u in status["units"].values()) else "failed"
    status["finished"] = now()
    save()
    return 0 if status["phase"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

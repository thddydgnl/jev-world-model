"""V1 calibration (kiis2026f/실험계획.md §4 V1) on the GPU server, started by hand.

Configurations are tried in stages by lever count (P1 counts as one). Each runs
the two world-model-free planners, validity and oracle, on dev (12 worlds x 1
root) and val (20 worlds x 2 roots) with policy seed 20260921. After a stage
every configuration is scored on G1-G3; the first stage with a configuration
meeting all three ends the search and the committed rule picks one (fewest
levers, then oracle closest to 80%). C then runs on the chosen configuration
for G4. If no stage passes, the configuration with the largest oracle -
validity gap is taken and the shortfall recorded; the criteria do not move.

Alongside stage 1 it runs the v1 structure's oracle on val (20 x 4), to see
whether val shows the candidate-coverage failures that capped K4 on test.

No world-model arm runs here and no test world is built. Everything a decision
used is written to artifacts/kiis_v1cal/status.json as it happens.
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
OUT = ROOT / "artifacts/kiis_v1cal"
SEED = 20260921
SPLITS = (("dev", 12, 1), ("val", 20, 2))
GPUS, PER_GPU = (0, 1), 2
G1, G2, G3, G4, TARGET = (0.75, 0.90), 0.15, 0.05, 0.20, 0.80


def config(levers: tuple[str, ...] = (), samples: int = 1, hint: bool | int = True) -> dict:
    """hint True is P1, 2 is P1' (named P1k); samples 2 is P2, 3 is P2x3."""
    extra = ["P2"] if samples == 2 else ([f"P2x{samples}"] if samples > 2 else [])
    parts = ["P1k" if hint == 2 else "P1", *levers] + extra
    return {"name": "+".join(parts), "levers": list(levers), "hint": hint,
            "samples": samples, "n_levers": len(parts)}


STAGES = [
    [config()],
    [config(("T1",)), config(samples=2)],
    [config(("T1", "T2")), config(("T1",), 2)],
    [config(("T1", "T2", "T3")), config(("T1", "T2"), 2)],
    [config(("T1", "T2", "T3"), 2)],
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Status:
    def __init__(self, path: Path, plan: str = "kiis2026f/실험계획.md §4 V1",
                 stages: list | None = None) -> None:
        self.path = path
        self.data = {"plan": plan, "started": now(), "phase": "start",
                     "criteria": {"G1": list(G1), "G2": G2, "G3": G3, "G4": G4, "target": TARGET},
                     "stages": stages if stages is not None else [[c["name"] for c in s] for s in STAGES],
                     "configs": {}, "v1a": None, "chosen": None, "log": []}
        self.save()

    def note(self, msg: str, **kw) -> None:
        self.data["log"].append({"t": now(), "msg": msg, **kw})
        print(f"[{now()}] {msg}", flush=True)
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False) + "\n")
        tmp.replace(self.path)


def runner_cmd(out: Path, split: str, worlds: int, roots: int, arms: str,
               cfg: dict | None) -> list[str]:
    cmd = [sys.executable, "-u", str(ROOT / "scripts/mvp_c_headroom.py"), "--device", "cuda:0",
           "--trap", "--split", split, "--worlds", str(worlds), "--roots", str(roots),
           "--cap", "30", "--horizon", "2", "--policy-seed", str(SEED),
           "--arms", arms, "--out", str(out)]
    if cfg is not None:
        cmd += ["--v2", "--levers", ",".join(cfg["levers"])]
        if cfg["hint"] == 2 and not isinstance(cfg["hint"], bool):
            cmd.append("--policy-hint-carry-key")
        elif cfg["hint"]:
            cmd.append("--policy-hint")
        if cfg["samples"] > 1:
            cmd += ["--policy-samples", str(cfg["samples"])]
    return cmd


def run_jobs(jobs: list[tuple[str, list[str], Path]], status: Status) -> bool:
    """Run (name, cmd, out) jobs, at most PER_GPU per GPU. True if all exit 0
    and wrote a finished manifest."""
    pending, running, ok = list(jobs), [], True
    load = {g: 0 for g in GPUS}
    while pending or running:
        while pending and min(load.values()) < PER_GPU:
            name, cmd, out = pending.pop(0)
            gpu = min(GPUS, key=lambda g: load[g])
            out.mkdir(parents=True, exist_ok=True)
            log = (out.parent / f"{out.name}.log").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            env.setdefault("HF_HOME", str(Path.home() / "hf"))
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            running.append((name, proc, out, log, gpu, time.time()))
            load[gpu] += 1
            status.note(f"started {name} on GPU{gpu}", pid=proc.pid)
        time.sleep(30)
        for item in list(running):
            name, proc, out, log, gpu, t0 = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            load[gpu] -= 1
            man = out / "run_manifest.json"
            done = man.exists() and bool(json.loads(man.read_text())["run"].get("finished"))
            good = proc.returncode == 0 and done
            ok &= good
            status.note(f"{'finished' if good else 'FAILED'} {name}", exit=proc.returncode,
                        minutes=round((time.time() - t0) / 60, 1))
    return ok


def score(dirs: list[Path], out: Path) -> dict:
    """Success rates and candidate-coverage failures from the replay diagnostic."""
    res = subprocess.run([sys.executable, str(ROOT / "scripts/kiis_k4_ceiling.py"),
                          "--runs", *map(str, dirs), "--out", str(out)],
                         cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ceiling diagnostic failed: {res.stdout[-500:]} {res.stderr[-500:]}")
    eps = json.loads(out.read_text())["episodes"]
    by = {}
    for arm in sorted({e["arm"] for e in eps}):
        mine = [e for e in eps if e["arm"] == arm]
        by[arm] = {"n": len(mine), "success": sum(e["kind"] == "success" for e in mine),
                   "coverage": sum(e["kind"] == "coverage" for e in mine),
                   "planner": sum(e["kind"] == "planner" for e in mine),
                   "trap": sum(e["kind"] == "trap" for e in mine),
                   "by_split": {s: sum(e["kind"] == "success" for e in mine if e["split"] == s)
                                / max(1, sum(e["split"] == s for e in mine))
                                for s in sorted({e["split"] for e in mine})}}
        by[arm]["rate"] = by[arm]["success"] / by[arm]["n"]
    return by


def judge(by: dict) -> dict:
    o, v = by["oracle"], by["validity"]
    gap = o["rate"] - v["rate"]
    cov = o["coverage"] / o["n"]
    g = {"G1": G1[0] <= o["rate"] <= G1[1], "G2": gap >= G2, "G3": cov <= G3}
    return {"oracle": round(o["rate"], 4), "validity": round(v["rate"], 4),
            "gap": round(gap, 4), "coverage_rate": round(cov, 4), **g,
            "pass": all(g.values())}


# ------------------------------------------------------------------ V1b
#
# kiis2026f/실험계획.md §4 V1b (9/26, after V1 stages 1-3): the policy's missing
# candidates, not the task, limited every V1 configuration, and the side room
# (T1) made it worse. V1b fixes the candidates first and adds task levers after.
#   A   P1' + P2; if its missing-candidate rate misses G3, P1' + P2x3. The first
#       to meet G3 is the base; if neither does, the one with fewer such failures.
#   B   on that base: B0 the base itself, B1 +T2 / +T3, B2 +T2+T3, B3 +T1+T2+T3.
#       The first stage with a configuration meeting G1-G3 ends the search; the
#       rule and the fallback are V1's. C then runs on the choice for G4.
# Run order is not decision order: B1 on a base runs alongside that base, and B2
# with B3, so that GPUs do not sit idle; the decision still reads the stages in
# order, and runs it does not use are marked as such.

V1B_OUT = ROOT / "artifacts/kiis_v1bcal"
V1B_BASES = [config((), 2, hint=2), config((), 3, hint=2)]
V1B_TASK = [[("T2",), ("T3",)], [("T2", "T3")], [("T1", "T2", "T3")]]


def on_base(base: dict, levers: tuple[str, ...]) -> dict:
    return config(levers, base["samples"], hint=base["hint"])


def batch(cfgs: list[dict], out: Path, status: Status, step: dict) -> bool:
    """Every configuration on dev and val (the long val runs first), then scored
    into status.data['configs']. False if a run failed."""
    jobs = [(f"{c['name']}/{s}", runner_cmd(out / c["name"] / s, s, w, r, "validity,oracle", c),
             out / c["name"] / s)
            for s, w, r in sorted(SPLITS, key=lambda x: x[0] != "val") for c in cfgs]
    if not run_jobs(jobs, status):
        return False
    for c in cfgs:
        by = score([out / c["name"] / s for s, _, _ in SPLITS], out / c["name"] / "ceiling.json")
        man = json.loads((out / c["name"] / "dev" / "run_manifest.json").read_text())
        res = {**c, **step[c["name"]], "condition_hash": man["config_hash"], "arms": by, **judge(by)}
        status.data["configs"][c["name"]] = res
        status.note(f"scored {c['name']}", **{x: res[x] for x in
                    ("oracle", "validity", "gap", "coverage_rate", "G1", "G2", "G3", "pass")})
    return True


def first_passing(stages: list[list[dict]], configs: dict) -> dict | None:
    for stage in stages:
        passing = [configs[c["name"]] for c in stage if configs[c["name"]]["pass"]]
        if passing:
            return min(passing, key=lambda c: abs(c["oracle"] - TARGET))
    return None


def run_v1b() -> int:
    out = V1B_OUT
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    status = Status(out / "status.json", plan="kiis2026f/실험계획.md §4 V1b",
                    stages={"A": [b["name"] for b in V1B_BASES],
                            "B": ["base", "base+T2 | base+T3", "base+T2+T3", "base+T1+T2+T3"]})
    configs = status.data["configs"]
    base = None
    for i, b in enumerate(V1B_BASES, start=1):
        b1 = [on_base(b, lv) for lv in V1B_TASK[0]]
        status.data["phase"] = f"A{i}: {b['name']} with its B1 alongside"
        step = {b["name"]: {"step": "A", "base": b["name"]},
                **{c["name"]: {"step": "B1", "base": b["name"]} for c in b1}}
        if not batch([b, *b1], out, status, step):
            status.data["phase"] = "failed"
            status.note(f"A{i}: a run failed; stopping for review")
            return 1
        if configs[b["name"]]["G3"]:
            base = b
            break
    met_g3 = base is not None
    if base is None:
        base = min(V1B_BASES, key=lambda b: (configs[b["name"]]["coverage_rate"], b["samples"]))
    status.data["base"] = {"name": base["name"], "met_G3": met_g3}
    status.note(f"base {base['name']}", met_G3=met_g3)

    stages = [[base], [on_base(base, lv) for lv in V1B_TASK[0]]]
    chosen = first_passing(stages, configs)
    if chosen is None:
        later = [[on_base(base, lv) for lv in st] for st in V1B_TASK[1:]]
        status.data["phase"] = "B2 and B3"
        step = {c["name"]: {"step": f"B{k}", "base": base["name"]}
                for k, st in enumerate(later, start=2) for c in st}
        if not batch([c for st in later for c in st], out, status, step):
            status.data["phase"] = "failed"
            status.note("B2/B3: a run failed; stopping for review")
            return 1
        stages += later
        chosen = first_passing(stages, configs)
    for c in configs.values():     # B1 runs on a base that was not taken decide nothing
        c["used_for_decision"] = c["step"] == "A" or c["base"] == base["name"]
    if chosen is not None:
        status.data["chosen"] = {"name": chosen["name"], "met_all_G1_G3": True,
                                 "rule": "first passing stage on the base; oracle closest to 80%"}
    else:
        pool = [configs[c["name"]] for st in stages for c in st]
        chosen = max(pool, key=lambda c: (c["gap"], -c["n_levers"], -abs(c["oracle"] - TARGET)))
        status.data["chosen"] = {"name": chosen["name"], "met_all_G1_G3": False,
                                 "rule": "no configuration met G1-G3; largest oracle - validity gap"}
    status.note(f"chosen {chosen['name']}", **status.data["chosen"])

    status.data["phase"] = "G4 (C on the chosen configuration)"
    cdir = out / chosen["name"] / "C"
    jobs = [(f"{chosen['name']}/C/{s}", runner_cmd(cdir / s, s, w, r, "C", chosen), cdir / s)
            for s, w, r in SPLITS]
    if not run_jobs(jobs, status):
        status.data["phase"] = "failed"
        status.note("C run failed; stopping for review")
        return 1
    c_by = score([cdir / s for s, _, _ in SPLITS], cdir / "ceiling.json")["C"]
    status.data["chosen"].update({"C": round(c_by["rate"], 4), "G4": c_by["rate"] <= G4})
    status.data["phase"] = "done"
    status.data["finished"] = now()
    status.note("done", **status.data["chosen"])
    (out / "DONE").write_text(status.data["chosen"]["name"] + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=("v1", "v1b"), default="v1")
    ap.add_argument("--stages", type=int, default=len(STAGES), help="v1 only")
    args = ap.parse_args()
    if args.plan == "v1b":
        return run_v1b()

    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    status = Status(OUT / "status.json")
    configs = status.data["configs"]
    chosen = None

    for k, stage in enumerate(STAGES[:args.stages], start=1):
        status.data["phase"] = f"stage {k}"
        jobs = [(f"{c['name']}/{s}", runner_cmd(OUT / c["name"] / s, s, w, r,
                                                "validity,oracle", c), OUT / c["name"] / s)
                for c in stage for s, w, r in SPLITS]
        if k == 1:
            jobs.append(("v1a_val_oracle", runner_cmd(OUT / "v1a_val_oracle", "val", 20, 4,
                                                      "oracle", None), OUT / "v1a_val_oracle"))
        if not run_jobs(jobs, status):
            status.data["phase"] = "failed"
            status.note(f"stage {k}: a run failed; stopping for review")
            return 1
        if k == 1:
            v1a = score([OUT / "v1a_val_oracle"], OUT / "v1a_val_oracle.ceiling.json")["oracle"]
            status.data["v1a"] = {"oracle": round(v1a["rate"], 4), "n": v1a["n"],
                                  "coverage": v1a["coverage"], "planner": v1a["planner"]}
            status.note("v1a scored", **status.data["v1a"])
        for c in stage:
            by = score([OUT / c["name"] / s for s, _, _ in SPLITS], OUT / c["name"] / "ceiling.json")
            man = json.loads((OUT / c["name"] / "dev" / "run_manifest.json").read_text())
            configs[c["name"]] = {**c, "stage": k, "condition_hash": man["config_hash"],
                                  "arms": by, **judge(by)}
            status.note(f"scored {c['name']}", **{x: configs[c["name"]][x] for x in
                        ("oracle", "validity", "gap", "coverage_rate", "G1", "G2", "G3", "pass")})
        passing = [configs[c["name"]] for c in stage if configs[c["name"]]["pass"]]
        if passing:
            chosen = min(passing, key=lambda c: abs(c["oracle"] - TARGET))
            status.data["chosen"] = {"name": chosen["name"], "met_all_G1_G3": True,
                                     "rule": "first passing stage; oracle closest to 80%"}
            break
    if chosen is None:
        pool = list(configs.values())
        chosen = max(pool, key=lambda c: (c["gap"], -c["n_levers"], -abs(c["oracle"] - TARGET)))
        status.data["chosen"] = {"name": chosen["name"], "met_all_G1_G3": False,
                                 "rule": "no configuration met G1-G3; largest oracle - validity gap"}
    status.note(f"chosen {chosen['name']}", **status.data["chosen"])

    status.data["phase"] = "G4 (C on the chosen configuration)"
    cdir = OUT / chosen["name"] / "C"
    jobs = [(f"{chosen['name']}/C/{s}", runner_cmd(cdir / s, s, w, r, "C", chosen), cdir / s)
            for s, w, r in SPLITS]
    if not run_jobs(jobs, status):
        status.data["phase"] = "failed"
        status.note("C run failed; stopping for review")
        return 1
    c_by = score([cdir / s for s, _, _ in SPLITS], cdir / "ceiling.json")["C"]
    status.data["chosen"].update({"C": round(c_by["rate"], 4), "G4": c_by["rate"] <= G4})
    status.data["phase"] = "done"
    status.data["finished"] = now()
    status.note("done", **status.data["chosen"])
    (OUT / "DONE").write_text(status.data["chosen"]["name"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V1, V1b and W (v3) calibration (kiis2026f/실험계획.md §4 V1, V1b, W) on the GPU
server, started by hand. The docstring below is V1's; V1b and W are described
above their own functions.


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


def config(levers: tuple[str, ...] = (), samples: int = 1, hint: bool | int = True,
           retry: bool = False) -> dict:
    """hint True is P1, 2 is P1' (named P1k); samples 2 is P2, 3 is P2x3;
    retry is P3 (v3). The P3 key is added only when on, so v1/V1b records
    keep their shape."""
    extra = ["P2"] if samples == 2 else ([f"P2x{samples}"] if samples > 2 else [])
    parts = ["P1k" if hint == 2 else "P1", *levers] + extra + (["P3"] if retry else [])
    cfg = {"name": "+".join(parts), "levers": list(levers), "hint": hint,
           "samples": samples, "n_levers": len(parts)}
    if retry:
        cfg["retry"] = True
    return cfg


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
        if cfg.get("retry"):
            cmd.append("--policy-retry-stuck")
    return cmd


def run_jobs(jobs: list[tuple[str, list[str], Path]], status: Status,
             slots: dict[int, int] | None = None) -> bool:
    """Run (name, cmd, out) jobs, at most slots[gpu] per GPU (default PER_GPU on
    each of GPUS). True if all exit 0 and wrote a finished manifest."""
    slots = slots or {g: PER_GPU for g in GPUS}
    pending, running, ok = list(jobs), [], True
    load = {g: 0 for g in slots}
    while pending or running:
        while pending and any(load[g] < slots[g] for g in slots):
            name, cmd, out = pending.pop(0)
            gpu = min((g for g in slots if load[g] < slots[g]), key=lambda g: load[g] / slots[g])
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


# ------------------------------------------------------------------ W (v3)
#
# kiis2026f/실험계획.md §4 W (9/27, committed before this runs). No task lever
# (the v1 structure), and a calibration set with many names: dev (12 x 1), val
# (20 x 2) and cal_test = the v2 test names (24 x 2), which K4v2 had already
# shown. Two settings, both run: W-a = P1' + P2, W-b = W-a + P3. Criteria:
#   G1 oracle >= 90% overall AND on cal_test (no upper bound)
#   G2 oracle - validity >= 15%p overall
#   G3 oracle episodes ended by missing candidates <= 5% overall AND on cal_test
#   G4 C <= 20% on the chosen setting
#   G5 monotonicity: on cal_test x 1 root (24 episodes; oracle re-run on the
#      same episodes) B0, D0 and A_jev each succeed at most once more than oracle
# Rule: the first setting in order (W-a, W-b) meeting G1-G3 gets G4 and G5; if
# it fails either and the next meets G1-G3, the next gets them. If none passes,
# the setting closest on G1 (the lower of the two oracle rates), then G2, is
# taken and the shortfall recorded. The criteria do not move.

W_OUT = ROOT / "artifacts/kiis_wcal"
W_SETS = (("test", 24, 2), ("val", 20, 2), ("dev", 12, 1))   # longest first; test = cal_test
W_G1, W_G2, W_G3, W_G4, W_G5_SLACK = 0.90, 0.15, 0.05, 0.20, 1
W_CONFIGS = [config((), 2, hint=2), config((), 2, hint=2, retry=True)]


def w_score(dirs: list[Path], out: Path) -> dict:
    """Per arm, overall and per split: n, success, coverage failures."""
    res = subprocess.run([sys.executable, str(ROOT / "scripts/kiis_k4_ceiling.py"),
                          "--runs", *map(str, dirs), "--out", str(out)],
                         cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ceiling diagnostic failed: {res.stdout[-500:]} {res.stderr[-500:]}")
    eps = json.loads(out.read_text())["episodes"]
    by: dict = {}
    for arm in sorted({e["arm"] for e in eps}):
        for part in ("all", *sorted({e["split"] for e in eps})):
            mine = [e for e in eps if e["arm"] == arm and part in ("all", e["split"])]
            n = len(mine)
            by.setdefault(arm, {})[part] = {
                "n": n, "success": sum(e["kind"] == "success" for e in mine),
                "coverage": sum(e["kind"] == "coverage" for e in mine),
                "planner": sum(e["kind"] == "planner" for e in mine),
                "rate": round(sum(e["kind"] == "success" for e in mine) / max(1, n), 4),
                "coverage_rate": round(sum(e["kind"] == "coverage" for e in mine) / max(1, n), 4)}
    return by


def w_judge(by: dict) -> dict:
    o, v = by["oracle"], by["validity"]
    gap = o["all"]["rate"] - v["all"]["rate"]
    g = {"G1": o["all"]["rate"] >= W_G1 and o["test"]["rate"] >= W_G1,
         "G2": gap >= W_G2,
         "G3": o["all"]["coverage_rate"] <= W_G3 and o["test"]["coverage_rate"] <= W_G3}
    return {"oracle": o["all"]["rate"], "oracle_cal_test": o["test"]["rate"],
            "validity": v["all"]["rate"], "gap": round(gap, 4),
            "coverage_rate": o["all"]["coverage_rate"],
            "coverage_rate_cal_test": o["test"]["coverage_rate"], **g,
            "pass_G1_G3": all(g.values())}


def w_g4_g5(cfg: dict, out: Path, status: Status, slots: dict[int, int]) -> bool:
    """C on the three sets (G4) and the G5 set (oracle, B0+D0, A_jev on cal_test
    x 1 root), side by side. Writes the verdicts into the config's record."""
    d = out / cfg["name"]
    jobs = [(f"{cfg['name']}/C/{s}", runner_cmd(d / "C" / s, s, w, r, "C", cfg), d / "C" / s)
            for s, w, r in W_SETS]
    g5 = {"oracle": "oracle", "zero_shot": "B0_typed,D0_gen", "A_jev": "A_jev"}
    for name, arms in g5.items():
        cmd = runner_cmd(d / "G5" / name, "test", 24, 1, arms, cfg)
        if name == "A_jev":
            cmd += ["--budget", "1"]
        jobs.append((f"{cfg['name']}/G5/{name}", cmd, d / "G5" / name))
    jobs.sort(key=lambda j: "/G5/zero_shot" not in j[0])        # slowest first
    if not run_jobs(jobs, status, slots):
        return False
    rec = status.data["configs"][cfg["name"]]
    c_by = w_score([d / "C" / s for s, _, _ in W_SETS], d / "C" / "ceiling.json")["C"]["all"]
    rec.update(C=c_by["rate"], G4=c_by["rate"] <= W_G4)
    g5_by = w_score([d / "G5" / n for n in g5], d / "G5" / "ceiling.json")
    o = g5_by["oracle"]["all"]["success"]
    counts = {a: g5_by[a]["all"]["success"] for a in ("oracle", "B0_typed", "D0_gen", "A_jev")}
    rec.update(G5_counts=counts, G5_n=g5_by["oracle"]["all"]["n"],
               G5=all(counts[a] <= o + W_G5_SLACK for a in ("B0_typed", "D0_gen", "A_jev")))
    status.note(f"G4/G5 {cfg['name']}", C=rec["C"], G4=rec["G4"], G5=rec["G5"], **counts)
    return True


def run_w(slots: dict[int, int]) -> int:
    out = W_OUT
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    status = Status(out / "status.json", plan="kiis2026f/실험계획.md §4 W",
                    stages={"order": [c["name"] for c in W_CONFIGS]})
    status.data["criteria"] = {"G1": W_G1, "G2": W_G2, "G3": W_G3, "G4": W_G4,
                               "G5_slack": W_G5_SLACK, "sets": [list(x) for x in W_SETS],
                               "G1_G3_also_on": "cal_test (split test, no lever)"}
    status.data["slots"] = {str(g): n for g, n in slots.items()}
    configs = status.data["configs"]

    status.data["phase"] = "G1-G3: validity and oracle, both settings"
    jobs = [(f"{c['name']}/{s}", runner_cmd(out / c["name"] / s, s, w, r, "validity,oracle", c),
             out / c["name"] / s) for s, w, r in W_SETS for c in W_CONFIGS]
    if not run_jobs(jobs, status, slots):
        status.data["phase"] = "failed"
        status.note("a G1-G3 run failed; stopping for review")
        return 1
    for c in W_CONFIGS:
        by = w_score([out / c["name"] / s for s, _, _ in W_SETS], out / c["name"] / "ceiling.json")
        man = json.loads((out / c["name"] / "dev" / "run_manifest.json").read_text())
        configs[c["name"]] = {**c, "condition_hash": man["config_hash"], "arms": by, **w_judge(by)}
        status.note(f"scored {c['name']}", **{x: configs[c["name"]][x] for x in
                    ("oracle", "oracle_cal_test", "validity", "gap", "coverage_rate",
                     "coverage_rate_cal_test", "G1", "G2", "G3", "pass_G1_G3")})

    chosen = None
    for c in W_CONFIGS:
        if not configs[c["name"]]["pass_G1_G3"]:
            continue
        status.data["phase"] = f"G4/G5 on {c['name']}"
        if not w_g4_g5(c, out, status, slots):
            status.data["phase"] = "failed"
            status.note("a G4/G5 run failed; stopping for review")
            return 1
        if configs[c["name"]]["G4"] and configs[c["name"]]["G5"]:
            chosen = c
            status.data["chosen"] = {"name": c["name"], "met_all": True,
                                     "rule": "first setting in order meeting G1-G5"}
            break
    if chosen is None:
        chosen = max(W_CONFIGS, key=lambda c: (min(configs[c["name"]]["oracle"],
                                                   configs[c["name"]]["oracle_cal_test"]),
                                               configs[c["name"]]["gap"]))
        if "G5" not in configs[chosen["name"]]:
            status.data["phase"] = f"G4/G5 on {chosen['name']} (record only)"
            if not w_g4_g5(chosen, out, status, slots):
                status.data["phase"] = "failed"
                return 1
        status.data["chosen"] = {"name": chosen["name"], "met_all": False,
                                 "rule": "no setting met G1-G5; closest on G1, then G2"}
    status.data["chosen"].update({k: configs[chosen["name"]].get(k) for k in
                                  ("condition_hash", "oracle", "oracle_cal_test", "validity", "gap",
                                   "coverage_rate", "coverage_rate_cal_test", "C",
                                   "G1", "G2", "G3", "G4", "G5", "G5_counts")})
    status.data["phase"] = "done"
    status.data["finished"] = now()
    status.note(f"chosen {chosen['name']}", **status.data["chosen"])
    (out / "DONE").write_text(chosen["name"] + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=("v1", "v1b", "w"), default="v1")
    ap.add_argument("--stages", type=int, default=len(STAGES), help="v1 only")
    ap.add_argument("--slots", default="0,0,1,1",
                    help="w only: one GPU id per process slot, e.g. 0,0,1")
    args = ap.parse_args()
    if args.plan == "v1b":
        return run_v1b()
    if args.plan == "w":
        slots: dict[int, int] = {}
        for g in args.slots.split(","):
            slots[int(g)] = slots.get(int(g), 0) + 1
        return run_w(slots)

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

"""K6: every number the KIIS paper uses, computed from the logs
(kiis2026f/실험계획.md K6, kiis2026f/논문구성안.md §7).

    python scripts/kiis_report.py            # writes kiis2026f/results/
    python scripts/kiis_report.py --check    # recompute; fail if slots.json would change

Writes slots.json (R1-R20, verdicts, the §5 sentences they select, the
rounded numbers of the online abstract), table1.md/.csv and fig2.svg/.csv.
Intervals are world-clustered bootstraps over the 24 test worlds (4000 draws,
a world drawn twice counts twice, analyze_arms.boot_paired); every quantity
gets its own RNG seeded 20260921, so no number depends on what else was
computed. Reads the local logs, which are not in git: artifacts/kiis_k4v3
(episodes and steps), artifacts/kiis_k5v3 and kiis_x1v3 (*.jsonl),
artifacts/kiis_k4 and kiis_k4v2 (R17), data/kiis/transitions{,_v3} (R20).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_arms as aa
import kiis_k4_hypotheses as hyp

K4 = ROOT / "artifacts/kiis_k4v3"
K5 = {"test": ROOT / "artifacts/kiis_k5v3", "x1": ROOT / "artifacts/kiis_x1v3"}
OUT = ROOT / "kiis2026f/results"
SEED = 20260921
MODELS = ("A_jev", "B0_typed", "B_typed", "D0_gen", "D_gen")
DEPTH = 4

# k-step contrasts over (rollout, k), k = 1..4, same form as hyp.TESTS
K5_TESTS = [
    ("H2  B0 - D0", {"B0_typed": 1, "D0_gen": -1}, 1),
    ("H3  (B - D) - (B0 - D0)", {"B_typed": 1, "D_gen": -1, "B0_typed": -1, "D0_gen": 1}, -1),
    ("H4  B - A", {"B_typed": 1, "A_jev": -1}, 1),
    ("H5  A - B0", {"A_jev": 1, "B0_typed": -1}, 1),
    ("    B - D", {"B_typed": 1, "D_gen": -1}, 0),
    ("    D - A", {"D_gen": 1, "A_jev": -1}, 0),
]

# kiis2026f/실험계획.md §5, the sentence each outcome selects
SENTENCES = {
    ("H1", "holds"): "세계모델은 기본 에이전트의 성공률을 R1에서 R2–R6 수준으로 높였다",
    ("H1", "partial"): "어떤 세계모델이 도움이 됐고 어떤 것이 안 됐는지 그대로 쓰고, 무효율·파싱 실패율로 원인을 짚는다",
    ("H1b", "holds"): "실패 기억만 있는 기본 에이전트(C_fm)와 비교해도 세계모델의 이득이 남았다",
    ("H1b", "partial"): "세계모델의 이득 중 상당 부분은 실패한 행동을 반복하지 않는 것으로 설명된다 — 그 차이를 그대로 쓴다",
    ("H2", "holds"): "같은 LLM에서 질문을 선택지로 바꾸기만 해도 성공률이 R4에서 R3으로 올랐다",
    ("H2", "other"): "학습 없는 LLM에서는 생성형이 타입화보다 낮지 않았다 — JEV 방식의 이득은 JEV 모델에 있다 (H5와 함께)",
    ("H3", "holds"): "학습하면 두 방식의 차이가 줄었다 — 타입화의 이점은 학습 데이터가 없을 때 크다",
    ("H3", "other"): "학습 후에도 타입화가 앞섰다 (또는 생성형이 앞섰다) — 방식 차이가 데이터로 사라지지 않는다",
    ("H4", "holds"): "전이 N개로 학습한 LLM은 JEV 방식을 frozen JEV 이상으로 재현했다",
    ("H4", "opposite"): "frozen JEV가 전이 N개로 학습한 LLM보다 높았다",
    ("H5", "holds"): "같은 질문 방식에서 JEV는 학습 없는 LLM보다 높았다",
    ("H7", "holds"): "한 구조로 학습한 LLM은 새 구조에서 frozen JEV보다 크게 떨어졌다 — 학습의 이득은 학습한 구조 안에서다",
    ("H7", "other"): "학습한 LLM의 하락이 JEV보다 크지 않았다 (또는 구분되지 않았다)",
    ("H8", "holds"): "같은 LLM에서 선택지로 물으면 여러 step을 굴릴 때 자기 오류가 덜 쌓였다 — 형식의 효과",
    ("H8", "other"): "두 방식의 오류 누적은 구분되지 않았다 (또는 생성형이 덜 쌓였다) — 그대로 쓴다",
    ("*", "not distinguishable"): "이 표본으로는 구분되지 않는다 (\"동등하다\"고 쓰지 않는다)",
}


def rng() -> random.Random:
    return random.Random(SEED)


def interval(per_world: dict) -> dict:
    point, lo, hi = aa.boot_paired(per_world, rng())
    return {"value": point, "ci": [lo, hi]}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ closed loop

def closed_loop() -> dict:
    eps = aa.load({str(K4 / str(s) / u): s for s in hyp.SEEDS for u in hyp.UNITS})
    hashes = {e.get("config_hash") for e in eps}
    arm_hash = defaultdict(set)
    for e in eps:
        arm_hash[e["arm"]].add(e.get("arm_hash"))
    if len(hashes) != 1 or any(len(h) != 1 for h in arm_hash.values()):
        raise SystemExit(f"refusing to pool: {hashes} {dict(arm_hash)}")
    worlds = aa.by_world(eps)
    arms = {}
    for arm in aa.ORDER:
        v = [e for e in eps if e["arm"] == arm]
        if not v:
            continue
        point, lo, hi = aa.boot_ci(worlds, lambda s, a=arm: aa.rate(s, a), rng())
        succ = [e for e in v if e["success"]]
        steps = sum(e["steps"] for e in v)
        req = [e.get("wm_requests", e.get("jev_requests")) for e in v]
        sec = [e.get("wm_seconds") for e in v]
        arms[arm] = {
            "n": len(v), "success": point, "ci": [lo, hi], "arm_hash": next(iter(arm_hash[arm])),
            "per_seed": {str(s): sum(e["success"] for e in v if e["seed"] == s) for s in hyp.SEEDS},
            "steps_mean": steps / len(v), "steps_success": sum(e["steps"] for e in succ) / len(succ),
            "invalid_rate": sum(e["invalid"] for e in v) / steps,
            "wm_requests_per_ep": None if None in req else sum(req) / len(v),
            "wm_gpu_s_per_ep": None if None in sec else sum(sec) / len(v)}
    tests = {}
    for name, coef, direction in hyp.TESTS:
        if not all(a in arms for a in coef):
            continue
        pw, pos, neg = hyp.contrast_by_world(eps, coef)
        point, lo, hi = aa.boot_paired(pw, rng())
        tests[name.strip()] = {"coef": coef, "n": sum(len(x) for x in pw.values()), "diff": point,
                               "ci": [lo, hi], "positive": pos, "negative": neg,
                               "verdict": hyp.verdict(lo, hi, direction)}
    parse = {}
    for arm in ("D0_gen", "D_gen"):
        v = [e for e in eps if e["arm"] == arm]
        f = sum(e.get("parse_failures", 0) for e in v)
        r = sum(e.get("wm_requests", 0) for e in v)
        # Predictions are not logged in closed loop, only requests = predictions
        # + retries, with failures <= retries <= predictions.
        parse[arm] = {"failed_predictions": f, "requests": r,
                      "rate_bounds": [f / (r - f), 2 * f / r] if f else [0.0, 0.0]}
    jev = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
    for s in hyp.SEEDS:
        calls, tokens, cost = re.findall(r"calls=(\d+) tokens=([\d,]+) cost=\$([\d.]+)",
                                         (K4 / str(s) / "core_b.log").read_text())[-1]
        jev["calls"] += int(calls)
        jev["tokens"] += int(tokens.replace(",", ""))
        jev["cost_usd"] += float(cost)
    jev["per_episode_usd"] = jev["cost_usd"] / arms["A_jev"]["n"]
    ceiling = {row["arm"]: {k: row.get(k, 0) for k in ("n", "success", "coverage", "planner", "trap")}
               for row in json.loads((K4 / "ceiling.json").read_text())["summary"]}
    import kiis_k4_replay as rp
    replay = rp.summarize(rp.replay_all(str(K4), [str(s) for s in hyp.SEEDS]))
    return {"condition": hashes.pop(), "episodes": len(eps), "arms": arms, "tests": tests,
            "parse": parse, "jev": jev, "failure_kinds": ceiling, "d0_replay": replay}


# ------------------------------------------------------------------ k-step rollouts

def k5_rows(split: str, name: str) -> dict:
    return {(r["id"], r["k"]): r for r in jsonl(K5[split] / f"{name}.jsonl")}


def rollouts() -> dict:
    out = {}
    per_world_mean = {}
    for split in ("test", "x1"):
        fr = {m: k5_rows(split, m) for m in MODELS}
        tf = {m: k5_rows(split, m + ".tf") for m in MODELS}
        summary = json.loads((K5[split] / "summary.json").read_text())
        curves = {}
        for m in MODELS:
            curves[m] = {}
            for k in range(1, DEPTH + 1):
                pw = defaultdict(list)
                for (_, kk), r in fr[m].items():
                    if kk == k:
                        pw[r["world"]].append(int(r["exact"]))
                curves[m][str(k)] = interval(pw)
        curves["persistence"] = {str(k): {"value": summary["models"]["persistence"]["all"][str(k)]["exact"]}
                                 for k in range(1, DEPTH + 1)}
        tests = {}
        for name, coef, direction in K5_TESTS:
            keys = set.intersection(*(set(fr[m]) for m in coef))
            pw = defaultdict(list)
            for key in keys:
                pw[fr[next(iter(coef))][key]["world"]].append(
                    sum(c * int(fr[m][key]["exact"]) for m, c in coef.items()))
            iv = interval(pw)
            tests[name.strip()] = {"coef": coef, "n": len(keys), "diff": iv["value"], "ci": iv["ci"],
                                   "verdict": hyp.verdict(*iv["ci"], direction)}
        parse = {m: {"failed": sum(bool(r.get("parse_failed")) for r in tf[m].values()), "rows": len(tf[m])}
                 for m in ("D0_gen", "D_gen")}
        parsed = [k for k, r in tf["D0_gen"].items() if not r.get("parse_failed") and k in tf["B0_typed"]]
        content = {"rows_all": len(tf["D0_gen"]), "rows_d0_parsed": len(parsed),
                   "all": {m: sum(int(tf[m][k]["exact"]) for k in tf[m]) / len(tf[m]) for m in ("B0_typed", "D0_gen")},
                   "d0_parsed": {m: sum(int(tf[m][k]["exact"]) for k in parsed) / len(parsed)
                                 for m in ("B0_typed", "D0_gen")}}
        acc = json.loads((K5[split] / "accumulation_h8.json").read_text())
        out[split] = {"rollouts": len({i for (i, _) in fr["A_jev"]}), "curves": curves, "tests": tests,
                      "drop_k1_k4": {m: curves[m]["1"]["value"] - curves[m][str(DEPTH)]["value"]
                                     for m in (*MODELS, "persistence")},
                      "parse_teacher_forced": parse, "content_teacher_forced": content,
                      "accumulation": {"G": {m: acc["models"][m]["G"] for m in acc["models"]},
                                       "pairs": acc["pairs"]}}
        if split == "test":
            out[split]["jev_repeats"] = summary.get("jev_repeats")
        per_world_mean[split] = {}
        for m in MODELS:
            acc_w = defaultdict(list)
            for r in fr[m].values():
                acc_w[r["world"]].append(int(r["exact"]))
            per_world_mean[split][m] = {w: sum(v) / len(v) for w, v in acc_w.items()}
    # H7: per-world drop (test - x1, k = 1..4 exact), paired against JEV
    drops = {m: {w: per_world_mean["test"][m][w] - per_world_mean["x1"][m][w]
                 for w in per_world_mean["test"][m] if w in per_world_mean["x1"][m]} for m in MODELS}
    h7 = {}
    for m in ("B_typed", "D_gen", "B0_typed", "D0_gen"):
        pw = {w: [drops[m][w] - drops["A_jev"][w]] for w in drops[m] if w in drops["A_jev"]}
        iv = interval(pw)
        h7[f"{m} - A_jev"] = {"diff": iv["value"], "ci": iv["ci"], "worlds": len(pw),
                              "verdict": hyp.verdict(*iv["ci"], 1)}
    out["h7"] = {"mean_drop": {m: sum(d.values()) / len(d) for m, d in drops.items()}, "vs_A_jev": h7}
    return out


# ------------------------------------------------------------------ other slots

def structure_overlap() -> dict:
    """R20: test transitions whose (state, action) with names replaced by ids
    is also in the training data — the same structure under other names."""
    from kiis_train_wm import load

    def key(t: dict) -> tuple:
        names = sorted(((e["name"], e["id"]) for e in t["canon"]["entities"] if e["id"] not in ("I", "P")),
                       key=lambda x: -len(x[0]))
        action = t["action"]
        for name, i in names:
            action = action.replace(name, i)
        return (json.dumps(t["canon"]["dynamic_facts"]), json.dumps(t["canon"]["static_facts"]), action)

    train_p = ROOT / "data/kiis/transitions/train.jsonl"
    test_p = ROOT / "data/kiis/transitions_v3/test.jsonl"
    meta = json.loads((ROOT / "artifacts/kiis_k3/B_typed/train_meta.json").read_text())
    train_all = jsonl(train_p)
    used = load(str(train_p), meta["n_transitions"], 0)
    test = jsonl(test_p)
    all_keys, used_keys = {key(t) for t in train_all}, {key(t) for t in used}
    return {"test": len(test), "train": len(train_all), "train_used": len(used),
            "in_train": sum(key(t) in all_keys for t in test),
            "in_train_used": sum(key(t) in used_keys for t in test),
            "train_sha": sha(train_p), "test_sha": sha(test_p)}


def earlier_designs() -> dict:
    """R17: the v3 calibration and what v1 and the v2 pilot showed."""
    cal = json.loads((ROOT / "artifacts/kiis_wcal/status.json").read_text())["chosen"]
    out = {"v3_calibration": {k: cal[k] for k in ("name", "condition_hash", "oracle", "oracle_cal_test",
                                                   "validity", "gap", "coverage_rate", "coverage_rate_cal_test",
                                                   "C", "G5_counts")}}
    for name, pattern in (("v1", "artifacts/kiis_k4/*/*/episodes.jsonl"),
                          ("v2_pilot", "artifacts/kiis_k4v2/20260921/core/episodes.jsonl")):
        n, s, h = defaultdict(int), defaultdict(int), set()
        for p in sorted(ROOT.glob(pattern)):
            for e in jsonl(p):
                n[e["arm"]] += 1
                s[e["arm"]] += e["success"]
                h.add(e.get("config_hash"))
        out[name] = {"condition": sorted(h), "success": {a: [s[a], n[a]] for a in sorted(n)}}
    return out


def small_slots() -> dict:
    crit = json.loads((ROOT / "artifacts/wm_criteria/results.json").read_text())["condition2"]
    train = {a: json.loads((ROOT / f"artifacts/kiis_k3/{a}/train_meta.json").read_text())
             for a in ("B_typed", "D_gen")}
    return {"R13": {"recursive": [crit["exact"], crit["pairs"]],
                    "direct_endpoint": [0, 18],
                    "source": "artifacts/wm_criteria/results.json; direct-endpoint 0/18 is the MVP-B probe "
                              "(설계.md §9) quoted by scripts/wm_criteria.py"},
            "R14": {"n_transitions": train["B_typed"]["n_transitions"],
                    "same_data": train["B_typed"]["data_sha"] == train["D_gen"]["data_sha"],
                    "gpu_hours": {a: t["gpu_hours"] for a, t in train.items()}}}


# ------------------------------------------------------------------ assembly

def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ci_text(ci) -> str:
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def pp(x: float) -> str:
    return f"{100 * x:+.1f}"


def verdicts(cl: dict, ro: dict) -> dict:
    t = cl["tests"]

    def group(names):
        vs = [t[n]["verdict"] for n in names]
        return "holds" if all(v == "holds" for v in vs) else ("partial" if "holds" in vs else "not distinguishable")

    acc = {s: ro[s]["accumulation"]["pairs"] for s in ("test", "x1")}
    h8 = [acc[s][f"{g} (all)"]["ci"][0] > 0 for s in ("test", "x1") for g in ("D0_gen - B0_typed", "D_gen - B_typed")]
    out = {
        "H1": group(["H1  A - C", "H1  B - C", "H1  D - C"]),
        "H1b": group(["H1b A - C_fm", "H1b B - C_fm", "H1b D - C_fm"]),
        "H2": t["H2  B0 - D0"]["verdict"],
        "H2 k-step": ro["test"]["tests"]["H2  B0 - D0"]["verdict"],
        "H3": t["H3  (B - D) - (B0 - D0)"]["verdict"],
        "H3 k-step": ro["test"]["tests"]["H3  (B - D) - (B0 - D0)"]["verdict"],
        "H4": t["H4  B - A"]["verdict"],
        "H4 k-step (test)": ro["test"]["tests"]["H4  B - A"]["verdict"],
        "H4 k-step (x1)": ro["x1"]["tests"]["H4  B - A"]["verdict"],
        "H5": t["H5  A - B0"]["verdict"],
        "H5 k-step": ro["test"]["tests"]["H5  A - B0"]["verdict"],
        "H6": "holds" if max(ro["test"]["drop_k1_k4"], key=ro["test"]["drop_k1_k4"].get) == "D0_gen"
              and all(v > 0 for v in ro["test"]["drop_k1_k4"].values()) else "other",
        "H7": "holds" if all(ro["h7"]["vs_A_jev"][f"{m} - A_jev"]["verdict"] == "holds"
                             for m in ("B_typed", "D_gen")) else "other",
        "H8": "holds" if all(h8) else "other",
    }
    return out


def sentences(v: dict) -> dict:
    """§5 sentences for the closed-loop verdicts and H6-H8. The k-step parts
    of H2-H5 are written in 3.3 by hand from R11, since §5 phrases them as
    success rates."""
    out = {}
    for h, verdict in v.items():
        if "k-step" in h:
            continue
        base = h.split()[0]
        key = (base, verdict) if (base, verdict) in SENTENCES else (base, "other")
        if verdict == "not distinguishable" and (base, verdict) not in SENTENCES:
            key = ("*", "not distinguishable")
        if base == "H6":
            out[h] = "k-step 곡선을 그대로 기술한다 (어느 모델이 얼마나 떨어지는지)"
        elif key in SENTENCES:
            out[h] = SENTENCES[key]
    return out


def slots(cl: dict, ro: dict, r17: dict, r20: dict, small: dict) -> dict:
    a, t = cl["arms"], cl["tests"]

    def arm(name, what):
        return {"what": what, "value": a[name]["success"], "ci": a[name]["ci"], "n": a[name]["n"],
                "text": pct(a[name]["success"]), "ci_text": ci_text(a[name]["ci"])}

    def diff(d):
        return {"value": d["diff"], "ci": d["ci"], "text": pp(d["diff"]), "ci_text": ci_text(d["ci"]),
                "verdict": d["verdict"], **({k: d[k] for k in ("positive", "negative", "n") if k in d})}

    test, x1 = ro["test"], ro["x1"]
    g = cl["d0_replay"]["goal_room_without_food"]
    over = cl["d0_replay"]["overrides"]
    pf_t = test["parse_teacher_forced"]
    pf_x = x1["parse_teacher_forced"]
    return {
        "R1": arm("C", "C success (%)"),
        "R2": arm("A_jev", "A (frozen JEV) success (%)"),
        "R3": arm("B0_typed", "B0 (Qwen typed, zero-shot) success (%)"),
        "R4": arm("D0_gen", "D0 (Qwen generative, zero-shot) success (%)"),
        "R5": arm("B_typed", "B (Qwen typed + LoRA) success (%)"),
        "R6": arm("D_gen", "D (Qwen generative + LoRA) success (%)"),
        "R7": {"validity": arm("validity", "validity success (%)"), "oracle": arm("oracle", "oracle success (%)"),
               "oracle_failures": {k: cl["failure_kinds"]["oracle"][k] for k in ("coverage", "planner", "trap")}},
        "R8": {name: diff(d) for name, d in t.items()},
        "R9": {"what": "invalid actions / all steps (%)",
               "values": {k: pct(v["invalid_rate"]) for k, v in a.items()}},
        "R10": {"jev": {"requests_per_ep": a["A_jev"]["wm_requests_per_ep"], "usd_total": cl["jev"]["cost_usd"],
                        "usd_per_ep": cl["jev"]["per_episode_usd"], "tokens_total": cl["jev"]["tokens"]},
                "gpu_s_per_ep": {m: a[m]["wm_gpu_s_per_ep"] for m in ("B0_typed", "D0_gen", "B_typed", "D_gen")},
                "training_gpu_h": small["R14"]["gpu_hours"]},
        "R11": {"what": "exact state match, free-running, test worlds (%)",
                "curves": {m: {k: {"value": c["value"], **({"ci": c["ci"]} if "ci" in c else {}),
                                   "text": pct(c["value"])} for k, c in cs.items()}
                           for m, cs in test["curves"].items()},
                "k_step_tests": {n: diff(d) for n, d in test["tests"].items()},
                "drop_k1_k4": {m: pp(v) for m, v in test["drop_k1_k4"].items()}},
        "R12": {"what": "generative parse failures after one retry, per one-step prediction "
                        "(teacher-forced rows); closed loop logs requests, not predictions",
                "teacher_forced": {"test": {m: {"failed": v["failed"], "rows": v["rows"],
                                                "text": pct(v["failed"] / v["rows"])} for m, v in pf_t.items()},
                                   "x1": {m: {"failed": v["failed"], "rows": v["rows"],
                                              "text": pct(v["failed"] / v["rows"])} for m, v in pf_x.items()}},
                "closed_loop": {m: {**v, "bounds_text": f"{pct(v['rate_bounds'][0])}–{pct(v['rate_bounds'][1])}"}
                                for m, v in cl["parse"].items()},
                "content_on_parsed_rows": {"test": test["content_teacher_forced"],
                                           "x1": x1["content_teacher_forced"]}},
        "R13": small["R13"],
        "R14": small["R14"],
        "R15": {"C_fm": arm("C_fm", "C_fm success (%)"),
                **{n: diff(t[n]) for n in ("H1b A - C_fm", "H1b B - C_fm", "H1b D - C_fm")}},
        "R16": {"what": "X1: per-world drop of k=1..4 exact (test - X1), paired against JEV",
                "vs_A_jev": {n: diff(d) for n, d in ro["h7"]["vs_A_jev"].items()},
                "mean_drop": {m: pp(v) for m, v in ro["h7"]["mean_drop"].items()},
                "x1_k4": {m: pct(x1["curves"][m][str(DEPTH)]["value"]) for m in MODELS},
                "x1_tests": {n: diff(d) for n, d in x1["tests"].items()}},
        "R17": r17,
        "R18": {"G": {s: {m: pp(v) for m, v in ro[s]["accumulation"]["G"].items()} for s in ("test", "x1")},
                "pairs": {s: {n: {"value": p["diff"], "ci": p["ci"], "text": pp(p["diff"]), "ci_text": ci_text(p["ci"])}
                              for n, p in ro[s]["accumulation"]["pairs"].items()} for s in ("test", "x1")},
                "source": "artifacts/kiis_k5v3|kiis_x1v3/accumulation_h8.json (kiis_k5_rollout.py accumulation)"},
        "R19": {"what": "closed-loop replay: entering the goal room without the food",
                "entered": {m: [v["entered"], v["n"]] for m, v in g.items()},
                "success_after": {m: [v["success_after"], v["entered"]] for m, v in g.items()},
                "success_never_entered": {m: [v["success_never_entered"], v["never_entered"]] for m, v in g.items()},
                "replaced_food_pickup": {m: v["by_policy_choice"]["take food"] for m, v in over.items()},
                "take_food_parse_failures": cl["d0_replay"]["take_food_parse_failures"]},
        "R20": {**r20, "text": pct(r20["in_train_used"] / r20["test"]),
                "text_full_train": pct(r20["in_train"] / r20["test"])},
    }


def table1(s: dict, cl: dict) -> tuple[str, list[list[str]]]:
    rows = [("Base agent, no WM (C)", "C"), ("+ failure memory, no WM (C_fm)", "C_fm"),
            ("JEV, typed, frozen (A)", "A_jev"), ("Qwen, typed, zero-shot (B0)", "B0_typed"),
            ("Qwen, generative, zero-shot (D0)", "D0_gen"), ("Qwen+LoRA, typed (B)", "B_typed"),
            ("Qwen+LoRA, generative (D)", "D_gen"), ("Validity oracle", "validity"), ("Full oracle", "oracle")]
    a = cl["arms"]
    body = [["Agent", "Success (%) [95% CI]", "Invalid (%)", "WM cost / ep."]]
    for label, arm in rows:
        if arm == "A_jev":
            cost = f"{a[arm]['wm_requests_per_ep']:.0f} req., ${cl['jev']['per_episode_usd']:.3f}"
        elif a[arm]["wm_gpu_s_per_ep"] is not None:
            cost = f"{a[arm]['wm_gpu_s_per_ep']:.0f} GPU-s"
        else:
            cost = "–"
        body.append([label, f"{pct(a[arm]['success'])} {ci_text(a[arm]['ci'])}",
                     pct(a[arm]["invalid_rate"]), cost])
    tf = s["R12"]["teacher_forced"]["test"]
    note = (f"N = {s['R14']['n_transitions']:,} training transitions for B and D "
            f"(B {s['R14']['gpu_hours']['B_typed']:.1f}, D {s['R14']['gpu_hours']['D_gen']:.1f} GPU-h). "
            f"Parse failures per one-step prediction: D0 {tf['D0_gen']['text']}%, D {tf['D_gen']['text']}%. "
            f"B and D matched the full oracle's outcome in every episode.")
    md = "| " + " | ".join(body[0]) + " |\n|" + "---|" * len(body[0]) + "\n"
    md += "".join("| " + " | ".join(r) + " |\n" for r in body[1:])
    return md + "\n" + note + "\n", body


def fig2_svg(ro: dict) -> str:
    style = {"A_jev": ("#222222", "", "JEV (A)"), "B0_typed": ("#1f6fb4", "6,4", "Qwen typed (B0)"),
             "B_typed": ("#1f6fb4", "", "Qwen+LoRA typed (B)"), "D0_gen": ("#c8372d", "6,4", "Qwen gen. (D0)"),
             "D_gen": ("#c8372d", "", "Qwen+LoRA gen. (D)"), "persistence": ("#8a8a8a", "2,3", "persistence")}
    panels = [("(a) Test worlds", ro["test"]["curves"], ("A_jev", "B0_typed", "B_typed", "D0_gen", "D_gen", "persistence")),
              ("(b) New structure (X1)", ro["x1"]["curves"], ("A_jev", "B_typed", "D_gen", "persistence"))]
    W, H, pw, ph, top, left, gap = 700, 350, 280, 220, 30, 55, 60
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           'font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for i, (title, curves, models) in enumerate(panels):
        x0 = left + i * (pw + gap)
        X = lambda k: x0 + (k - 1) * pw / 3
        Y = lambda v: top + ph * (1 - v)
        out.append(f'<text x="{x0 + pw / 2}" y="{top - 10}" text-anchor="middle" font-weight="bold">{title}</text>')
        for v in range(0, 101, 20):
            y = Y(v / 100)
            out.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + pw}" y2="{y}" stroke="#e5e5e5"/>')
            out.append(f'<text x="{x0 - 6}" y="{y + 4}" text-anchor="end">{v}</text>')
        for k in range(1, DEPTH + 1):
            out.append(f'<text x="{X(k)}" y="{top + ph + 16}" text-anchor="middle">{k}</text>')
        out.append(f'<rect x="{x0}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#444"/>')
        out.append(f'<text x="{x0 + pw / 2}" y="{top + ph + 34}" text-anchor="middle">rollout steps k</text>')
        if i == 0:
            out.append(f'<text transform="translate({x0 - 38},{top + ph / 2}) rotate(-90)" text-anchor="middle">'
                       'exact state match (%)</text>')
        for m in models:
            color, dash, _ = style[m]
            pts = " ".join(f"{X(k)},{Y(curves[m][str(k)]['value'])}" for k in range(1, DEPTH + 1))
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>')
            if m != "persistence":
                for k in range(1, DEPTH + 1):
                    out.append(f'<circle cx="{X(k)}" cy="{Y(curves[m][str(k)]["value"])}" r="3" fill="{color}"/>')
    for j, m in enumerate(("A_jev", "B_typed", "D_gen", "persistence", "B0_typed", "D0_gen")):
        color, dash, label = style[m]
        lx = left + (j % 3) * 200
        y = top + ph + 58 + (j // 3) * 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 24}" y2="{y}" stroke="{color}" stroke-width="2"{dash_attr}/>')
        out.append(f'<text x="{lx + 30}" y="{y + 4}">{label}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def build() -> dict:
    cl = closed_loop()
    ro = rollouts()
    small = small_slots()
    s = slots(cl, ro, earlier_designs(), structure_overlap(), small)
    v = verdicts(cl, ro)
    a = cl["arms"]
    wm = sorted({f"{100 * a[m]['success']:.0f}" for m in ("A_jev", "B_typed", "D_gen")}, key=float)
    return {"condition": cl["condition"], "episodes": cl["episodes"],
            "bootstrap": {"draws": aa.B, "clusters": "world (24)", "seed": SEED},
            "slots": s, "verdicts": v, "sentences": sentences(v),
            "abstract_online": {"C": f"{100 * a['C']['success']:.0f}",
                                "world_models": "–".join(wm),
                                "C_fm": f"{100 * a['C_fm']['success']:.0f}",
                                "B0_minus_D0": f"{100 * cl['tests']['H2  B0 - D0']['diff']:.0f}"},
            "_closed_loop": cl, "_rollouts": ro}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="recompute and compare with the saved slots.json")
    args = ap.parse_args()
    res = build()
    public = {k: v for k, v in res.items() if not k.startswith("_")}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "slots.json"
    if args.check:
        saved = json.loads(path.read_text())
        same = all(saved.get(k) == json.loads(json.dumps(public[k])) for k in public)
        print("slots.json is up to date" if same else "slots.json differs from a fresh computation")
        return 0 if same else 1
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    path.write_text(json.dumps({"revision": rev, **public}, indent=1, ensure_ascii=False) + "\n")
    md, body = table1(public["slots"], res["_closed_loop"])
    (OUT / "table1.md").write_text(md)
    with open(OUT / "table1.csv", "w", newline="") as f:
        csv.writer(f).writerows(body)
    (OUT / "fig2.svg").write_text(fig2_svg(res["_rollouts"]))
    with open(OUT / "fig2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "model", "k", "exact", "ci_lo", "ci_hi"])
        for split in ("test", "x1"):
            for m, cs in res["_rollouts"][split]["curves"].items():
                for k, c in cs.items():
                    lo, hi = c.get("ci", [None, None])
                    w.writerow([split, m, k, f"{c['value']:.4f}", "" if lo is None else f"{lo:.4f}",
                                "" if hi is None else f"{hi:.4f}"])
    print(f"wrote {path.relative_to(ROOT)}, table1.md/.csv, fig2.svg/.csv")
    for h, v in public["verdicts"].items():
        print(f"  {h:<18} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

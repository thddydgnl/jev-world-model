# A World Model for LLM Agents

**Comparing a Decision Model with General-Purpose LLMs**
(LLM 에이전트의 월드모델: 결정모델과 범용 LLM의 비교)

English | [한국어](README.ko.md)

Code, data, run logs, analysis and manuscript for a paper submitted to the **KIIS 2026 Fall Conference** (Korean Institute of Intelligent Systems, oral presentation track, Nov 26–28, 2026). Authors: **Yong Hwi Song** (School of Information and Communication Engineering, Chungbuk National University) and **Keon Myung Lee** (School of Software, Chungbuk National University).

> **Paper:** [`KIIS2026f_원고.pdf`](kiis2026f/paper/KIIS2026f_원고.pdf) is the completed manuscript (Oct 8, 2026; two pages, society template), [`KIIS2026f_원고.docx`](kiis2026f/paper/KIIS2026f_원고.docx) the submission file and [`원고.md`](kiis2026f/paper/원고.md) the same text in Markdown. The PDF was rendered on macOS with substitute fonts, so line breaks may differ slightly from the society's print version.

---

## 1. Overview

LLM agents often choose actions without knowing their consequences, so they repeat invalid or irrelevant commands. A **world model** that predicts the result of an action *before* it is executed can fix this. The usual way to build one is to have an LLM write the next state as text, either zero-shot or after fine-tuning.

This project asks a different question: **can a *decision model* serve as the agent's world model?** A decision model takes a state and a question and returns a probability distribution over fixed options, instead of generating text. We use **JEV** (TypeSafe AI, `jev-1.13.0`), **frozen and never trained on the task**, and ask it seven multiple-choice questions per action: *does the command execute?* and *what is the value of each of six state variables afterwards?* The answers are assembled into the next state, and the step is repeated to look several actions ahead.

The same agent is then run with LLM world models in a **2 × 2 design** (query style × training), with everything else held fixed:

| | No training | Trained on 6,000 engine-labelled transitions |
|---|---|---|
| **Decision-style** (ask with options, read probabilities) | **A** JEV (frozen) · **B0** Qwen3-4B | **B** Qwen3-4B + LoRA |
| **Generative** (write the next state, parse it) | **D0** Qwen3-4B | **D** Qwen3-4B + LoRA |
| No world model | **C** policy only · **C_fm** policy + failure memory | |
| Reference | **oracle** engine truth plugged into the same planner | |

Every agent shares the same policy (Qwen3-4B), the same candidate plans, the same planner and utility. **Only the world model differs.**

**Three findings** (TextWorld, 24 unseen-name evaluation worlds, 288 episodes per agent):

1. **A frozen decision model works as a world model without task training.** JEV reaches 93.4 % task success, above the failure-memory baseline (79.5 %) and not distinguishable from the oracle (92.7 %).
2. **Zero-shot, asking an LLM with options beats letting it write the next state** (83.0 % vs 59.0 %). The gap is a *format* effect: 26.0 % of generated states were unparseable, and the parse-failure rule turned those into "fails". On parsed answers the generative model was not less accurate.
3. **Fine-tuning closes the gap on the trained structure but does not transfer.** LoRA-tuned B and D match the oracle episode-for-episode and predict 4-step rollouts at 98.5 % / 96.8 %, above JEV (77.9 %). On a new world structure never seen in training, they drop to 71.5 % / 66.5 % while JEV stays at 85.5 %: a drop 25.6 and 26.7 points larger than JEV's.

---

## 2. Paper

| | |
|---|---|
| Title (KO) | LLM 에이전트의 월드모델: 결정모델과 범용 LLM의 비교 |
| Title (EN) | A World Model for LLM Agents: Comparing a Decision Model with General-Purpose LLMs |
| Authors | Yong Hwi Song¹ · Keon Myung Lee² — ¹School of Information and Communication Engineering, ²School of Software, Chungbuk National University |
| Venue | KIIS 2026 Fall Conference (한국지능시스템학회 2026 추계학술대회), oral presentation track, paper no. 00027 |
| Status | Abstract submitted Oct 1, 2026 · manuscript completed Oct 8 · camera-ready due Oct 23 · conference Nov 26–28 (Gimpo University Global Campus) · Proceedings of KIIS Fall Conference 2026, Vol. 36, No. 2 |
| Format | A4, two pages, two columns, society Word template |
| Keywords | World Model, LLM Agent, Decision Model, Fine-tuning, TextWorld |

Files in [`kiis2026f/paper/`](kiis2026f/paper/README.md):

- [`원고.tmpl.md`](kiis2026f/paper/원고.tmpl.md) — the manuscript text. Every number is a named slot filled from `results/slots.json`; the build refuses to run if a number or a verdict the text relies on has changed.
- [`원고.md`](kiis2026f/paper/원고.md), [`KIIS2026f_원고.docx`](kiis2026f/paper/KIIS2026f_원고.docx), [`KIIS2026f_원고.pdf`](kiis2026f/paper/KIIS2026f_원고.pdf) — build outputs (`scripts/kiis_paper.py`).
- [`fig1.svg`](kiis2026f/paper/fig1.svg) (one prediction step in each style), [`fig2.png`](kiis2026f/paper/fig2.png) (rollout accuracy), [`fig_architecture.svg`](kiis2026f/paper/fig_architecture.svg) (system diagram, for the talk).
- [`report/LLM_에이전트의_월드모델_설명.docx`](kiis2026f/report/LLM_에이전트의_월드모델_설명.docx) — a 12-page explanatory report in Korean (background, comparison design, method, setup, results, interpretation), built from the same `slots.json` ([`report/README.md`](kiis2026f/report/README.md)).

### Abstract

We present a method that uses JEV, a decision model that answers with probabilities over options, as the world model of an LLM agent, and compare it with using an LLM directly or after fine-tuning. On a TextWorld task, JEV used without task training reached a success rate of 93.4 %, higher than an agent that only remembers failed actions (79.5 %) and not distinguishable from a reference with perfect predictions (92.7 %). An untrained LLM reached 83.0 % when asked with options and 59.0 % when asked to generate the next state; the difference came from 26.0 % of generated states being unparseable. An LLM fine-tuned on 6,000 transitions reached 92.7 % on the trained structure, but on a new structure its multi-step prediction accuracy fell 25.6–26.7 points more than JEV's.

---

## 3. Results

### Table 1 — closed-loop task success (24 test worlds × 4 start states × 3 policy seeds = 288 episodes per agent)

| Agent | Success (%) [95 % CI] | Invalid actions (%) | World-model cost / episode |
|---|---|---|---|
| C (policy only) | 17.0 [7.3, 29.2] | 87.8 | – |
| C_fm (+ failure memory) | 79.5 [70.5, 87.5] | 35.0 | – |
| **A (JEV, frozen)** | **93.4 [85.1, 98.6]** | 15.4 | 129 requests, $0.010 |
| B0 (LLM, decision-style) | 83.0 [73.3, 91.3] | 48.6 | 49 GPU-s |
| D0 (LLM, generative) | 59.0 [48.6, 69.1] | 36.3 | 115 GPU-s |
| B (LLM + LoRA, decision-style) | 92.7 [85.8, 97.9] | 13.2 | 42 GPU-s |
| D (LLM + LoRA, generative) | 92.7 [85.8, 97.9] | 13.2 | 73 GPU-s |
| Oracle (engine truth) | 92.7 [85.8, 97.9] | 12.8 | – |

Confidence intervals are world-clustered bootstrap (24 worlds, 4,000 resamples). B and D were fine-tuned on the same 6,000 transitions (13.9 and 4.5 GPU-h). Parse failures per one-step prediction: D0 26.0 %, D 0.1 %. B and D matched the oracle's outcome in every one of the 288 episodes.

Paired differences that matter: A − C_fm **+13.9 pts [2.1, 24.3]**; B0 − D0 **+24.0 pts [11.5, 35.8]**; A − oracle +0.7 [−1.7, 3.1] and A − B0 +10.4 [−1.7, 21.5] are *not* distinguishable at this sample size.

### Figure 2 — state prediction accuracy vs rollout length

![Figure 2](kiis2026f/paper/fig2.png)

Exact state match after k predicted steps (beam-top state, all variables correct). (a) 800 rollouts on the test worlds; (b) 400 rollouts on a **new structure** (extra side room, decoy key, second container; 9 state variables) that no model was trained on. *persistence* predicts that nothing changes.

| k = 4 exact match | JEV (A) | B0 | B | D0 | D | persistence |
|---|---|---|---|---|---|---|
| Test worlds | 77.9 | 9.1 | **98.5** | 6.2 | 96.8 | 15.6 |
| New structure | **85.5** | 4.2 | 71.5 | 2.2 | 66.5 | 18.0 |

Untrained LLMs fall below persistence from k = 2 on. JEV's errors are almost entirely in the *does it execute* judgement (7.0 % one-step); when that judgement is right its state errors are 0.0 %.

### Figure 1 — why zero-shot generative fails

![Figure 1](kiis2026f/paper/fig1.png)

The most common D0 failure: the model writes the apple's new location without deleting the old one, so one object is in two places, the answer is unparseable, and the rule "unparseable after one regeneration = the command fails" penalises taking the food. Agents then enter the far room without the food (D0 40 % of episodes vs C_fm 21 %) and almost never recover.

### Pre-registered hypotheses and verdicts

All expectations were committed before the main run ([`kiis2026f/실험계획.md`](kiis2026f/실험계획.md) §5). A difference whose world-level 95 % CI contains 0 is reported as *not distinguishable*, never as "equal".

| # | Expectation | Closed-loop success | k-step prediction |
|---|---|---|---|
| H1 | World model > C | holds (A, B, D; +75 pts) | — |
| H1b | World model > C_fm | holds (A +13.9, B/D +13.2) | — |
| H2 | Zero-shot: decision-style > generative | holds (+24.0) | borderline (+2.3 [−0.1, 4.6]) |
| H3 | Fine-tuning shrinks the style gap | holds (−24.0) | not distinguishable |
| H4 | Fine-tuned decision-style ≥ JEV | not distinguishable (both at oracle) | holds on test (+15.4); **reversed** on new structure (−10.1) |
| H5 | JEV > zero-shot LLM | not distinguishable (+10.4) | holds (+63.2) |
| H6 | Accuracy falls with k, generative zero-shot most | — | holds (D0 −33.8 pts k=1→4) |
| H7 | Fine-tuned models drop more than JEV on a new structure | — | holds (B +25.6, D +26.7 larger drop) |
| H8 | Decision-style accumulates less error than generative (same LLM) | — | not distinguishable |

Full numbers: [`kiis2026f/results/slots.json`](kiis2026f/results/slots.json), [`artifacts/kiis_k4v3/README.md`](artifacts/kiis_k4v3/README.md), [`artifacts/kiis_k5v3/README.md`](artifacts/kiis_k5v3/README.md).

### What this work does *not* claim

The paper plan keeps a list of claims the data cannot support ([`kiis2026f/논문구성안.md`](kiis2026f/논문구성안.md) §9). The main ones:

- Decision-style is not shown to be a *more accurate* world model than generative: on parsed answers D0 was not worse than B0. The advantage is **format robustness**, and its size depends on the parse-failure rule.
- JEV is not shown to be better than a fine-tuned LLM (or vice versa) in closed loop; all three sit at the oracle.
- Nothing is claimed beyond one task structure, one LLM (Qwen3-4B), one training seed and one parsing rule. Evaluation worlds have new names but the same structure as training (69.5 % of evaluation transitions exist in training data up to renaming); the new-structure result is a rollout-only measurement.
- The oracle is a diagnostic reference, not a performance ceiling; the planner's remaining failures are candidate shortage and planner limits, not world-model errors.

---

## 4. Method

![Architecture](kiis2026f/paper/fig_architecture.svg)

**Agent loop (receding horizon).** At every step the policy (Qwen3-4B, sampling T = 0.7) proposes up to 16 candidate plans of length 2. The planner scores every prefix of length 1–2 by imagining its end state with the world model and computing

```
J = conj + 0.25·progress − 0.1·E[#invalid commands] − 0.01·h + 0.05·policy prior
```

where *conj* is the probability that all three goal atoms hold and *progress* the fraction that hold. Only the first action of the best prefix is executed; the agent re-plans from the real observation. Prefixes that start with a command already seen to fail in the current state are dropped (failure memory, shared by every planning agent). Coefficients were fixed in the design document and never tuned.

**World model = one-step prediction, applied recursively.** Every world model is only ever asked "from this state, attempt this one command". Multi-step lookahead feeds the predicted state back in. A beam of width 4 carries the *executes* / *fails* branches weighted by the execution probability (generative models have no probability, so their beam is effectively 1). No transition rules are coded anywhere: the next state is rebuilt from the predicted variable values alone, which a round-trip test guarantees is lossless.

**Decision-style query (A, B0, B).** Two requests per (state, command): one with the six state-variable questions (room of the player; location of the key and the two foods; open/closed/locked state of the box and the door), each with all type-valid values as options; and a *separate* request asking whether the command executes. Keeping the validity question separate matters: bundled with the state questions it drifts toward "executes" (measured 0.285 → 0.576). For the LLM the option probability is the next-token distribution over the option letters, renormalised. JEV is used through its public probability interface only.

**Generative query (D0, D).** The LLM writes `result: executes|fails` followed by every fact of the resulting state as `predicate(arg, arg)` lines; a strict parser requires exactly one legal value per variable. An unparseable answer is regenerated once (sampling), and if it still fails it is scored as "fails, state unchanged".

**Fine-tuning (B, D).** Same 6,000 transitions from 150 training worlds, labelled by the game engine (never by JEV). LoRA r = 16, α = 32, dropout 0.05 on all attention and MLP projections; loss only on answer tokens; AdamW lr 1e-4, 3 % warm-up, micro-batches of 8,192 tokens × 4 accumulation, 2 epochs, epoch chosen by transition accuracy on 200 validation transitions. B learns one-token answers to 7 questions per transition (42,000 examples); D learns one fact list per transition. Validation: B0 52 % → B 100 %, D0 40 % → D 100 %.

**Fairness devices.** The policy's random seed is keyed on (world, root, step) only, so every agent receives identical candidates in identical states; the LoRA adapter is switched off while the policy samples. Each run records a *condition hash* (every fixed setting) and an *arm hash*; the analysis script refuses to combine logs whose hashes differ.

---

## 5. Experimental setup

| | |
|---|---|
| Environment | TextWorld 1.7.0, symbolic JSON backend, fully observable, deterministic. The policy sees the static command catalogue, never the list of currently admissible commands. |
| Task | Two rooms. A closed box holds the key to the locked door; the goal food is on a table; a decoy food lies on the floor. Goal: hold the goal food, with the door open, in the far room. Shortest solution: 6 actions. |
| Worlds | Same structure, different vocabulary per world. Split **by vocabulary**: dev 12 / train 150 / val 20 / **test 24** (words never seen in training or calibration). World fingerprints are pinned in `kiis2026f/worlds_manifest_v3.json`. |
| Episodes | 24 test worlds × 4 start states × 3 policy seeds = **288 per agent**, 9 agents, step cap 30. |
| Rollout evaluation | 800 rollouts (400 starts × {policy plan, random sequence}) of length 4 on test worlds; 400 on the new structure (X1). Metric: exact state match at k = 1..4; also teacher-forced vs free-running (error accumulation) and beam-1 controls. |
| Statistics | World is the independent unit. 95 % CIs from world-clustered bootstrap (4,000 resamples); agents compared as paired differences on identical (seed, world, start). |
| Calibration | Task difficulty and policy prompt were chosen on dev/val worlds using **only agents without a world model** (criteria G1–G5 fixed in advance: oracle 90 % or more, oracle − validity gap, candidate coverage, C ≤ 20 %, monotonicity). Test worlds were never used to change code, prompts or coefficients. |
| Hardware | Local Mac for JEV calls, labelling and analysis; a lab server with 2 × RTX A5000 24 GB for Qwen3-4B inference and LoRA training. |
| Cost of the final runs | JEV: $2.77 (closed loop) + $2.42 (rollouts). GPU: ≈ 53 h wall-clock on two GPUs for the closed loop, 44.9 GPU-h for rollouts, 18.5 GPU-h for training. |

---

## 6. Repository layout

```
.
├── README.md · README.ko.md
├── 아이디어.md            research proposal: question, JEV analysis, five architecture candidates (Sep 20)
├── 설계.md                implementation design: interfaces, utility, training, evaluation, MVP plan
├── src/
│   ├── env/              worlds.py (world family, vocabulary split, fingerprints) · serialize.py (canonical state)
│   │                     transitions.py (engine-labelled one-step transitions)
│   ├── agent/            policy.py (Qwen3-4B candidate planner) · task.py (goals, state schema, utility, failure memory)
│   ├── wm/               recursive_forecaster.py (one-step rollout, beam) · jev_forecaster.py · llm_backends.py
│   ├── jev_client.py     pinned jev-1.13.0, schema validation, budget cap, retries, raw-response archive
│   ├── forecast.py       JEV request builder
│   └── runinfo.py        condition hash, arm hash, code revision
├── scripts/              pipeline stages (see §7) and MVP-era scripts
├── tests/                13 standalone regression tests (round-trip, parity, parser leakage, contracts, bootstrap)
├── data/
│   ├── phase2/worlds     320 world definitions
│   ├── kiis/             transition sets and rollout files (large .jsonl excluded from git)
│   └── mvp_a/            engine-labelled counterfactual dataset of the MVP phase
├── artifacts/            every run: run_manifest.json, logs, summaries and a README per stage
│   ├── kiis_k4v3/        closed-loop main run (Table 1), hypotheses.json, failure-cause replay
│   ├── kiis_k5v3/ kiis_x1v3/   rollout evaluation (Figure 2), error accumulation
│   ├── kiis_k3/          LoRA training records and validation gates
│   ├── kiis_wcal/ kiis_v1cal/ kiis_v1bcal/   calibration runs (no-world-model agents only)
│   └── mvp_b/ mvp_c/ wm_criteria/ …          MVP-phase results (earlier conditions, not in Table 1)
├── kiis2026f/            the paper workspace
│   ├── paper/            manuscript source, build outputs, figures
│   ├── results/          slots.json, table1, fig2 — generated by scripts/kiis_report.py
│   ├── report/           12-page explanatory report (docx) and its builder
│   ├── 실험계획.md         experiment plan: agent definitions, frozen conditions, stages K0–K7, hypotheses, change log
│   ├── 논문구성안.md       paper plan: result slots R1–R20, claims not made, talk outline
│   ├── 전체_실험_해설.md   study guide of every experiment, decision and reason
│   ├── 공개_LLM_월드모델_JEV_비교계획.md   plan for the next study (ScienceWorld / ALFWorld)
│   └── worlds_manifest*.json   pinned world fingerprints
├── docs/                 MVP-era protocol (confirmatory vs exploratory), claims ledger, limitations ledger, arm A design
├── fullpaper/            plan for the extended journal version
├── environment.lock      pinned Python packages of the local side
└── .env.example          TYPESAFE_API_KEY placeholder
```

Most planning and record documents are in Korean; code, logs and manifests are in English.

**What is not in the repository.** Raw JEV requests/responses (`artifacts/jev_raw/`): their public-release scope under TypeSafe's customer agreement is unconfirmed, so only aggregates are published. Per-episode `.jsonl` logs and LoRA adapter weights are excluded for size; `kiis_report.py --check` recomputes `slots.json` from them on a machine that has them.

---

## 7. Reproducing

### Environment

```bash
python -m venv .venv && .venv/bin/pip install -r <(grep -v '^#' environment.lock)   # local side: textworld, httpx, …
cp .env.example .env                                                                 # TYPESAFE_API_KEY=…
.venv/bin/python scripts/phase0_api_check.py                                         # confirms model jev-1.13.0 and schema
```

The GPU side used Python 3.10, `torch 2.6.0+cu124`, `transformers 5.17.0`, `peft 0.21.0`, `textworld 1.7.0`, and `Qwen/Qwen3-4B` in bf16 (two processes per 24 GB GPU).

### Tests (no GPU, no API key)

```bash
for t in tests/test_*.py; do .venv/bin/python "$t" || break; done
```

### Pipeline (stage names follow `kiis2026f/실험계획.md`)

| Stage | Command | Output |
|---|---|---|
| K0 worlds | `scripts/kiis_worlds.py` | `kiis2026f/worlds_manifest_v3.json` |
| K2 transitions | `scripts/kiis_transitions.py --v3` | `data/kiis/transitions_v3/` |
| K3 training | `scripts/kiis_train_wm.py --style typed\|gen --n 6000 --epochs 2 --lr 1e-4 --micro-tokens 8192 --accum 4 --eval-n 200 --seed 0 --out …` | adapter, `train_meta.json` |
| K3 validation | `scripts/kiis_validate_wm.py --adapter …` · `scripts/kiis_check_k3.py` | 14 gates |
| W calibration | `scripts/kiis_calibrate.py` | `artifacts/kiis_wcal/` |
| K4 closed loop | `scripts/kiis_k4_queue.py --version v3` (runs `scripts/mvp_c_headroom.py` per arm) | `artifacts/kiis_k4v3/` |
| K4 analysis | `scripts/kiis_k4_hypotheses.py` · `scripts/kiis_k4_ceiling.py` · `scripts/kiis_k4_replay.py` | `hypotheses.json`, failure causes |
| K5 rollouts | `REUSE_ROLLOUTS=1 bash scripts/kiis_run_k5v3.sh` (→ `scripts/kiis_k5_rollout.py build/eval/report/compare/accumulation`) | `artifacts/kiis_k5v3/`, `kiis_x1v3/` |
| K6 numbers | `scripts/kiis_report.py` (`--check` to verify) | `kiis2026f/results/` |
| K7 manuscript | `python3 scripts/kiis_paper.py --pdf` (macOS + Word) | `kiis2026f/paper/` |

Every run writes a `run_manifest.json` with the condition hash (`448b4fac3efe` for the final v3 condition), arm hash, code revision, package versions and the fingerprints of the worlds it used.

---

## 8. How the project evolved

The repository keeps the full record of design changes, including the ones that invalidated earlier results. Dates are 2026.

| Date | Step | What happened |
|---|---|---|
| Sep 20 | Proposal and design | Research question, analysis of what JEV's public interface does and does not allow, five architecture candidates. |
| Sep 21 | Phase 0, MVP-A/B/C | API pinned; engine-labelled counterfactual dataset (0 label violations); JEV's one-shot "state after a whole sequence" forecasts were 100 % correct on executed paths but **failed a pre-registered order-swap probe 0/18**; oracle headroom over the plain agent +60 pts, mostly action validity. |
| Sep 22 | Architecture switch | A direct validity question is answered at 92 %; the **recursive one-step** architecture passes all three operational world-model criteria (order swap 14/18); fixes to failure memory, invalid-count penalty and question separation. |
| Sep 23 | KIIS plan, K0–K1 | 2 × 2 + C design, conditions frozen with hashes, vocabulary-based splits, pilot on dev worlds. |
| Sep 24 | K2–K3 | Typed and generative training pipelines, tiny-overfit checks, LoRA training of B (13.9 GPU-h) and D (4.5 GPU-h), 14/14 validation gates. |
| Sep 26 | K4 v1, K5 v1 | 2,304 closed-loop episodes: **every good world model tied with the oracle at 58.7 %**, because the policy never proposed the needed action in 76 % of oracle failures. Rollout evaluation worked (JEV 81.9 % at k = 4). Decision: keep v1 as a record and re-calibrate the task on dev/val only. |
| Sep 26–27 | v2 | Calibration with no-world-model agents under pre-committed criteria; B and D retrained; the v2 pilot showed the calibration did not transfer to test names. Decision: v3. |
| Sep 27–28 | v3 (W) | New test vocabulary, stricter criteria (oracle ≥ 90 %, monotonicity), a prompt lever against repetition; the v1 adapters passed reuse checks on v3 validation. |
| Sep 28–30 | K4v3, K5v3, X1 | 2,592 closed-loop episodes over three seeds and 1,200 rollouts × 5 models, through an API outage, an OOM and a scheduler fix, all recorded. |
| Sep 30 | K6, K7 | Every number computed from logs into `slots.json`; paired bootstrap corrected to count a world drawn twice twice; manuscript drafted and revised. |
| Oct 1–8 | Submission | Abstract submitted (paper 00027); extended technical report written; title and author list finalized and manuscript completed (Oct 8). |

---

## 9. Limitations and next steps

The results come from one task structure, one LLM, one training seed and one parse-failure rule. The evaluation worlds share their structure with the training worlds, so the fine-tuned models' near-perfect test numbers are in-distribution; the only out-of-structure measurement is the rollout evaluation. JEV's responses are non-deterministic (7.5 % of rollout cells changed across three repeats), and the state-variable schema is researcher-designed. The planner looks ahead only two actions.

The planned extension ([`fullpaper/`](fullpaper/README.md), [`kiis2026f/공개_LLM_월드모델_JEV_비교계획.md`](kiis2026f/공개_LLM_월드모델_JEV_비교계획.md)) moves to ScienceWorld and ALFWorld, compares against a public generative world model, adds training-size curves and longer horizons, and tests whether the format-robustness advantage survives when the generative output is schema-constrained.

---

## 10. Citation

```bibtex
@inproceedings{song2026worldmodel,
  title     = {A World Model for LLM Agents: Comparing a Decision Model with General-Purpose LLMs},
  author    = {Song, Yong Hwi and Lee, Keon Myung},
  booktitle = {Proceedings of the KIIS 2026 Fall Conference (Korean Institute of Intelligent Systems), Vol. 36, No. 2},
  year      = {2026},
  note      = {Oral presentation. Korean title: LLM 에이전트의 월드모델: 결정모델과 범용 LLM의 비교}
}
```

See also [`CITATION.cff`](CITATION.cff).

## 11. License and acknowledgements

Code is released under the [MIT License](LICENSE). The manuscript, figures and reports are © 2026 the authors. TextWorld is MIT-licensed (Microsoft); Qwen3-4B is Apache-2.0 (Alibaba); JEV is a commercial API of TypeSafe AI and its raw outputs are not redistributed here.

**Contact:** Yong Hwi Song · thddydgnl1937@gmail.com · [github.com/thddydgnl](https://github.com/thddydgnl)

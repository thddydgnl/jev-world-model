# 실험 프로토콜 — 확증 / 탐색 구분

**실행일:** 2026-09-21 · **모델:** `jev-1.13.0` (pinned) · **환경:** TextWorld 1.7.0

이 문서의 목적은 하나다: **무엇이 데이터를 보기 전에 정해졌고, 무엇이 데이터를 본
뒤에 나왔는지**를 논문에서 흐리지 않게 고정하는 것. 두 종류의 주장은 증거 강도가
다르며, 섞어 쓰면 결과 전체의 신뢰가 무너진다.

---

## A. 확증적 (Confirmatory) — JEV 호출 이전에 고정됨

근거 문서: `설계.md` §24 (sanity check와 중단 기준), §25 (Feasibility MVP),
§9 (action conditioning 검사세트), §19 (최소 ablation AB1).
이 문서들은 첫 API 호출보다 먼저 작성되었다.

### A.1 사전 고정된 게이트와 임계값

| # | 게이트 | 임계값 | 출처 | 결과 |
|---|---|---|---|---|
| 1 | 0-step 현재 사실 추출 정확도 | > 95% | §24 "0-step extraction" | **100.0%** PASS |
| 2 | changed-fact 정확도 − persistence | ≥ +10%p | §25 "약 10 percentage points" | **+96.6%p** PASS |
| 3 | 반사실 pair-exact | ≥ 50% | §24 "effect sensitivity" | **94.0%** PASS |
| 4 | 행동 제거 대조 순반응성 | ≥ +30%p | §19 AB1 | **+87.4%p** PASS |
| 5 | oracle headroom (MVP-C) | ≥ ~10 points | §24 "oracle headroom" | **+60.4%p** PASS |

임계값은 결과를 본 뒤 조정하지 않았다 (§24: "실험 후 결과가 유리해지도록 gate를
바꾸지 않는다").

### A.2 사전 고정된 실험 설계

- **순서 교환 pair** (`open→take` vs `take→open`): `설계.md` §9에 명시된 검사.
  MVP-A 데이터 생성 시점에 이미 포함되었고, JEV 호출 전에 GT가 확정되었다.
- **null 대조 pair**: 실제 결과가 같은 행동쌍 (§9 "같은 결과 pair는 불필요한
  예측 변화의 음성대조군").
- **changed / unchanged 층화** (§21.1 "no-change shortcut").
- **0-step 조건**: JEV 능력이 아니라 입력 가독성을 재는 것으로 사전 정의 (§25 MVP-B).
- **데이터 규모**: 8 worlds × 4 roots × 6 sequences × h∈{1,2}, seed 20260921.

### A.3 확증적 결과

1. 다섯 게이트 모두 통과.
2. **순서 교환 pair는 0/18 실패.** 이것은 사전 등록된 검사의 실패이며, 따라서
   확증적 negative result다. 탐색 중 우연히 발견한 것이 아니다.
3. MVP-C에서 oracle headroom +60.4%p, world 단위 10승 2무 0패 (n=12).

### A.4 실행 중 변경 1건 — 반드시 논문에 기록

MVP-C 최초 파일럿(4 pair)에서 arm C의 invalid rate가 100%로 나왔다. 원인은
정책 프롬프트에 **행동 이력이 없어** 무효 행동이 no-op → 상태 불변 → 동일 제안이
반복되는 퇴행 루프였다.

`설계.md` §3.5는 "지난 실제 행동과 그 관측된 성공·실패"를 온라인 허용 정보로
명시하므로, 이는 **사양 대비 구현 누락**이지 결과를 보고 설정을 유리하게 바꾼
것이 아니다. 이력을 추가하고(모든 arm에 동일) **전량 재실행**했으며, 파일럿
수치는 폐기했다.

그럼에도 이것은 "중간 결과를 본 뒤의 변경"이므로 여기에 기록한다.
파일럿 수치(C 0%, oracle 50%)는 `artifacts/mvp_c/` 에 남기지 않았다.

### A.5 사양 대비 편차

| 항목 | 설계 | 실제 | 사유 |
|---|---|---|---|
| step cap | 30 (§3.3) | 15 | MVP 규모. 최적 경로 5단계 대비 3배 여유 |
| episodes | 50–100 (§25) | 96 (48 pair) | 범위 내 |
| world template | 다양화 (§17) | 1종 변형 | MVP 규모 |

---

## B. 탐색적 (Exploratory) — 데이터를 본 뒤 도출됨

### B.1 발견 경위

게이트 3의 집계치(pair-exact 94%)와 순서 교환 pair(0/18) 사이의 불일치를
확인하는 과정에서 도출되었다. 즉 **A.3의 확증적 실패를 설명하기 위한 사후 분석**이다.

### B.2 탐색적 가설

> JEV는 행동의 precondition을 모델링하지 않고, 명령이 항상 성공한다고 가정한다.

### B.3 탐색적 증거

prefix에 포함된 "엔진이 거부한 명령 수"로 층화한 정확도:

| h | 거부된 명령 | n | 정확도 |
|---:|---:|---:|---:|
| 1 | 0 | 525 | 100.0% |
| 1 | 1 | 435 | 91.3% |
| 2 | 0 | 320 | 100.0% |
| 2 | 1 | 410 | 90.2% |
| 2 | 2 | 230 | 82.6% |

`take→open` 조건에서 JEV는 32/32 모두 `inventory`로 예측(상자가 닫힌 19건 전부 오답).

**이 층화 변수(거부된 명령 수)는 사전에 정의되지 않았다.** `설계.md` §10.1에
`invalid_attempt_count`가 target으로 존재하지만, 이를 *정확도의 층화 축*으로
쓰겠다는 계획은 없었다.

### B.4 논문에서의 취급

- B.2를 확증된 결론으로 쓰지 않는다. "사후 분석에서 관찰되었다"로 기술한다.
- 확증하려면 **아래 C의 사전 등록 재현 실험**이 필요하다.

---

## C. B를 확증으로 승격시키기 위한 사전 등록 (미실행)

아래는 지금 고정하며, 새 데이터를 보기 전에 작성되었다.

**가설 C1.** 새로 생성한 held-out world 집합에서, JEV의 endpoint 정확도는
prefix 내 엔진 거부 명령 수가 0인 항목에서 ≥95%, 1개 이상인 항목에서 그보다
최소 15%p 낮을 것이다.

**가설 C2.** 이 격차는 질문 종류와 무관하게 나타나되, 거부된 명령이 직접 관여한
entity의 질문에서 가장 클 것이다.

**설계.** 기존과 교차하지 않는 world template 2종 이상(용기 중첩, 3방 이상),
world 16개 이상, root 64개 이상. 층화 변수는 위 정의 그대로 사용하고,
분석은 world 단위 clustered bootstrap으로 한다.

**반증 조건.** 거부 0개와 1개 이상의 격차가 15%p 미만이면 B.2를 철회하고,
순서 교환 실패를 template 특수성으로 재해석한다.

---

## D. 이번 실행에서 하지 않은 것

- MVP-C (oracle headroom 대비 C): 미실행. 따라서 **"WM이 Agent를 개선한다"는
  어떤 주장도 현재 근거가 없다.** 본 실행은 intrinsic 예측 능력만 다룬다.
- h=3, h=4: 미검증.
- Arm B (학습된 typed LLM), Arm C (No-WM policy): 미구현.
- 통계적 추론: 현재 수치는 점추정이며 CI를 계산하지 않았다. world 8개는
  world-clustered CI를 의미 있게 내기에 부족하다.

---

## E. 재현 방법

```bash
python -m venv .venv && .venv/bin/pip install -r <(grep -v '^#' environment.lock)
export TYPESAFE_API_KEY=...        # 또는 .env / Keychain
.venv/bin/python scripts/phase0_api_check.py
.venv/bin/python scripts/phase0_smoke.py
.venv/bin/python scripts/mvp_a_dataset.py
.venv/bin/python scripts/mvp_a_audit.py
.venv/bin/python scripts/mvp_b_jev.py
.venv/bin/python scripts/mvp_b_analyze.py
.venv/bin/python scripts/mvp_b_precondition.py
```

seed·설정·해시는 `artifacts/run_manifest.json`. 원시 요청/응답 448건은
`artifacts/jev_raw/`에 보관되어 있으므로, `jev-1.13.0`이 은퇴해도 intrinsic
분석은 재현 가능하다. 단 **공개 범위는 MCA 확인 대상**이다 (`docs/limitations.md`).

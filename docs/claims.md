# 주장 대장 (claims ledger)

`limitations.md`의 짝. 저쪽이 "무엇을 조심할지"라면 여기는 **"무엇을 말할 수
있는지"**다. 논문을 쓸 때 여기서 시작한다.

각 주장은 **근거 → 강도 → 무엇이 이걸 깨는가** 순으로 적는다.
강도 표기: **확증** = 데이터 보기 전 고정된 지표 / **탐색** = 사후 도출 /
**불가** = 현재 근거 없음.

범위: Phase 0 + MVP-A + MVP-B + MVP-C (2026-09-21). `jev-1.13.0`, TextWorld symbolic,
완전관측, h∈{1,2}.
MVP-A/B: 8 worlds / 32 roots / 384 prefixes / 1,920 labels / 448 API calls.
MVP-C: 12 worlds / 48 paired episodes / Qwen3-4B on RTX A5000.

---

## A. 지금 쓸 수 있는 주장 (확증)

### A1. typed 판단 출력은 행동 조건부 미래 정보를 담는다

**근거.** 실제 결과가 현재 상태와 달라진 항목에서 JEV의 정확도는
h=1 **100.0%** (74/74), h=2 **96.6%** (148/148 중 143). 같은 항목에서
persistence(현재 사실 유지)는 정의상 0%.

**깨지는 조건.** 이 격차가 다른 world template에서 재현되지 않으면.

> 논문 문장 예: "On facts whose value changes as a result of the action
> sequence, the typed forecasts are correct for 100% (h=1) and 96.6% (h=2) of
> items, against a persistence baseline that is necessarily 0% on this stratum."

### A2. 현재 상태 분류기가 아니다 — 행동에 실제로 반응한다

**근거.** 사전 등록된 행동 제거 대조(AB1). 정답이 바뀌어야 하는 항목에서
예측도 바뀐 비율 **96.6%** (143/148), 정답이 그대로인 항목에서 예측이 바뀐
비율 **9.2%** (75/812). 순반응성 **+87.4%p**.

**깨지는 조건.** 순반응성이 다른 설정에서 30%p 아래로 떨어지면.

### A3. 결과는 입력 가독성 문제가 아니다

**근거.** 0-step 조건(현재 사실 추출) **160/160 = 100%**, 5개 질문 전부 100%.
직렬화·명명·선택지 매핑이 원인이라는 대안 설명을 배제한다.

**깨지는 조건.** 없음. 이 수치는 상한이다.

> 이 주장은 작아 보이지만 **A1·A2를 해석 가능하게 만드는 전제**다.
> 논문에서 빼지 말 것.

### A4. 사전 등록된 순서 민감도 검사에 실패한다

**근거.** `open→take` vs `take→open` (q_key_parent, h=2), 정답이 다른 18쌍에서
둘 다 맞힌 경우 **0/18**. 실패 양상이 18/18 동일: `take→open`을 항상
`inventory`로 예측. 정답 확률은 0.13–0.33으로 신호는 있으나 argmax가 일관되게 틀림.

**이 검사는 `설계.md` §9에 첫 API 호출 이전에 명시되어 있었다.**
(`docs/protocol.md` §A.2 참조.) 따라서 사후에 고른 지표가 아니다.

**깨지는 조건.** 프롬프트 형식을 바꿔 성능이 회복되면 — 미검증이며
현재 최대 교란 요인(`limitations.md` §4).

> 논문 문장 예: "A pre-registered order-sensitivity probe fails completely
> (0/18), with an identical error mode in every case."

### A5. 이 진단 환경에는 WM이 기여할 여지가 크다

**근거.** MVP-C. 동일 world·goal·seed·후보집합·효용함수에서 선택 규칙만 바꾼
48 pair 비교. C 성공률 **35.4%**, 같은 후보를 실제 endpoint로 채점한 oracle
**95.8%** → headroom **+60.4%p**. world 단위로 oracle 우세 10 / 동률 2 / 열세 0 (n=12).

임계값 ≥+10%p는 `설계.md` §24 "oracle headroom"에 사전 명시.

**깨지는 조건.** 난도를 낮추거나 step cap을 늘리면 C가 따라잡아 격차가 줄 수 있다.
oracle은 상한이 아니다(§22.4).

> 논문 문장 예: "Holding the world, goal, candidate set and utility fixed and
> varying only the selection rule, replacing the policy's own preference with
> the true endpoint raises success from 35.4% to 95.8% (n=48 paired episodes,
> favourable in 10 of 12 worlds and unfavourable in none)."

---

## B. 조건부로 쓸 수 있는 주장 (탐색 — 헤지 필수)

### B1. 오류는 precondition 위반에 국소화된다

**근거.** prefix에 포함된 "엔진이 거부한 명령 수"로 층화한 정확도:

| h | 거부 | n | 정확도 |
|---:|---:|---:|---:|
| 1 | 0 | 525 | **100.0%** |
| 1 | 1 | 435 | 91.3% |
| 2 | 0 | 320 | **100.0%** |
| 2 | 1 | 410 | 90.2% |
| 2 | 2 | 230 | 82.6% |

거부 0개에서는 변화한 사실만 봐도 h1 74/74, h2 66/66 = 100%.

**강도.** 층화 변수가 **사후 정의**되었다. A4의 실패를 설명하려고 만든 축이다.

**승격 조건.** `docs/protocol.md` §C의 held-out 실험(가설 C1: 격차 ≥15%p).

> 논문 표현: "In a post-hoc analysis, …" 또는 확증 실험 후 A-tier로 이동.
> **지금 단계에서 "JEV assumes commands always succeed"를 단정하지 말 것.**

### B2. 정상 실행 경로에서는 완벽하다

**근거.** B1의 거부 0개 행 (845건 전수 정답).

**강도.** B1과 동일하게 사후 층화에 의존. 또한 **world template 1종**이므로
템플릿 단순성이 상당 부분 기여했을 수 있다.

> 이게 현재 결과의 **가장 강한 긍정 수치**다. B1의 부정 서술에 가려지지 않게
> 할 것. 다만 확증 전까지는 "in this diagnostic suite"로 범위를 묶는다.

### B3. 그 여지는 대부분 '행동 유효성'이다

**근거.** 선택된 행동이 실제 실행 가능했던 비율 — C **14.3%** (89/621) 대
oracle **85.4%** (276/323). C 실패 31건 중 26건이 progress 0.00에서 정지.

**강도.** 유효성 분해는 **사후** 분석이며, validity-only arm을 실제로
돌리지 않았다. `설계.md` §19 AB3가 그 arm이다.

**승격 조건.** validity-only arm(효용에서 conj·progress를 제거하고
invalid 항과 policy prior만 사용) 실행. 그 성공률이 oracle에 가까우면 B3 확증.

### B4. 두 MVP를 합치면 본 실험에 대한 예측이 나온다

MVP-C의 여지는 precondition 중심이고(B3), MVP-B는 JEV가 precondition에
정확히 취약함을 보인다(B1). 따라서:

> **예측 P1.** frozen JEV 기반 A는 MVP-C가 보인 +60.4%p headroom의 상당 부분을
> 회수하지 못한다.
> **예측 P2.** 환경 정답으로 학습한 B는 precondition을 학습하므로 A를 능가한다.
> **예측 P3.** validity-only arm이 A와 비슷하거나 더 낫다.

**강도.** 예측이다. 본 실험 전에 사전등록하면 확증 가능해진다.
지금 주장으로 쓰지 말 것.

---

## C. 아직 쓸 수 없는 주장 (불가)

| 주장 | 왜 불가능한가 | 무엇이 필요한가 |
|---|---|---|
| **JEV가** Agent 성능을 개선한다 | A arm 미구현. MVP-C는 oracle만 측정 | Phase 4 |
| frozen JEV가 학습된 LLM보다 낫다/못하다 | Arm B 미구현 | Phase 5 |
| 생성형 WM보다 효율적이다 | Arm D 없음 | `설계.md` §20 선택 D |
| h가 늘면 무너진다 | h∈{1,2}만 측정 | h=3,4 |
| 다른 환경으로 일반화된다 | template 1종 | 새 template 2종 이상 |
| 확률이 잘 보정되어 있다 | calibration·ECE 미측정 | validation split |
| JEV 내부 표현이나 RLCD의 효과 | 원리적으로 식별 불가 | — (주장하지 않음) |

---

## 논문 골격으로 옮기면

현재 재료로 쓸 수 있는 가장 정직하고 가장 센 이야기:

1. **Setup** — 공개 typed API를 행동 조건부 미래 예측기로 쓰는 인터페이스 (A3로 타당성 확보)
2. **It works** — A1, A2. 행동 조건부 정보가 실재하며 현재 상태 분류가 아니다
3. **But there is a sharp seam** — A4. 사전 등록 검사의 완전 실패, 단일 실패 양상
4. **Where the seam is** — B1, B2. 사후 분석이 실패를 precondition 위반에 국소화
5. **And the seam is where the value is** — A5, B3. 같은 환경에서 WM이 회수할
   여지는 +60.4%p인데 그 대부분이 precondition이다. 즉 JEV의 약점과 과제의
   병목이 **같은 지점**이다
6. **What it implies** — B4의 예측 P1–P3, `설계.md` §11 utility의 낙관 편향,
   AB3의 주 실험 승격

3번이 논문의 중심이다. **사전 등록된 실패**는 흔치 않은 증거이고,
4번이 그걸 해석 가능한 발견으로 바꾼다.

# Arm A 설계 — JEV를 planner에 연결하는 구조

작성 2026-09-21. `설계.md` §7–§11에 대한 **측정 기반 개정안**이다.
근거는 전부 오늘 실행한 MVP-A/B/B2/C·AB3에 있다.

---

## 1. 설계를 규정한 측정값

| 관찰 | 수치 | 출처 |
|---|---|---|
| 0-step 현재 사실 추출 | **100%** (160/160) | MVP-B |
| endpoint 예측, **전 명령 정상 실행** 시 | **100%** (845건) | MVP-B §B1 |
| endpoint 예측, 거부 명령 1개 | 90–91% | MVP-B |
| endpoint 예측, 순서 교환(실패 포함) | **0/18** | MVP-B |
| **유효성 직접 질의** (지금 실행되나?) | **91.6%** | MVP-B2 |
| **유효성 직접 질의** (1-step 뒤) | **92.2%** | MVP-B2 |
| headroom 중 유효성이 차지하는 몫 | **97%** | AB3 |

읽는 법: **JEV는 원자적이고 현재 상태로 판정 가능한 질문에 강하다.**
조건 연쇄("이게 성공하면 그다음은…")를 내부에서 처리하지 못한다.

> **설계 원칙: 연쇄를 JEV에게 시키지 않는다. 우리가 조합한다.**

---

## 2. 실행 패턴 분해

후보 prefix `u = (a₁, …, a_h)`에 대해, 각 명령의 성공 여부 조합이
실행 패턴을 만든다. h=2면 4가지:

| 패턴 | 실제 실행되는 행동열 | 확률 |
|---|---|---|
| ✓✓ | `(a₁, a₂)` | `p₁ · p₂` |
| ✓✗ | `(a₁)` | `p₁ · (1−p₂)` |
| ✗✓ | `(a₂)` | `(1−p₁) · p₂′` |
| ✗✗ | `()` → 현재 상태 (공짜) | `(1−p₁)(1−p₂′)` |

`p₂` = a₁ 성공 후 a₂의 유효성, `p₂′` = a₁ 실패 후(= 현재 상태에서) a₂의 유효성.

**모든 패턴이 "전부 정상 실행되는 행동열"로 환원된다.** 그 조건에서 JEV의
endpoint 정확도는 100%다. 즉 **JEV에게 실패를 추론하라고 요구하지 않는다.**

```
JEV 에게 묻는 것                     우리가 하는 것
──────────────────────────         ─────────────────────────
p₁  = P(a₁ 실행 | 현재)        ─┐
p₂  = P(a₂ 실행 | a₁ 시도 후)  ─┤
E(a₁,a₂) = endpoint 분포       ─┼──→  Σ P(패턴) × 패턴별 endpoint
E(a₁)    = endpoint 분포       ─┤      = 목표 항
E(a₂)    = endpoint 분포       ─┘
현재 상태 = 이미 앎 (질의 불필요)
```

### 가지치기

`p₁ ≥ 0.95`이면 ✗ 가지를 버린다. 대부분의 후보에서 ✓✓ 하나만 남아
요청 수가 MVP-B 수준으로 돌아온다. 길이-1 prefix의 endpoint는 planner가
어차피 따로 평가하므로 **재사용**한다.

---

## 3. 효용함수 (설계.md §11 개정)

```
J(u; g) = Σ_패턴 P(패턴) · [ G_g(end) + α · (1/M)Σ_j g_j(end) ]
        − λ · Σ_i (1 − p_i)                    ← 기대 무효 횟수
        − c · h
        + β · b_π(u)
```

**변경점.** §10.1은 `invalid_attempt_count`(행동열 전체 실패 횟수)를 하나의
Choice로 물었다. MVP-B2는 **명령별 이진 유효성**이 측정된 형식(92%)임을
보였으므로, 기대 무효 횟수를 `Σ(1−p_i)`로 직접 구성한다.

계수는 §11의 초기값 유지: `α=0.25, λ=0.1, c=0.01, β=0.05`.
arm별 튜닝 금지, train/validation에서만 공통 조정.

---

## 4. 질문 형식 (측정된 문구 그대로)

### 4.1 유효성 — MVP-B2에서 92%

```json
{
  "type": "choice",
  "instructions": "Consider the command below against `current_state` as it is now. Command: `take brass key from wooden box`. Would this command actually execute, or would it fail because its requirements are not met in that state? Answer about whether it RUNS, not whether it is useful.",
  "criteria": {
    "executes": "The command runs and takes effect.",
    "fails": "The command cannot run; its preconditions are unmet, so nothing changes."
  }
}
```

1-step 뒤 유효성은 `instructions` 앞에 프레임만 바꾼다:

> "Starting from `current_state`, first attempt these commands in order:
> 'open wooden box'. A command that cannot run fails and changes nothing.
> THEN consider the command below."

**`"not whether it is useful"` 을 빼지 말 것.** 유효성과 바람직함을 분리한다.

### 4.2 endpoint — MVP-B에서 정상 실행 시 100%

```json
{
  "type": "choice",
  "instructions": "Starting from `current_state`, attempt exactly these 2 command(s) in this order: `open wooden box` then `take brass key from wooden box`. Apply `rollout_convention`. Report the resulting state AFTER the sequence, not the current state. Question: Where is the brass key located? Give its DIRECT container, supporter, room floor, or the player's inventory.",
  "criteria": { "inventory": "...", "in:c_0": "...", "at:r_0": "...", "on:s_0": "..." }
}
```

**선택지는 타입상 가능한 모든 값을 열거한다.** MVP-A에서 target support
위반 0으로 검증됨. 예시처럼 몇 개만 고르면 정답이 선택지 밖으로 나간다.

---

## 5. 상태 페이로드

MVP-B에서 0-step 100%를 만든 구성을 그대로 유지한다.

```json
{
  "current_state": {
    "player_location": "r_0",
    "entities": [{"id":"k_0","name":"brass key","type":"k"}, ...],
    "facts_canonical": [["in","k_0","c_0"], ["closed","c_0"], ...],
    "facts_readable": ["The brass key (k_0) is inside the wooden box (c_0).", ...]
  },
  "action_sequence": ["open wooden box", "take brass key from wooden box"],
  "horizon": 2,
  "rollout_convention": "Attempt each command in order ... A command that cannot be executed fails and changes nothing ...",
  "query_catalog": [ ... ]
}
```

- canonical ID **와** 사람이 읽는 렌더링을 **둘 다** 넣는다. 후자가 빠지면
  0-step 정확도가 떨어질 위험이 있고, 그게 모든 해석의 전제다.
- `query_catalog` 전체를 넣어 arm B와 **정보를 대칭**시킨다(§7).
  비용이 약 60% 늘지만 공정성 근거다.
- 온라인 whitelist는 `설계.md` §3.5 그대로. `admissible_commands`,
  `_valid_actions`, `_winning_policy` 는 절대 넣지 않는다.

---

## 6. 요청 묶기

같은 요청의 질문들은 **독립 평가**되고 state를 공유한다(공식 문서 확인).

| 요청 | 공유 state | 질문 |
|---|---|---|
| **1개** | root | 모든 후보 **첫 행동의 유효성** (K=8) |
| **prefix당 1개** | root + `action_sequence` | 그 prefix의 endpoint 질문들 + **다음 행동 유효성** |

K=8, H=2 → 중복 제거 후 약 12 prefix → 계획스텝당 **약 13 요청**.
동시성 8에서 약 1.5초.

캐시 키는 §19 그대로:
`arm, model_version, schema_version, state_hash, action_sequence, query_catalog_hash, calibration_version`.

---

## 7. 알려진 위험

| 위험 | 근거 | 대응 |
|---|---|---|
| **확신에 찬 오류** | 유효성 오판 시 평균 p=0.76 (정답 거절 시 0.09) | 확률 임계값으로 못 거름. §17 calibration을 validation에서 필수 수행 |
| `insert` 51.5%, `close` 66.7% | MVP-B2 동사별 분해 | 현 과제 체인에는 미등장. 다른 과제로 확장 시 재측정 |
| 92%가 planner에서 몇 점인지 미지 | intrinsic 수치일 뿐 | **arm A를 만들어야 답이 나옴.** 이 설계의 목표는 "JEV에게 잘하는 질문만 던지기"이지 92%→100%가 아님 |
| 탐침 집합이 인위적 50/50 | MVP-B2 한계 | 실제 planner 후보 분포에서 재측정 |
| 패턴 분해가 독립성을 가정 | `p₂`는 a₁ 성공 조건부로 물었으나 결합분포는 아님 | §10.3의 conjunction bound를 consistency diagnostic으로 기록 |

---

## 8. 검증 순서 (구현 시)

1. **parity unit test** — 합성 forecast를 넣어 arm A와 oracle이 같은 확률에서
   같은 행동을 고르는지 (§32 "같은 확률에서 A/B의 행동이 다름" 대응).
2. **p₁ 보정 검사** — validation에서 예측 유효성 vs 실제 유효성 reliability diagram.
3. **가지치기 영향** — `p₁≥0.95` 컷의 유무로 결정이 달라지는 비율.
4. **ablation** — 유효성 질문 제거(= MVP-B 방식 그대로)했을 때 성능 차이.
   이것이 본 설계의 기여를 직접 측정한다.

4번이 핵심이다. **"질문을 어떻게 쪼개느냐가 성능을 가른다"**가 이 연구의
가장 새로운 주장이 될 수 있다.

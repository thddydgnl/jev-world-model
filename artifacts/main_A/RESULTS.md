# Arm A 본 실행 — JEV를 planner에 연결한 결과 (2026-09-21)

`docs/arm_a_design.md` 구현. 48 에피소드 짝비교(같은 world·goal·seed·후보·효용),
함정 world, cap 30. arm A만 GPU0, validity+oracle은 GPU1에서 병렬 실행했고
정책 seed가 (world, root, arm, step)에 고정되어 있어 분할이 안전하다.

## 결과

| arm | n | success | steps | invalid | progress |
|---|---:|---:|---:|---:|---:|
| validity | 48 | 75.0% | 13.4 | 10.9% | 87.5% |
| **A (JEV)** | 48 | **93.8%** | 10.6 | 38.2% | 96.5% |
| oracle | 48 | 97.9% | 8.4 | 23.1% | 99.3% |

| 분해 | 값 |
|---|---:|
| 예측 구간 `oracle − validity` | **+22.9%p** |
| **arm A 회수 `A − validity`** | **+18.8%p (구간의 82%)** |
| 상한과 차이 `A − oracle` | −4.2%p |

world 단위 A vs validity: 우세 **5** / 동률 7 / 열세 **0** (n=12).

## 핵심 관찰 — 이득의 출처는 유효성 분류가 아니다

| arm | invalid rate |
|---|---:|
| validity | 10.9% |
| **A** | **38.2%** |
| oracle | 23.1% |

**arm A는 유효 행동 회피가 validity arm보다 3.5배 나쁜데 성공률은 18.8%p 높다.**
따라서 이득을 "JEV가 유효성을 잘 분류해서"로 설명할 수 없다. 목표 지향
endpoint 예측이 기여한다.

AB3에서 headroom의 77~79%가 유효성이었으므로 "JEV는 좋은 precondition
분류기일 뿐"이라는 대안 가설이 있었는데, 이 대조가 그것을 크게 약화시킨다.
완전 배제하려면 질문 분리 ablation이 필요하다.

## 비용·부하

| 항목 | 값 |
|---|---:|
| JEV 요청 | 10,927 (에피소드당 228) |
| 입력 토큰 | 21,224,588 |
| 비용 | **$0.891** |
| 정책 호출 | 510 (oracle 수준으로 효율적) |
| **conjunction bound 위반** | **1,609 / 10,927 = 14.7%** |

bound 위반은 질의한 conjunction 확률이 marginal들의 Fréchet 범위를 벗어난
비율이다(`설계.md` §10.3). 자동 보정하지 않고 기록만 했다. **JEV의 joint
일관성에 대한 측정 가능한 결함**이며, 성공률 집계가 이를 가린다.

## 사전 등록 예측의 기각

`docs/claims.md` B4에 commit `edda2ef`로 고정했던 예측:

> **P1.** frozen JEV 기반 A는 headroom의 상당 부분을 회수하지 못한다.

**기각되었다.** A는 예측 구간의 82%를 회수했다.

P1의 전제는 MVP-B의 "JEV는 precondition을 모델링하지 않는다"였다.
MVP-B2가 그 전제를 뒤집었고(유효성 직접 질의 92%), 그에 맞춰 질문을
재설계하니 실제로 회수되었다. **예측을 먼저 기록했기 때문에 "왜 틀렸는가"가
서사가 된다.**

## 한계

- **단일 seed, n=48, CI 없음.** `A − oracle = −4.2%p`는 **노이즈 범위이므로
  "A ≈ oracle"이라고 말할 수 없다.** `A − validity = +18.8%p`는 오늘 실측한
  노이즈 폭(~7%p)보다 크므로 읽을 수 있다.
- world template 1종, h≤2, 함정 world 한정.
- 쉬운 world에서는 예측 구간 자체가 +2.1%p였다(AB3). 이 결과는
  **"WM이 필요한 과제에서" arm A가 작동한다**까지만 말한다.
- 질문 분리 ablation 미실행 → 유효성 질문의 기여분 미정량.

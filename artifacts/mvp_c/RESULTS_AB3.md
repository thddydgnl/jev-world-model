# AB3: validity-only ablation (2026-09-21)

`설계.md` §19 AB3 / §20 "validity-only". 세 arm이 **동일한** world·goal·seed·
후보집합·효용함수를 쓰고, 효용에 들어가는 **정보량만** 다르다.

| arm | 효용에 쓰는 정보 |
|---|---|
| C | 정책 선호만 (효용 미사용) |
| validity | 실제 invalid count + policy prior. **목표 항(conj·progress)을 0으로** |
| oracle | 위 + 실제 endpoint의 목표 달성·진척 |

validity는 oracle과 **같은 엔진 분기를 읽되** "명령이 실행되는가"만 쓴다.
따라서 `oracle − validity`가 **유효성 필터를 넘어선 목표 지향 예측의 기여**다.

## 결과 (48 paired episodes, 12 worlds)

| arm | n | success | steps | invalid | progress |
|---|---:|---:|---:|---:|---:|
| C | 48 | 29.2% | 13.3 | 87.5% | 34.0% |
| **validity** | 48 | **93.8%** | 6.8 | 18.0% | 95.8% |
| oracle | 48 | 95.8% | 6.7 | 15.3% | 96.5% |

| 분해 | 값 | 비중 |
|---|---:|---:|
| 전체 headroom (oracle − C) | +66.7%p | 100% |
| **유효성으로 설명 (validity − C)** | **+64.6%p** | **97%** |
| 목표 예측이 필요 (oracle − validity) | +2.1%p | 3% |

world 단위 oracle vs validity: 우세 2 / 동률 9 / 열세 1 (n=12).
에피소드 단위 불일치 **5/48** (oracle만 3, validity만 2) — 유의차 없음.

## 해석

**이 과제는 World Model을 필요로 하지 않는다.** "명령이 실행되는가"만 알면
93.8%가 풀리고, 완벽한 결과 예측을 더해도 +2.1%p다.

MVP-B와 겹치면 문제가 분명해진다. JEV는 precondition에 취약하므로
**중요한 64.6%p에서 약하고 무의미한 2.1%p를 놓고 경쟁**하게 된다.
이 환경에서는 RQ3("어떤 예측 품질이 행동 개선을 만드는가")을 **측정할 신호가
없다**.

## 사전 등록 확인

예측 **P3**("validity-only arm이 A와 비슷하거나 더 낫다")은 이 실험을 돌리기
전에 `docs/claims.md` B4에 기록되어 commit `edda2ef`로 고정되었다.
AB3 자체도 `설계.md` §19에 사전 명시된 ablation이다. 따라서 이 결과는
**확증적**이다.

다만 P3의 정확한 대상은 arm A(JEV)였고, 여기서 비교한 것은 oracle이다.
"완벽한 예측조차 유효성 너머로는 +2.1%p"라는 더 강한 형태로 확인되었다.

## 원인 — JEV가 아니라 환경 설계

world에 **비가역성이 없다**. 모든 유효 행동이 중립이거나 전진이고, 목표를
달성 불가능하게 만드는 행동이 없다. 따라서 "유효한 것 중 정책 선호를 따른다"로
충분하다. **예지력은 일부 유효 행동이 나쁠 때만 값어치가 있다.**

→ 후속: `build_trap_world` (비가역 함정 `eat {목표물}`) 에서 재측정.

## 부수 발견 — 재실행 간 변동

동일 설정 재실행에서 C 성공률이 **35.4% → 29.2%** 로 6.2%p 흔들렸다.
바뀐 것은 정책 seed 소비 순서뿐이다(arm 추가로 `Policy.calls` 진행이 달라짐).

**n=48·CI 없음에서 10%p 미만 차이는 해석하지 않는다.** 본 실험에서는
`설계.md` §22.2대로 다중 seed와 world-clustered bootstrap이 필수다.

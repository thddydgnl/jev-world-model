# MVP-C 결과 — oracle headroom (2026-09-21)

**질문:** World Model이 기여할 여지가 애초에 존재하는가? (`설계.md` §25 MVP-C)

두 arm이 **동일한** world·goal·seed·후보집합·효용함수를 쓰고, 다른 것은
**어느 prefix를 고르는가** 하나뿐이다.

- **C** — 정책 자신의 최선호 plan의 첫 행동
- **oracle** — 같은 후보 prefix들을 **실제 endpoint**로 채점해 argmax

## 설정

| 항목 | 값 |
|---|---|
| policy | Qwen3-4B (bf16), temp 0.7, top_p 0.9, seed 20260921 |
| 하드웨어 | RTX A5000 24GB ×2 중 1장 (`yonghwi-racedreamer`) |
| worlds | 12 (전부 잠긴 문 → 5단계 의존 체인) |
| episodes | 48 pair = 96 episodes |
| K / H / step cap | 8 / 2 / 15 |
| goal | 3-atom conjunction: `in(key,I) ∧ open(door) ∧ at(P,room_b)` |
| catalog | 31 commands (정적, type-correct) |
| policy calls | 944 (repair 0, fallback 0) |

## 결과

| arm | n | success | steps | invalid rate | progress |
|---|---:|---:|---:|---:|---:|
| C | 48 | 35.4% | 12.9 | 85.7% | 38.9% |
| oracle | 48 | **95.8%** | 6.7 | 14.6% | 97.2% |

**oracle headroom = +60.4%p** (게이트 ≥ +10%p) → **PASS**

world 단위(독립 단위, `설계.md` §22.2): oracle 우세 **10** / 동률 2 / 열세 **0** (n=12).

## 핵심 해석 — headroom은 대부분 "행동 유효성"이다

선택된 행동이 실제로 실행 가능했던 비율:

| arm | 유효 step / 전체 | 비율 |
|---|---:|---:|
| C | 89 / 621 | **14.3%** |
| oracle | 276 / 323 | **85.4%** |

C의 실패 31건 중 **26건이 progress 0.00**에서 멈췄다 — 열쇠조차 얻지 못했다.
Qwen3-4B가 목표 관련 행동(`open steel door`)을 제안하지만 precondition을
만족하지 못한다.

## MVP-B와의 교차 — 본 실험에 대한 예측

- **MVP-C:** 여지가 +60.4%p 있고, 그 대부분이 **행동 유효성**이다.
- **MVP-B:** JEV는 **precondition에 정확히 취약하다** (거부 명령 0개 100% →
  2개 82.6%, 순서교환 0/18).

두 결과를 합치면: **JEV 기반 WM은 이 headroom의 상당 부분을 회수하지 못할
가능성이 높다.** 이것은 본 실험 전에 사전등록 가능한 날카로운 예측이다.
동시에 환경 정답으로 학습한 B는 precondition을 배울 수 있으므로
**B > A**가 예측된다.

`설계.md` §19의 AB3(full forecast vs validity-only)는 이제 부차 ablation이
아니라 **주 실험의 핵심**이다.

## 알려진 편차·한계

- **step cap 15** (`설계.md` §3.3은 30). MVP 규모를 위한 축소. C에 불리하게
  작용할 수 있으나 최적 경로가 5단계라 3배 여유가 있다.
- **파일럿 후 설정 변경 1건.** 최초 파일럿에서 정책에 행동 이력을 주지 않아
  C가 무효 행동을 15회 반복하는 퇴행이 발생했다. `설계.md` §3.5가 "지난 실제
  행동과 그 관측된 성공·실패"를 온라인 허용 정보로 명시하므로 이는 사양 대비
  구현 누락이었고, 수정 후 전량 재실행했다. 파일럿 수치는 폐기.
- world template 1종(MVP-A/B와 동일 계열, 문 잠김 변형만 사용).
- oracle은 **성능 상한이 아니다**. 효용이 근시안적이고 후보가 같은 정책에서
  나온다 (`설계.md` §22.4).
- CI 미계산. world 12개.

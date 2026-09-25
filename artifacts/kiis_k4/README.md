# K4 — 본 실행, closed-loop (test world)

기준 계획: `kiis2026f/실험계획.md` K4, 사전점검 `kiis2026f/K4_사전점검.md`.
실행 revision `fff42e3` (수정 파일 0), 조건 해시 `0fe479090329`.
실행: 2026-09-24 23:35 KST → 2026-09-26 00:18 KST (약 24.7시간), `scripts/kiis_run_k4.py --mode full`.

## 실행 기록

- 정책 seed 3개(20260921, 777, 1234) × 작업 4개 = 12개 프로세스, **모두 exit 0**.
  [`launch_status.json`](launch_status.json)
- test world 24개 × root 4개 × seed 3개 = **arm당 288 episode, 8 arm, 합계 2,304 episode**.
  예정한 episode가 모두 결과(성공 또는 step 제한 30)를 갖는다.
- 12개 매니페스트의 조건 해시, revision, 24개 world 지문이 모두 같다. world 지문은
  `kiis2026f/worlds_manifest.json`의 test 24개와 일치한다. arm 해시는 arm마다 하나이고,
  사전점검에 적은 값과 같다.
- **통과 기준(예정한 episode마다 결과나 실패 사유가 있고, 설정 해시가 전부 같다): 통과.**

실행 전에 같은 스크립트의 `--mode smoke`가 test te000/r0, seed 20260921로 8 arm을 한 번씩
돌렸다(`../kiis_k4_smoke/`). smoke 뒤 코드는 바뀌지 않았고, smoke의 8개 episode는 본 실행의
같은 episode와 상태·step·무효 수까지 같다. 변경 기록은 실험계획 §11.

## 표 1 원자료 (seed 3개 합산, world-clustered 95% CI)

전체 출력: [`analysis.txt`](analysis.txt) (`scripts/analyze_arms.py`).

| arm | 성공률 [95% CI] | 평균 steps | 무효율 | 세계모델 비용 / episode |
|---|---:|---:|---:|---|
| C | 2.8% [0, 7] | 29.6 | 95.4% | — |
| validity (유효성 oracle) | 53.1% [40, 67] | 20.2 | 71.5% | — |
| oracle | 58.7% [44, 72] | 19.1 | 73.7% | — |
| A_jev | 58.7% [45, 72] | 19.2 | 74.4% | JEV 82.5 요청, $0.0060 |
| B0_typed | 47.6% [35, 61] | 23.8 | 77.1% | GPU 36.2 s |
| D0_gen | 41.0% [30, 52] | 24.2 | 71.5% | GPU 94.2 s, 파싱 실패 20.2% |
| B_typed | 58.7% [45, 72] | 19.2 | 73.9% | GPU 52.3 s |
| D_gen | 58.7% [45, 72] | 19.1 | 73.7% | GPU 100.0 s, 파싱 실패 0.02% (1/6,000) |

JEV 비용은 seed별 $0.567 / $0.575 / $0.591, 합계 **$1.733** (23,758 요청).

짝차이 (같은 seed·world·root, * = CI가 0을 포함하지 않음):

| 비교 | 차이 [95% CI] | 한쪽만 성공 |
|---|---:|---|
| A_jev − C | +55.9%p [45.6, 66.1] * | 162 / 1 |
| B_typed − C | +55.9%p [45.6, 66.1] * | 162 / 1 |
| D_gen − C | +55.9%p [45.8, 66.0] * | 162 / 1 |
| B0 − C | +44.8%p [35.0, 54.5] * | 130 / 1 |
| D0 − C | +38.2%p [29.2, 47.2] * | 111 / 1 |
| B0 − D0 | +6.6%p [0.6, 12.2] * | 48 / 29 |
| B − D | 0.0%p [−0.6, 0.6] | 1 / 1 |
| B − A | 0.0%p [−1.3, 1.3] | 6 / 6 |
| A − B0 | +11.1%p [4.8, 17.2] * | 51 / 19 |
| A − validity | +5.6%p [2.1, 8.9] * | 22 / 6 |
| A − oracle | 0.0%p [−1.5, 1.6] | 6 / 6 |
| oracle − validity | +5.6%p [2.1, 8.3] * | 17 / 1 |

사전 기대(§5): H1 성립(A·B·D 모두 C보다 높음), H2 성립(B0 > D0), H3 성립(학습 후 차이
+6.6 → 0.0%p), H4 성립(B − A = 0, CI [−1.3, 1.3]), H5 성립(A > B0). H6은 K5에서 판정.

**해석상 주의: A·B·D가 oracle과 같은 58.7%다.** 세 모델 모두 이 계획기와 후보 집합으로
도달할 수 있는 상한에 있어서, closed-loop 성공률로는 셋의 차이를 가를 수 없다. H4의 성립은
"B가 A와 같은 상한에 도달했다"는 뜻이다. 세계모델 사이의 비교는 K5(k-step 정확도)가 맡는다.

무효율은 실패 episode(대부분 30 step 동안 무효 행동을 반복)가 평균을 끌어올린다. 성공 episode만
보면 oracle의 평균 step은 11.5다.

## oracle 상한은 왜 dev(89.6%)보다 낮은가 — 정책 후보 분석

[`scripts/kiis_k4_ceiling.py`](../../scripts/kiis_k4_ceiling.py)로 모든 episode를 root에서 다시
재생했다 (모델 호출 없음). root는 runner와 같은 RNG 순서로 다시 만들고, 2,304 + 192 episode의
**모든 step에서 기록된 유효 여부가 재생 결과와 일치**했다.
결과: [`ceiling.json`](ceiling.json).

"유용한 행동" = 지금 실행 가능하고 해답 경로에 속하는 행동 (상자 열기, 열쇠 집기, 문 열기·잠금 해제,
목표 물체 집기, 동쪽 이동, 목표 없이 동쪽 방에 있으면 서쪽 이동). step 제한으로 끝난 episode를
마지막 10 step 기준으로 나눴다.

- **후보 부족**: 정책의 후보 계획 중 유용한 행동으로 시작하는 것이 하나도 없음
- **계획기**: 유용한 첫 행동이 후보에 있었지만 고르지 않음
- 함정(목표 물체를 먹음): 전 arm 0건

| split | arm | n | 성공 | 후보 부족 | 계획기 | 유용 후보가 있던 step 비율 |
|---|---|---:|---:|---:|---:|---:|
| dev (K1) | oracle | 48 | 43 | **0** | 5 | 83.4% |
| dev (K1) | A_jev | 48 | 44 | 0 | 4 | 83.5% |
| test | oracle | 288 | 169 | **91** | 28 | 23.3% |
| test | A_jev | 288 | 169 | 88 | 31 | 23.5% |
| test | B_typed | 288 | 169 | 92 | 27 | 23.2% |
| test | D_gen | 288 | 169 | 91 | 28 | 23.3% |
| test | validity | 288 | 153 | 97 | 38 | 25.2% |
| test | B0_typed | 288 | 137 | 93 | 58 | 34.7% |
| test | D0_gen | 288 | 118 | 76 | 94 | 38.2% |
| test | C | 288 | 8 | 51 | 229 | 66.0% |

- test에서 oracle 실패 119건 중 **91건(76%)은 정책이 필요한 행동을 후보로 내지 않아서**다.
  dev에서는 이런 실패가 0건이었다. 어떤 세계모델도 후보에 없는 행동은 고를 수 없다.
- oracle이 한 번도 성공하지 못한 world: te000 (clay urn), te004 (clay urn), te013 (plastic bin).
  예: te000에서 열쇠가 닫힌 clay urn 안에 있는데 정책이 `open clay urn`을 제안하지 않는다.
  같은 clay urn인 te009는 12/12 성공이라 이름만으로 설명되지는 않는다 (root 상태도 관여).
- dev 상자 이름은 wooden box, iron chest, tin crate, glass case였다. test의 이름(clay urn,
  plastic bin, carved bureau 등)은 정책(학습하지 않은 Qwen3-4B)에게 "여는 용기"로 덜 전형적일
  수 있다 — 가설이며, KIIS에서는 검증하지 않는다.
- 이 분석은 표 1의 상한을 설명하는 기술적 분석이다. 이 결과를 보고 정책·계획기를 고쳐
  test에서 다시 돌리면 규칙 3(test 결과를 보고 고치지 않는다)에 걸린다.

## 재현

```bash
python3 scripts/analyze_arms.py --sources artifacts/kiis_k4/20260921/core=20260921,...   # 12개 작업
.venv/bin/python scripts/kiis_k4_ceiling.py --runs artifacts/kiis_k4/*/*/ artifacts/kiis_k1/{C,validity,oracle,A_jev} --out artifacts/kiis_k4/ceiling.json
```

`episodes.jsonl`·`steps.jsonl`은 `.gitignore` 규칙(`*.jsonl`)으로 Git에서 제외된다.
서버 `~/jev-wm/artifacts/kiis_k4/`와 로컬 사본(2026-09-26 동기화)에 있다.

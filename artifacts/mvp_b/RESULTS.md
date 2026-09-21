# MVP-A / MVP-B 결과 (2026-09-21)

환경: TextWorld 1.7.0 symbolic JSON backend / 모델: `jev-1.13.0` (pinned)
규모: 8 worlds × 4 roots = 32 roots, 6 sequences × h∈{1,2} = 384 prefixes, 5 questions
API: 448 requests, 1,119,380 input tokens, $0.047, 실패 0건

## Phase 0
- API 접근: PASS (auth, pinned model 반환, schema 검증)
- TextWorld smoke test: 11/11 PASS
  - no-quest physics: reset 후 미종료, 행동 후 상태 변화, 파서 실패 no-op
  - clone isolation: branch 실행 후 원본 해시 불변, replay 결정론
  - whitelist: 금지 키 0건 (원시 GameState에는 12개 실존 — 테스트 유효)

## MVP-A
- target support 위반 0, state 충돌 0, 대조 pair 844
- 라벨러 독립 교차검증 불일치 0/384
- 설계 의도 확인: 상자가 닫힌 root에서만 순서가 결과를 바꿈 (18:1 vs 0:13)

## MVP-B
| 게이트 | 결과 |
|---|---|
| 0-step 입력 가독성 | **100.0%** (160/160) |
| changed-fact vs persistence | h1 +100%p, h2 +96.6%p |
| 반사실 pair-exact | 94.0% (n=844) |
| 행동 제거 대조 순반응성 | +87.4% |

### 핵심 발견 — precondition blindness

집계 지표가 가린 구조적 한계. JEV 정확도를 **엔진이 거부한 명령 수**로 층화하면:

| h | 거부된 명령 | n | 정확도 |
|---:|---:|---:|---:|
| 1 | 0 | 525 | **100.0%** |
| 1 | 1 | 435 | 91.3% |
| 2 | 0 | 320 | **100.0%** |
| 2 | 1 | 410 | 90.2% |
| 2 | 2 | 230 | 82.6% |

변화한 사실만 봐도 거부 0개에서 h1 74/74, h2 66/66 = 100%.

순서 교환 pair(`open→take` vs `take→open`, q_key_parent, h=2)는 **0/18**.
`take→open`에서 JEV는 32번 전부 `inventory`로 예측 — 상자가 닫혀 있어
take가 실패하는 19건을 모두 틀림. p(정답)은 0.13–0.33으로 신호는 있으나
argmax가 일관되게 틀림.

**해석:** JEV는 행동 조건부 미래 예측을 수행하며, 행동열이 정상 실행되면
완벽하다. 오류는 거의 전부 precondition 위반에 집중된다. 즉 명령이 항상
성공한다고 가정한다.

## 설계에 대한 함의
1. `설계.md` §10.1의 `invalid_attempt_count`가 JEV의 최약점 → 부차 target이
   아니라 headline 측정으로 승격.
2. §11 utility의 `−λ·E[N_invalid(u)]/h` 항이 낙관 편향 → planner가 무효
   행동 포함 plan을 과선택할 것.
3. AB3(full facts vs validity-only)이 핵심 ablation이 되며, 예측 방향이 반대:
   "validity만으로 이득이 설명된다"가 아니라 "validity가 JEV의 약점"이다.
4. B(환경 정답 학습)는 precondition을 학습할 수 있으므로 **B > A on
   invalid-containing prefixes**가 사전등록 가능한 날카로운 예측이 된다.

## 한계
- world template 1종(2방·상자·열쇠·문), 8 worlds. 100%는 템플릿 단순성을
  반영할 수 있음.
- null pair 예측 일치 88.4% → 불필요한 예측 변화 11.6% 존재.
- h=4 미검증. MVP-C(oracle headroom)는 미실행.

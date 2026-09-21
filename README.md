# JEV Typed Counterfactual World Model

JEV의 typed 판단 출력이 **행동 조건부 미래 사실 예측기**로 쓰일 수 있는지,
그리고 그 예측이 실제 행동 개선으로 이어지는지를 통제 비교하는 연구.

- 연구 설계: [`아이디어.md`](아이디어.md) · 구현 설계: [`설계.md`](설계.md)
- 실행 프로토콜(확증/탐색 구분): [`docs/protocol.md`](docs/protocol.md)
- **주장 대장 — 논문 쓸 때 여기서 시작**: [`docs/claims.md`](docs/claims.md)
- 한계(내부 기록, 직역 금지): [`docs/limitations.md`](docs/limitations.md)
- 결과 요약: [`artifacts/mvp_b/RESULTS.md`](artifacts/mvp_b/RESULTS.md)

## 진행 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| Phase 0 API 접근 | 완료 | `jev-1.13.0` 고정, schema 검증 |
| Phase 0 TextWorld smoke | 완료 | 11/11 |
| MVP-A 라벨 신뢰성 | 완료 | support 위반 0, 충돌 0, 교차검증 0 불일치 |
| MVP-B JEV 예측력 | 완료 | 게이트 4/4 통과 + precondition 한계 발견 |
| MVP-C oracle headroom | **미실행** | GPU 필요 (Qwen3-4B policy) |
| Phase 1+ 본 실험 | 미실행 | |

## 핵심 결과 (MVP-B)

행동열이 정상 실행되면 JEV의 endpoint 예측은 **완벽**하다(거부 명령 0개에서
h=1 525건, h=2 320건 모두 100%). 오류는 거의 전부 **precondition 위반**에
집중된다 — 명령이 항상 성공한다고 가정한다.

| h | 엔진이 거부한 명령 | n | 정확도 |
|---:|---:|---:|---:|
| 1 | 0 | 525 | 100.0% |
| 1 | 1 | 435 | 91.3% |
| 2 | 0 | 320 | 100.0% |
| 2 | 1 | 410 | 90.2% |
| 2 | 2 | 230 | 82.6% |

사전 등록된 순서 교환 검사(`open→take` vs `take→open`)는 **0/18** 실패.

## 구조

```
src/
  config.py            API key 해석 (env -> .env -> Keychain)
  jev_client.py        JEV 호출·검증·원시응답 보관
  forecast.py          요청 페이로드 빌더 (설계.md §7)
  env/
    serialize.py       canonical state + 온라인 정보 whitelist (§3.5, §5)
    worlds.py          world 생성 · query catalog · 결정론적 labeler
scripts/
  phase0_api_check.py  API 접근·모델 고정·schema
  phase0_smoke.py      no-quest physics / clone isolation / whitelist
  mvp_a_dataset.py     라벨 데이터셋 생성 + 게이트
  mvp_a_audit.py       라벨러 독립 교차검증
  mvp_b_jev.py         JEV 4조건 실행 (h0/h1/h2/noact)
  mvp_b_analyze.py     게이트 판정
  mvp_b_precondition.py  거부 명령 수별 층화 (탐색적)
artifacts/
  run_manifest.json    seed·설정·해시
  jev_raw/             원시 요청/응답 448건 (공개 범위 미확인)
```

## 재현

`docs/protocol.md` §E 참조. API 키는 `TYPESAFE_API_KEY`로 읽는다.

## 주의

- `artifacts/jev_raw/`는 `.gitignore`에 있다. MCA 공개 범위 확인 전까지
  커밋·공개하지 않는다.
- 본 결과는 intrinsic 예측 능력만 다룬다. **Agent 성능에 대한 주장은 아직 없다.**

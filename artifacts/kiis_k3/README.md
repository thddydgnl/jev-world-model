# K3 — 본 학습과 최종 검증

2026-09-24. 기준 계획: `kiis2026f/실험계획.md` K3.
학습 revision: `ca988eacc7be7f708f95115149da92079a9247c0`.

## 학습 결과

B와 D 모두 동일한 train 전이 6,000개, seed 0, 2 epoch, LoRA r=16/α=32로 학습했다.
둘 다 val 200개의 전이 정확도로 epoch 2가 선택됐다.

| 모델 | 학습 GPU-h | epoch 1 val 정확도 | epoch 2 val 정확도 | 종료 (KST) |
|---|---:|---:|---:|---|
| B_typed | 13.936 | 99.5% | 100% | 2026-09-24 15:40 |
| D_gen | 4.548 | 98.5% | 100% | 2026-09-24 06:16 |

합계 18.484 GPU-h. 이는 학습 비용이며 후속 검증 비용은 포함하지 않는다.
검증 정확도는 1-step 전이 정확도이며 test closed-loop 성공률을 뜻하지 않는다.

## 동기화

서버 `kiis-mvf-gpu:~/jev-wm/artifacts/kiis_k3/`에서 최종·epoch별 어댑터,
설정, 학습 메타데이터를 로컬에 가져왔다. `sync_manifest.json`의 20개 파일 모두
크기와 SHA-256이 일치한다. 학습 로그는 `kiis_k3_B_typed.log`, `kiis_k3_D_gen.log`다.
큰 가중치와 episode JSONL은 기존 `.gitignore` 규칙에 따라 Git에서 제외된다.

## 최종 검증

동일한 val 200개에서 두 모델 모두 정확도 개선, 검증 손실 감소,
선택된 가중치 일치, 학습 당시 평가 결과 재현을 통과했다.

| 방식 | 학습 전 정확도 | 학습 후 정확도 | 파싱 실패율 (전 → 후) |
|---|---:|---:|---:|
| 타입화 B0 → B | 52.0% (104/200) | 100% (200/200) | 0% → 0% |
| 생성형 D0 → D | 40.0% (80/200) | 100% (200/200) | 30.5% → 0% |

| 방식 | 기본 모델 val CE | epoch 1 val CE | epoch 2 val CE |
|---|---:|---:|---:|
| 타입화 | 3.764451 | 0.001617410 | 0.000709117 |
| 생성형 | 0.234234 | 0.000555529 | 0.000082602 |

최종 어댑터 가중치 SHA-256 앞 16자리: B `b8b322c8ba8eb8d7`, D `16ee9b707f6b3f58`.
**최종 판정: K3 통과.** 서버와 로컬에서 `scripts/kiis_check_k3.py`의 **14개 게이트가 모두 통과**했다.
종합 판정과 비교한 후보 목록은 [`validation/gates.json`](validation/gates.json)에 있다.

dev t000/r0, 정책 seed 20260921, step cap 30으로 새로 실행한 결과:

| arm | 결과 | steps | 무효 행동 수 | 파싱 실패 수 |
|---|---|---:|---:|---:|
| B0_typed | 성공 | 14 | 8 | 0 |
| B_typed | 성공 | 20 | 14 | 0 |
| D_gen | 성공 | 23 | 8 | 0 |

세 실행의 step 0 후보 전체, world 지문, 정책 seed, 조건 해시 `0fe479090329`가 일치한다.
B·D 실행에 기록된 어댑터 해시도 최종 선택 가중치와 일치한다.
검증 코드·학습/검증 데이터 9개 파일의 서버/로컬 SHA-256도 일치한다.

이 dev 표는 arm당 1 episode의 동작 확인이다. val 100%가 closed-loop 성능을 보장하지 않으며,
이 smoke 결과로 학습 모델의 성능 우위를 주장하지 않는다. test 본 실행(K4)과 다단계 평가(K5)는
아직 실행하지 않았다.

학습 때 선택에 쓴 것과 같은 val 200개를 `kiis_train_wm.load(..., 200, 0)`으로
선택한다. 기본 모델과 최종 어댑터는 기존 추론 backend로 정확도를 비교한다.
추가로 기본 모델·epoch 1·epoch 2의 target-token cross-entropy를 측정한다.
손실은 정답 토큰 수로 가중하며, 타입화 선택지 순서는 검증 예제마다 한 번 고정해
모든 checkpoint에서 동일하게 쓴다. 타입화와 생성형의 손실 절댓값을 서로 비교하지 않는다.
최종 어댑터는 학습 당시 선택을 그대로 사용한다.

## 재현 (저장소 루트, GPU 환경)

각 스타일을 별도 GPU에 배정할 수 있다. `CUDA_VISIBLE_DEVICES`와 `HF_HOME`은
환경에 맞게 설정한다. 다음 명령은 학습이나 JEV API 호출을 하지 않는다.

```bash
.venv/bin/python -u scripts/kiis_validate_wm.py --adapter artifacts/kiis_k3/B_typed --out artifacts/kiis_k3/validation/B_typed.json
.venv/bin/python -u scripts/kiis_validate_wm.py --adapter artifacts/kiis_k3/D_gen --out artifacts/kiis_k3/validation/D_gen.json
.venv/bin/python -u scripts/mvp_c_headroom.py --trap --split dev --worlds 1 --roots 1 --cap 30 --arms B_typed --adapter artifacts/kiis_k3/B_typed --policy-seed 20260921 --out artifacts/kiis_k3/validation/smoke_B_typed
.venv/bin/python -u scripts/mvp_c_headroom.py --trap --split dev --worlds 1 --roots 1 --cap 30 --arms D_gen --adapter artifacts/kiis_k3/D_gen --policy-seed 20260921 --out artifacts/kiis_k3/validation/smoke_D_gen
.venv/bin/python -u scripts/mvp_c_headroom.py --trap --split dev --worlds 1 --roots 1 --cap 30 --arms B0_typed --policy-seed 20260921 --out artifacts/kiis_k3/validation/smoke_B0_typed
python3 scripts/kiis_check_k3.py
```

마지막 검사는 동기화된 파일 해시, 학습·검증 조건 일치, 검증 게이트,
dev 실행 종료 기록·완주, 공통 조건 해시, world 지문, 정책 seed,
step 0 후보 전체 일치, 사용한 최종 어댑터 해시를 확인한다.

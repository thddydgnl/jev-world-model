# KIIS 2026 추계 — 논문 수치 (K6)

`scripts/kiis_report.py`가 로그에서 이 폴더의 파일을 모두 만든다. **논문의 숫자는 전부 `slots.json`에서 가져오고
손으로 옮겨 적지 않는다** ([실험계획.md](../실험계획.md) K6, [논문구성안.md](../논문구성안.md) §7).

```bash
.venv/bin/python scripts/kiis_report.py          # 다시 만든다 (약 1분)
.venv/bin/python scripts/kiis_report.py --check  # 로그에서 다시 계산해 slots.json과 같은지 확인
```

| 파일 | 내용 |
|---|---|
| `slots.json` | 결과 슬롯 R1–R20 (값, 95% CI, 본문용 문자열), 가설 판정, 판정이 고른 §5 문장, 온라인 초록의 반올림 값 |
| `table1.md`, `table1.csv` | 표 1 (영문, 양식 규칙): 성공률 [CI], 무효율, 세계모델 비용. 아래 주석 한 줄 |
| `fig2.svg`, `fig2.csv` | 그림 2: (a) 평가 world (b) X1의 k = 1..4 상태 완전일치. CSV는 같은 값과 CI (워드·한글에서 다시 그릴 때) |

## 계산 규칙

- 신뢰구간: world 24개 단위 bootstrap 4000회, 같은 world가 두 번 뽑히면 두 번 센다 (`analyze_arms.boot_paired`, 9/30 수정).
  수치마다 난수를 seed 20260921로 따로 시작해, 한 수치가 다른 수치의 계산 순서에 영향을 받지 않는다.
- closed-loop: `artifacts/kiis_k4v3` 15개 단위, 조건 `448b4fac3efe`, arm마다 288 episode. 조건·arm 해시가 섞이면 멈춘다.
- k-step: `artifacts/kiis_k5v3`(평가 world), `artifacts/kiis_x1v3`(X1)의 행 단위 기록. 짝차이는 (rollout, k) 단위로 짝짓는다.
  오류 누적(R18)은 `kiis_k5_rollout.py accumulation`이 같은 행에서 만든 `accumulation_h8.json`을 읽는다.
- 파싱 실패율(R12)은 §6.1 정의 그대로 "재생성 뒤에도 파싱되지 않은 예측 / 전체 예측"이고, 예측 단위로 기록된 한 step
  예측(teacher-forced)에서 잰다. closed-loop 기록에는 예측 수가 없고 요청 수만 있어 범위로만 적는다.
- 구조 겹침(R20): 평가 world 전이의 (상태, 행동)을 이름 → id로 바꿔 B·D 학습 전이 6,000개와 비교한다.
- 로컬에만 있는 입력 (git 제외): `artifacts/kiis_k4v3/*/*/*.jsonl`, `artifacts/kiis_k5v3`·`kiis_x1v3`의 `*.jsonl`,
  `artifacts/kiis_k4`·`kiis_k4v2`의 `episodes.jsonl` (R17), `data/kiis/transitions{,_v3}/*.jsonl` (R20).

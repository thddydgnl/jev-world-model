# 연구 설명서 — LLM 에이전트의 월드 모델: 결정 모델과 범용 LLM의 비교

2쪽 논문([`../paper/`](../paper/README.md))의 배경·설계·방법·결과·해석을 풀어 쓴 12쪽 분량의 설명 문서다.
숫자는 모두 [`../results/slots.json`](../results/README.md)의 v3 값이며, 빌더가 대조한다.

| 파일 | 내용 |
|---|---|
| `LLM_에이전트의_월드모델_설명.docx` | **읽을 문서.** v2 빌드를 바탕으로 2026-10-08에 마무리한 판 (제목을 논문에 맞추고 투고 일정 절을 뺌) |
| `build_report_v2.py` → `LLM_에이전트의_월드모델_v2.docx` | v2 빌더와 그 산출물 (A4, 여백 12.7 mm, Apple SD Gothic Neo). 글을 고칠 때는 이 스크립트를 고쳐 다시 만든다 |
| `build_report.py` → `교수님_검토용_연구설명서.docx` | 첫 판 (2026-10-02) 빌더와 산출물 |

```bash
.venv/bin/python kiis2026f/report/build_report_v2.py   # python-docx, Pillow 필요; 중간 그림은 .qa/ (git 제외)
```

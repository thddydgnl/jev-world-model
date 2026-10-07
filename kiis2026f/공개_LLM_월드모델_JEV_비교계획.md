# 더 어려운 환경에서 공개 LLM 월드모델과 JEV를 비교하는 연구 계획

작성·공개 소스 확인: **2026-09-29**. 상태: **구현 전 계획**. 사용자 요청에 따라 기존 소규모 TextWorld를 새 연구의 주 환경에서 제외하고, **ScienceWorld를 주 환경, ALFWorld를 추가 검증 환경**으로 선택한다. 아래 실험 규모·통과 기준·일정은 제안이며 측정 결과가 아니다. 이번 작업은 코드 조사와 문서 작성이며, 환경 설치·모델 다운로드·유료 추론·새 GPU 실험은 실행하지 않았다.

**1. 권장안: ScienceWorld + Word2World + 동일 LLM의 생성형/타입화 비교**

핵심 질문은 **“어려운 환경에서 행동의 결과를 미리 예측하는 것이 실제 문제 해결을 돕는가, 그 예측을 JEV식 타입화 질의로 바꾸면 정확도와 비용이 어떻게 달라지는가?”**다. 새 이름과 물체만 추가한 기존 환경보다, 측정·가열·연결·실험 순서가 결과를 바꾸는 환경으로 옮긴다.

| 결정 | 구체적 선택 |
|---|---|
| 주 환경 | [ScienceWorld](https://github.com/allenai/ScienceWorld): 과학 실험과 도구 사용, 부분 관측, 여러 행동에 걸친 상태 변화 |
| 공개 LLM 월드모델 | [Word2World](https://github.com/X1AOX1A/Word2World)의 환경별 다음 관측 생성 모델과 평가 코드 |
| 주 비교 | 동일 Qwen의 자유 생성·스키마 제약 생성·타입화 질의, frozen JEV, 공개 Word2World checkpoint |
| 추가 환경 | [ALFWorld](https://github.com/alfworld/alfworld): 일상 과제에서 결과가 재현되는지 확인 |
| 기존 TextWorld | 기존 KIIS 결과와 회귀 점검에 사용. 새 연구의 일반화 근거는 새 환경에서 확보 |
| 후속 환경 | WebArena-Verified. 브라우저 상태 복원·관측 변환 비용을 감당할 수 있을 때 확장 |
| 실험 ID | `sw_wm_v1`과 `alf_wm_v1`. 현재 KIIS의 K4v3·K5v3와 별도 manifest·결과 디렉터리 사용 |

처음부터 모든 공개 방법을 같은 규모로 재현하지 않는다. **ScienceWorld에서 환경 재현 → 예측 평가 → 고정 후보 선택 → 실제 에이전트 평가**를 완성하고, 같은 구현을 ALFWorld로 확장한다. 아래 P0–P5는 이 순서를 실행 가능한 작업으로 나눈 것이다.

**2. 지금까지의 결과에서 가져갈 것과 새 환경에서 해결할 것**

[현재 KIIS 진행 기록](README.md)과 [K5v3 결과](../artifacts/kiis_k5v3/README.md)에 따르면, 기존 파이프라인·학습·다단계 평가가 이미 구현되어 있다. K5v3는 완료됐고 K4v3는 9월 29일 진행 확인 시 실행 중이었다. 이는 조회 시점의 기록이며 이 문서가 실시간 상태판은 아니다.

| 현재 관찰 | 새 계획에 반영할 점 |
|---|---|
| 기존 K4에서 여러 강한 arm의 성공률이 같고 정책 후보 부족이 주요 실패 원인이었음 | 환경 난이도뿐 아니라 **좋은 행동 후보가 존재하는지** 먼저 확인 |
| K5v3에서 학습한 생성형과 타입화가 쉬운 조건의 높은 정확도에 근접 | 정확한 출력 형식 외에 내용·다단계·의사결정 차이가 드러나는 과제 필요 |
| 타입화가 내용 오류의 누적을 더 억제한다는 H8은 현재 구분되지 않음 | JEV 우위를 전제로 실험 설계하지 않고, 차이가 없다는 결론도 허용 |
| 기존 ALFWorld 소규모 probe: 8개 train 게임·24개 상태·794개 admissible action 중 planner 거리 감소 38개(4.8%), 비가역 실패 0개 | ALFWorld라는 이름만으로 충분히 어렵거나 월드모델이 필요하다고 판단하지 않음 |

ALFWorld probe의 근거는 [probe 코드](../scripts/alfworld_probe.py)와 [로그](../artifacts/alf_probe8.log)다. 당시 사용한 planner의 남은 계획 길이를 **보장된 최적 거리**로 해석하지 않는다. 소수 게임의 특정 경로에서 얻은 결과여서 환경 전체의 난이도를 대표하지도 않는다.

**3. ScienceWorld를 먼저 선택하는 이유와 과제 범위**

ScienceWorld는 물질의 상태 변화, 측정, 전기 회로, 마찰 등의 과제를 제공한다. 본 연구에는 성공 여부만 맞히는 짧은 과제보다, 같은 물체에 다른 행동 순서를 적용해 결과 차이를 확인할 수 있는 과제가 적합하다. 이것이 현재 코드와 연구 질문을 고려한 선정 판단이다. [공식 과제 목록과 환경 설명](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/README.md)

| 단계 | 과제 | 측정하려는 어려움 |
|---|---|---|
| 연결 점검 | `boil`, `melt`, `freeze`, `use-thermometer` | 행동·시간·관측·점수 처리와 물리 변수 추출 검증 |
| 주 평가: 열 | `measure-melting-point-known-substance`, `measure-melting-point-unknown-substance` | 가열과 측정 순서, 관측하지 않은 물성에 대한 불확실성 |
| 주 평가: 전기 | `test-conductivity`, `test-conductivity-of-unknown-substances` | 연결 상태와 재료에 따른 결과, 잘못된 실험 구성 |
| 주 평가: 역학 | `inclined-plane-friction-named-surfaces`, `inclined-plane-friction-unnamed-surfaces` | 실험 조건·측정 결과·대상 식별을 여러 단계에 걸쳐 추적 |

위 여섯 과제의 공식 ID를 실행 시 API에서 확인해 manifest에 저장한다. 온도 등을 엔진에서 읽어 정답을 만들 수 있는지, 관측만으로 어떤 정보를 예측할 수 있는지는 P0에서 검증한다. 검증 전부터 특정 실패를 비가역적이라고 부르거나 모든 물리 변수를 추출할 수 있다고 가정하지 않는다.

주 조건의 simplification은 **빈 문자열**로 한다. `easy`, `teleportAction`, 자동 물주기 등은 별도 조건이며 조용히 활성화하지 않는다. 행동 제한은 우선 80회, 가상 rollout 깊이는 1·2·4·8, 실제 계획 깊이는 3으로 제안한다. 80회가 정상적인 해결 경로를 부당하게 자르면 dev에서 모든 arm에 공통으로 조정하고 test 전에 동결한다.

추가 환경 ALFWorld는 여섯 task family와 seen/unseen scene을 층화한다. ALFWorld의 텍스트 실행기는 TextWorld를 이용하지만, 과제·장면·부분 관측 구조는 현재 소규모 자체 환경과 다르다. 따라서 **완전히 다른 엔진**이라는 설명은 하지 않는다. ScienceWorld를 먼저 선택한 이유에는 이 차이도 포함된다. [ALFWorld 공식 구현](https://github.com/alfworld/alfworld)

WebArena-Verified는 실제 웹 상호작용 확장에 적합한 후보지만, 첫 연구에서 동시에 붙이면 브라우저 상태 복원과 DOM 표현까지 비교 변수가 늘어난다. ScienceWorld·ALFWorld 결과를 확보한 뒤 별도 연구 단계로 둔다. [공식 저장소](https://github.com/ServiceNow/webarena-verified)

**4. “일반적인 LLM 에이전트 월드모델”의 기준 구현**

월드모델은 **현재까지의 정보와 가상의 행동을 받아 다음 관측 또는 상태를 예측하는 모듈**로 정의한다. 그 예측을 실제 실행 전에 후보 선택에 사용해야 한다. 다음 행동만 추천하는 ReAct 정책이나 성공 점수만 내는 가치 모델은 전이 월드모델과 구별한다.

Word2World는 행동·관측 이력을 받아 다음 환경 응답을 생성하는 코드와 ScienceWorld/ALFWorld 전용 checkpoint를 공개한다. 따라서 새 환경에서 직접 사용할 생성형 기준으로 RAP의 Blocksworld 예제보다 연결이 가깝다. 다만 원래 시스템은 모델을 가상 환경으로 사용하는 평가도 포함하므로, **여기에 공통 계획기를 붙이는 실험은 우리의 확장 구현**이라고 표시한다. [Word2World 논문](https://arxiv.org/abs/2512.18832), [공식 README](https://github.com/X1AOX1A/Word2World/blob/e8dc240269fb222de3c242b09c70bfbf0b4ac1c9/README.md)

| 재사용 부품 | 실제 용도 | 이번에 추가할 것 |
|---|---|---|
| [WorldModel 및 상호작용 루프](https://github.com/X1AOX1A/Word2World/blob/e8dc240269fb222de3c242b09c70bfbf0b4ac1c9/scripts/interact_with_world_model/run.py) | 이력에 행동을 넣고 다음 관측 생성, 모델의 성공 표식 처리 | 공통 backend 인터페이스와 계획 후보 평가 |
| [한 단계 평가](https://github.com/X1AOX1A/Word2World/blob/e8dc240269fb222de3c242b09c70bfbf0b4ac1c9/scripts/single_step_accuracy/run.py) | 실제 이력 prefix를 사용하는 다음 관측 평가 | 자유 rollout, 의미 단위 평가, 실패 포함 집계 |
| [WM2Real 평가](https://github.com/X1AOX1A/Word2World/blob/e8dc240269fb222de3c242b09c70bfbf0b4ac1c9/scripts/interact_with_world_model/cal_wm2real.py) | 모델 안에서 얻은 행동열을 실제 환경에서 재실행 | 동일 시작점 복원 검증과 허위 성공 분석 |
| [ScienceWorld checkpoint](https://huggingface.co/X1AOX1A/WorldModel-Sciworld-Qwen2.5-7B) | 학습된 공개 생성형 월드모델 | 입력 조건을 명시한 평가 어댑터 |
| [ALFWorld checkpoint](https://huggingface.co/X1AOX1A/WorldModel-Alfworld-Qwen2.5-7B) | 추가 환경의 공개 월드모델 | 동일 지표와 보고 형식 |

공개 checkpoint는 환경 trajectory로 학습한 모델이다. frozen JEV와 비교할 수 있지만, 차이를 **출력 형식만의 효과**로 해석할 수 없다. 카드의 기반 모델은 `Qwen2.5-7B`이며 `Qwen2.5-7B-Instruct`와 같은 가중치라고 취급하지 않는다. [ScienceWorld 모델 카드](https://huggingface.co/X1AOX1A/WorldModel-Sciworld-Qwen2.5-7B)

[RAP / LLM Reasoners](https://github.com/maitrix-org/llm-reasoners)는 `WorldModel`·탐색 인터페이스의 재사용 후보로 남긴다. Blocksworld 재현은 필요하면 연결 점검으로 수행하지만 새 연구의 주 환경은 아니다. [WorldCoder](https://github.com/haotang1995/WorldCoder)의 논문에 ALFWorld가 등장한다고 해서 공개 checkout에 그 어댑터까지 있다고 가정하지 않는다. 조사한 공개 tree에서는 `worldcoder/envs/minigrid_env`가 확인되었다.

**5. 관측 정보가 다른 비교를 막는 두 평가 조건**

Word2World 논문은 ScienceWorld와 ALFWorld 월드모델에 전체 초기 상태 설명을 제공한다고 명시한다. 실제 ScienceWorld 초기화 도구는 여러 방의 관측과 goal progress 등을 모은다. 이런 정보는 일반 에이전트가 처음 받는 관측보다 풍부하다. 해당 코드를 모든 숨은 물리 변수의 완전한 dump라고 가정해서도 안 된다. [논문 §4](https://arxiv.org/html/2512.18832v1), [초기 설명 생성 코드](https://github.com/X1AOX1A/Word2World/blob/e8dc240269fb222de3c242b09c70bfbf0b4ac1c9/scripts/collect_init_context/collect_wm_instruct_sciworld.py)

| 조건 | 입력 | 역할·보고 이름 |
|---|---|---|
| **H: 관측 이력만, 주 조건** | 목표, 실제로 받은 관측·행동 이력, 공개된 행동 문법. 모든 arm에 같은 정보 | 실제 부분 관측 에이전트 비교. 공개 checkpoint는 `W2W-H`라는 **입력 변경 실험**으로 명시 |
| **I: 풍부한 초기 설명, 보조 조건** | 공개 프로토콜의 초기 설명을 같은 내용으로 모든 WM에 제공 | 공개 방식의 기본 동작 확인 및 입력 정보량의 영향. H와 별도 표 |

H에서는 전체 object tree, `info['valid']`의 엔진 계산 행동 목록, 숨은 goal progress, gold action sequence를 정책·WM·평가 함수에 전달하지 않는다. 기본 관측 계약은 목표·행동 응답과, 해당 프로토콜에서 무료 조회로 허용한 현재 `look`/inventory로 고정한다. 모든 arm에 동일한 문자열을 전달하고 입력 hash로 점검한다. 환경 점수·숨은 상태는 별도 평가 프로세스에서만 읽는다.

I에서 closed-loop를 추가한다면 초기 설명을 정책과 세계모델 없는 대조군에도 똑같이 제공한다. 그렇지 않으면 WM의 전이 예측 효과와 추가 정보 효과가 섞인다. 원본 Word2World 재실행에서는 원래 agent/WM의 서로 다른 입력 계약을 보존하되, 그것을 공정한 모듈 교체 결과표에 합치지 않는다.

`W2W-H`는 학습 시의 초기 정보가 빠졌기 때문에 성능이 낮아질 수 있다. 그 하락만으로 “JEV가 학습된 생성형보다 우수하다”고 결론내리지 않는다. **동일 입력 조건의 비교, 원래 초기화 조건의 진단, 동일 LLM 통제 실험**을 함께 제시한다. H에서 알 수 없는 숨은 물성을 한 번의 결정론적 정답 예측만으로 평가하는 한계도 남기며, 관측 가능한 결과 예측과 실제 에이전트 성과를 우선한다.

**6. 비교군: 같은 가중치의 방식 차이와 공개 시스템 차이를 분리**

| ID | 월드모델·모델 | 역할 |
|---|---|---|
| `C_MEM` | 공통 정책 + 관측 이력·실패 기억, 가상 전이 없음 | 월드모델 추가의 실제 이득 |
| `G_FREE` | Qwen3-4B가 다음 관측을 자유 생성 | 기존 연구와 연결되는 일반 생성형. E1/E2 필수, E3는 확장 |
| `G_SCHEMA` | 같은 Qwen3-4B가 공통 관측 frame을 스키마 제약 생성 | 형식 오류를 통제하는 **주 생성형 비교군** |
| `T_QWEN` | 같은 Qwen3-4B가 frame의 질문별 선택지에 답함 | 같은 가중치에서 타입화 질의의 효과 |
| `T_JEV` | frozen JEV에 동일 질문·선택지 제공 | 실제 JEV 시스템 비교 |
| `W2W-H` | 공개 ScienceWorld 7B checkpoint, H 입력 | GitHub에 구현·배포된 학습형 WM의 적용 결과 |
| `VALIDITY` | 엔진의 실행 가능 여부만 후보 선택에 사용 | dev 진단: 무효 행동 회피만으로 이득을 설명할 수 있는가 |
| `ORACLE_WM` | 실제 다음 관측을 공통 frame·가치 평가기에 제공 | dev 진단: 전이가 정확해지면 현재 계획기가 개선되는가 |

정책은 우선 **Qwen2.5-7B-Instruct**, 월드모델 통제군은 **Qwen3-4B**로 분리한다. 정책의 정확한 revision·프롬프트·메모리 방식은 dev에서 동결한다. 정책이 공통적으로 후보를 못 만들면 dev에서 정책을 보강하고 모든 arm에 적용한다. test에서 JEV 결과를 보고 정책을 바꾸지 않는다.

`C_MEM`은 같은 후보 생성기의 1순위 행동을 선택한다. 실패 기억·중복 제거·반복 탈출 규칙은 WM arm과 공유한다. 후보 생성의 sampling 설정과 root별 seed를 저장해 같은 입력에서는 같은 후보가 나오게 한다. 외부 WM arm에만 더 강한 정책이나 별도의 정답 예시를 주지 않는다.

`G_FREE/G_SCHEMA/T_QWEN`은 동일 checkpoint·정밀도·tokenizer·thinking 설정과 같은 dev 전이 4개를 사용한다. 예시는 각 출력 형식에 맞춰 바꾸되 사실 정보는 같게 한다. `T_JEV`에도 같은 예시·문법을 제공한다. 추가 학습은 1차 실험에서 하지 않는다. 공개 W2W의 기존 학습은 이 조건과 별도 표시한다.

주 비교는 단일 예측을 사용한다. 생성형은 greedy, 타입화는 변수별 argmax로 읽는다. JEV의 확률 beam과 생성형 다중 샘플은 계산 예산을 맞춘 후속 실험이다. 현재 `StepPrediction.top`의 변수별 확률은 **전체 상태의 결합분포가 아니다**. 독립적인 최빈값 조합과 생성형의 한 상태 샘플이 같다고 설명하지 않는다.

판정할 차이는 `T_QWEN − G_SCHEMA`(같은 가중치의 예측 프로토콜), `T_JEV − G_SCHEMA`(실제 시스템), `T_JEV − W2W-H`(학습·크기·입력 적합성까지 다른 공개 시스템), 각 arm `− C_MEM`(에이전트 효용)이다. `VALIDITY/ORACLE_WM`은 엔진 정보를 쓰는 진단군으로 표시한다.

**7. 상태 표현과 계획기: 기존 작은 상태를 그대로 확장하지 않는다**

```text
실제 목표·관측·행동 이력
        ↓
공통 정책이 후보 행동열 K개 생성
        ↓
각 WM이 가상 다음 관측/관측 frame을 H단계 예측
        ↓
공통 frame → 고정 가치 평가기 → 후보 선택
        ↓
첫 행동만 실제 환경에서 실행 → 실제 관측으로 이력 갱신
```

주 공통 표현 `ObservationFrame`은 **다음에 관측될 정보**를 담는다. 숨은 전체 세계 상태와 구별한다. 목표·행동에 관련된 사건, 알려진 물체의 가시성·위치 관계, 관측된 물질 상태, 측정값 구간, 연결 결과, 실패 응답, 성공 표식, 새 물체 발견을 포함한다. 표현하지 못하는 정보는 `unrepresented`, 응답에 없는 정보는 `not_reported`, 예측 보류는 `abstain`으로 구분한다. 이 셋을 하나의 “정답 unknown”으로 합치지 않는다.

| 구현 규칙 | 이유 |
|---|---|
| root의 실제 관측에 등장한 entity ID와 공개 문법으로 질문·선택지를 구성 | 닫힌 공간의 물체 목록을 정답 엔진에서 가져오는 누출 방지 |
| 기존 물체는 지속 ID, 새 발견은 공통 discovery 사건으로 기록. 생성형 원문의 새 물체 목록은 별도 보존 | Choice 질의만으로 미지의 물체 이름을 생성할 수 있다는 가정 방지 |
| 온도 등은 train/dev에서 고정한 구간으로 비교. 수치 예측이 가능한 arm의 MAE는 보조 지표 | JEV 범주형 출력과 자유 수치 출력을 동일 정확도라고 부르지 않음 |
| 원문 → frame 변환은 고정 parser를 사용하고, 없는 사실을 물리 규칙으로 채우지 않음 | 파서에 전이 정답을 구현하는 문제 방지 |
| 생성형·타입화 모두 같은 frame 필드와 타입상 허용값을 제공 | 타입화에만 좁은 답 공간을 주지 않음 |
| 실제 성공은 환경 평가기로 판정 | 모델이 쓴 성공 표식은 예측이며 결과가 아님 |

새 entity와 긴 자연어 관측을 타입화로 완전히 표현하기 어려운 점은 **연구 대상의 한계**다. E1 공통 정확도는 비교 가능한 frame 필드에 대해 계산하고, 별도로 정보 포괄률·파서 recall·새 물체 발견 누락률을 공개한다. 이를 전체 상태 완전일치율로 부르지 않는다. 타입화 frame 밖의 유용한 정보를 생성형이 예측하는지도 보조 원문 평가로 남긴다. E3 실제 성과에는 표현 범위의 한계도 포함된다.

재귀 예측에서 `G_FREE/W2W-H`는 자신의 생성 관측을, `G_SCHEMA/T_QWEN/T_JEV`는 공통 결정론적 serializer로 만든 관측을 다음 입력에 넣는다. 이는 의도한 표현 차이이며 완전히 동일한 내부 이력을 가졌다고 주장하지 않는다. 같은 가중치의 `G_SCHEMA/T_QWEN`은 serializer를 공유한다. teacher-forced 평가에서는 모든 arm이 매번 같은 실제 이력을 받는다.

후보는 root의 실제 관측에서 한 번 생성하고 WM 점수를 보기 전에 동결한다. 기본값은 **K=4, H=3**이다. root에서 모르는 물체를 특정하는 행동은 생성하지 않고, 탐색 후 새 물체에 의존하는 계획은 discovery frontier에서 평가를 끝낸 뒤 실제 관측을 받고 재계획한다. 이 제한과 frontier 중단률을 보고한다. 모델 예측을 보고 후보를 바꾸는 방식은 후속 실험으로 둔다.

공통 가치 평가기는 정책과 같은 frozen Qwen을 사용해 목표·실제 이력·예측 frame에서 목표 달성 가능성을 0–1로 평가한다. dev에서 프롬프트를 고정하고 실패 반복·행동 길이 페널티를 같은 방식으로 적용한다. 숨은 목표 세부 항목과 실제 미래 점수를 입력하지 않는다. `ORACLE_WM`도 같은 가치 평가기를 사용하므로 **최적 정책의 상한은 아니다**. 후보의 실제 점수로 고르는 `ORACLE_SCORE`는 오프라인 E2 진단에만 둔다.

파싱 실패 시 해당 가상 가지를 종료하고 실패를 기록한다. 전체 후보가 실패하면 모든 arm에 같은 정책 1순위 fallback을 적용한다. 실제 결과를 이용한 복구는 없다. 주 비교의 생성 재시도는 0회이며, 형식 복구 1회 허용 조건은 추가 비용 실험으로 분리한다.

**7-a. JEV에 넣을 질문의 구체적 설계**

질문은 **“이 행동을 한 번 시도하면 다음 환경 응답에 무엇이 나타나는가?”**로 통일한다. 행동이 좋은지 또는 과제를 해결할지를 바로 판단시키는 것은 공통 가치 평가기의 역할이다. 현재 [step_questions()](../src/wm/recursive_forecaster.py)의 `type: choice`, `instructions`, `criteria` 구조를 재사용하되, 기존 완전 관측 상태 payload를 관측 이력과 가상 행동 payload로 교체한다. 아래는 설계 초안이며 API 호출 결과가 아니다.

Word2World와 비교하는 주 출력의 범위는 **명령 한 번에 대한 다음 텍스트 응답**으로 고정한다. 입력에 실제로 확보한 look/inventory를 넣을 수는 있지만, 예측 정답에 다음 시점의 무료 조회 결과를 자동 추가하지 않는다. 그러한 관측 확장은 모든 WM의 출력 계약을 같이 바꾼 별도 조건이다. 숨은 전체 상태 예측은 별도 실험으로 정의해야 한다.

입력에는 실제 목표·관측·행동 이력, 이미 관측한 entity의 ID/이름 대응, 시도할 명령 하나, `horizon=1`, 환경의 시간 진행 규약을 넣는다. 두 번째 가상 step부터는 앞서 예측한 응답을 이어 붙이고 실제 관측과 예측의 출처를 구별한다. 실제 다음 응답·숨은 물성·정답 행동열은 넣지 않는다.

| 질문 ID 예시 | 질문 내용 | 선택지 설계 초안 |
|---|---|---|
| `command_response` | 다음 응답은 이 명령의 처리를 어떻게 보고하는가? | 처리됨 / 거부됨 / 대상 재지정 요청 / 기타 응답 / 판단 보류 |
| `temperature_report(sample_1)` | 다음 응답에 sample_1의 온도가 보고된다면 어느 구간인가? | T<0 / 0≤T<25 / 25≤T<50 / 50≤T<100 / T≥100 / 온도 미보고 / 판단 보류 |
| `phase_report(sample_1)` | 다음 응답은 sample_1의 물질 상태를 어떻게 기술하는가? | 고체 / 액체 / 기체 / 혼합 상태 / 기타 명시 상태 / 상태 미보고 / 판단 보류 |
| `light_report(bulb_1)` | 다음 응답은 bulb_1의 점등 상태를 어떻게 기술하는가? | 켜짐 / 꺼짐 / 기타 명시 상태 / 점등 상태 미보고 / 판단 보류 |
| `location_report(object_1)` | 다음 응답에 명시되는 object_1의 직접 위치는 어디인가? | 관측된 장소·용기 ID / 인벤토리 / 기타 명시 위치 / 위치 미보고 / 판단 보류 |
| `new_entity_mentioned` | 다음 응답에 기존 entity 목록에 없는 물체가 이름과 함께 등장하는가? | 등장함 / 등장하지 않음 / 판단 보류 |
| `completion_reported` | 다음 응답에 과제 완료가 명시되는가? | 명시됨 / 명시되지 않음 / 판단 보류 |

온도 구간은 설명용 초안이며 단위·범위를 dev에서 확인해 동결한다. 각 선택지는 상호 배타적으로 정의하고, 여러 수치·물체가 등장할 때의 지칭 및 관측 순서 규칙도 명시한다. task에 필요한 정밀도보다 구간이 거칠면 모든 비교군의 동일 필드를 함께 세분화한다. 구간 정확도를 정확한 온도 예측이라고 부르지 않는다.

예를 들어 **이전에 sample_1의 온도가 40°C라고 관측했고 가열 장치가 켜져 있는 이력**에서, 후보 행동이 **온도계로 sample_1을 측정하는 것**이면 다음 질문을 만든다. 40°C는 이전 측정값이며 현재 온도를 정답으로 알려 주는 값이 아니다. 아래 `candidate_action` 자리에는 P0에서 검증한 실제 환경 명령 문자열을 넣는다.

```json
{
  "temperature_report_sample_1": {
    "type": "choice",
    "instructions": "Using only the supplied observation/action history, predict the response to one attempt of candidate_action. Which temperature interval will that response report for sample_1? Predict the future response, not the last recorded reading. Use not_reported only if the response will contain no temperature for sample_1; use abstain if you cannot make a prediction. Assume no additional measurement actions.",
    "criteria": {
      "below_0": "The response reports T < 0 degrees Celsius.",
      "from_0_to_25": "The response reports 0 <= T < 25 degrees Celsius.",
      "from_25_to_50": "The response reports 25 <= T < 50 degrees Celsius.",
      "from_50_to_100": "The response reports 50 <= T < 100 degrees Celsius.",
      "at_least_100": "The response reports T >= 100 degrees Celsius.",
      "not_reported": "The response reports no temperature for sample_1.",
      "abstain": "I abstain from predicting which of the above outcomes will occur."
    }
  }
}
```

이는 기존 `client.ask(state, questions, tag)`의 **questions 인자 예시**이며 전체 HTTP 요청 schema는 아니다. 입력 state·질문·허용값·원문 응답·각 선택지 확률을 함께 저장한다. 확률이 반환된다는 사실만으로 calibration을 보장하지는 않는다.

`not_reported`는 실제 응답의 성질이고 `abstain`은 모델의 예측 보류다. 현재 정보로 온도를 몰라도 환경이 측정값을 보고한다면 `not_reported`는 오답이다. `unrepresented`는 parser/frame이 표현하지 못한 내용을 기록하는 평가 메타데이터이며 “모르면 고르는 정답 선택지”로 제공하지 않는다. 응답에 상태가 없다고 해서 실제 물리 상태가 그대로 유지됐다고 가정하지 않는다. 이전 측정값은 측정 시점을 붙인 이력으로만 남긴다.

질문 생성은 **고정 템플릿 + 실제 관측에서 얻은 entity + task family별 고정 필드 집합**으로 구현한다. 공통 질문에 열/전기/역학 모듈을 붙이고, 이미 추적 중인 관련 변수도 계속 포함한다. 예상되는 정답이나 엔진의 미래 효과를 보고 질문을 빼지 않는다. 질문 수를 줄이는 규칙·질문당 선택지·선택지 순서도 dev에서 동결하고 모든 arm에 공유한다. 기준선 `G_SCHEMA`는 같은 필드를 한 번에 생성하고, `T_QWEN`은 JEV와 같은 질문에 답한다.

실행 가능/응답 종류 질문은 기존 코드처럼 상태 결과 질문과 별도 요청으로 분리하는 것을 기본안으로 삼는다. 다른 질문의 “이미 실행됐다”는 표현이 성공을 전제하지 않도록 모든 문구를 **시도한다**로 통일한다. 명령이 거부될 것으로 예측해도 온도 등 결과 질문을 건너뛰거나 이전 상태를 복사하지 않는다. 요청 분리·묶음의 효과는 dev에서 점검하고 비용에 반영한다.

새 물체의 이름은 Choice만으로 생성하지 않는다. 미지 entity가 필요해지는 가상 계획은 발견 지점에서 종료하고, 선택한 행동의 실제 응답에서 이름을 얻은 뒤 다음 계획의 질문에 포함한다. 생성형이 미지 entity를 더 풍부하게 예측할 수 있는 장점과 타입화의 표현 한계를 원문·포괄률·실제 과제 성과로 남긴다. 이런 환경별 질문 설계 자체의 구현 비용도 기록한다.

**8. 환경을 바꿔도 쉬운 문제만 남는 일을 막는 dev 점검**

환경 선정 기준은 JEV 성능이 아니다. 다음 검사는 `C_MEM`, 공통 후보, 엔진 진단을 중심으로 먼저 수행한다. 숫자는 현재 측정값이 아닌 사전 점검 기준이다.

| 점검 | 제안 기준과 조치 |
|---|---|
| 정책의 해결 능력 | dev 24개에서 `C_MEM` 성공률이 80% 이상이면 ceiling 가능성을 기록. 10% 미만이고 oracle도 낮으면 후보·관측·정책부터 점검 |
| 후보 내 선택 여지 | dev 고정 후보 상태 중 30% 이상에서 후보별 실제 효용이 달라지는지 확인. 모두 동점인 상태를 본 결과에서 삭제하지 않음 |
| 전이 예측의 기여 가능성 | `ORACLE_WM`이 `C_MEM`보다 평균 최종 점수 10점 이상 개선되는지 점검. 개선이 없으면 공통 가치 평가기·계획 깊이를 먼저 수정 |
| 단순 무효 행동 회피와의 구별 | `VALIDITY` 대비 `ORACLE_WM` 차이 확인. 차이가 작으면 결과 예측보다 유효성 판단 과제임을 인정 |
| 상태 변화의 비중 | 실제 변화·유효한 무변화·실패 응답을 분리. 전체 정확도와 변화 구간 정확도를 모두 보고 |
| 다단계 의존 | 짧은 과제만 남았는지, 측정 전/후·연결 전/후·가열 시간에 따라 후보 결과가 다른지 확인 |

실패 시 dev에서만 H=3→6, 정책 후보 개선, 과제군 확장을 정해진 순서로 검토한다. test를 보고 쉬운 episode를 제거하거나 JEV가 이긴 과제만 채택하지 않는다. 어떤 과제군을 추가·제외했는지와 이유를 동결 문서에 남긴다. ALFWorld도 같은 검사를 통과해야 별도 일반화 근거로 사용한다.

**9. 데이터 분할·실험 수량·평가 절차**

ScienceWorld의 공식 train/dev/test variation API를 사용한다. 과제별 단순 정수 순번이나 AgentGym의 평탄화 ID를 공식 split과 동일하다고 가정하지 않는다. `task_name + variation_id + 초기 상태 hash + env revision`으로 매핑한다. W2W 공개 학습 데이터와 중복 여부도 확인하고, 확인할 수 없으면 “학습 데이터에 없던 환경”이라는 표현을 쓰지 않는다.

| 단계 | 기본 규모 | 용도 |
|---|---|---|
| 개발 | 공식 train에서 24 variation | 환경·frame·parser 구현, 예시 선택 |
| dev | 공식 dev에서 24 variation | 난이도·후보·처리량·공통 정책 확정 |
| 본 평가 1차 | 여섯 주 과제에서 공식 test 최대 10개씩, `N = Σ min(10, 해당 과제의 사용 가능한 test 수)` | 목표 N≤60. 이미 사용한 variation·공개 train 중복 처리 규칙 사전 고정 |
| 정밀도 확장 | test 최대 20개씩, 총 N≤120 | 최초 test 결과를 보기 전에 비용·검정력으로 확정한 경우만 confirmatory 확장 |
| ALFWorld | 여섯 family의 seen/unseen에 최대 5개씩, N≤60 | ScienceWorld 조건 확정 뒤 외부 환경 검증 |

부족한 test 수를 train으로 보충하지 않는다. 가족별 표본 수와 분모를 그대로 보고한다. ALFWorld는 scene을 묶음으로 취급하고 같은 장면의 변형을 독립 표본처럼 세지 않는다. 별도 이름의 미지 물질 과제가 있다고 해서 전체 모델에 대해 OOD라고 단정하지도 않는다.

| 평가 | 수량·절차 | 주 판단 |
|---|---|---|
| P0 공개 방식 동작 확인 | 공개 저장 trajectory 200전이, I 초기화, 원래 prompt·action protocol | 다음 관측 재생, 모델 로딩·토큰·성공 처리·버전 호환 확인 |
| E1 예측 | N variation × 시작 prefix 3개 × 행동열 2개 = 6N, 최대 360 rollout/WM. k=1·2·4·8 | 자유 rollout 대 teacher-forced. 변화·무변화·실패 층별 정확도 |
| E2 고정 후보 선택 | N × 시작 상태 2개 = 2N, 최대 120상태. K=4, H=3 | 같은 상태·후보에서 실제 효용을 얼마나 좋은 순서로 고르는가 |
| E3 실제 에이전트 | N × 정책 seed 3 × 5 arm = 15N, 최대 900 episode | `C_MEM/G_SCHEMA/T_QWEN/T_JEV/W2W-H`의 실제 성공·최종 점수·비용 |
| E4 입력 정보 효과 | 사전 hash로 뽑은 test 최대 20 variation의 E1을 I 조건으로 모든 WM에 실행 | H와 I 차이. 공개 checkpoint만 추가 정보를 받는 비교 금지 |
| 추가 분석 | E3의 H=1·6, `G_FREE` 추가, 확률/샘플 기반 계획, ALFWorld | 본 결과와 별도 표·예산 |

E1의 두 행동열은 공통 정책 행동열과 문법상 가능한 교란 행동열로 구성한다. 교란은 과제 관련 객체·순서·도구를 바꾸되 엔진 정답으로 좋은 행동만 고르지 않는다. 엔진으로 전이를 수집한 뒤 평가 층을 라벨링하고, 원래 빈도로 집계한 값과 층별 macro 평균을 함께 보고한다. 실제 prefix는 특정 WM의 성공 경로만 사용하지 않고 공통 정책의 성공·실패 이력에서 시간 구간별로 추출한다.

k=8은 한 번의 rollout에서 중간 지점을 평가한다. 실제 성공·종료 뒤의 반복 패딩으로 정확도를 올리지 않는다. 종료 예측 자체와 종료 전 구간 지표를 분리하고 horizon별 유효 분모를 보고한다. teacher-forced와 자유 rollout은 동일 행동열을 사용하며, 잘못 예측한 뒤 실제 상태로 되돌려 주는 것은 teacher-forced에만 허용한다.

E2는 reset+실제 prefix replay로 동일 시작점을 복원한 뒤 각 후보를 실제 실행한다. 시작 상태 hash가 다르면 그 평가를 중단한다. 온라인 정책/WM이 이 branch 실행 결과를 열람하면 안 된다. 무료 snapshot/clone API가 있다고 가정하지 않는다.

P0에서는 관측·무효 행동·대기 명령 각각이 시간을 진행시키는지 확인한다. ScienceWorld에서는 가열 등 다른 과정이 진행될 수 있으므로, 기존 환경의 “행동 실패면 이전 상태 복사” 처리를 그대로 적용하지 않는다. 추가 `look` 호출도 실제 action과 무료 조회 API를 구별하고 reset/replay 때 동일하게 재현한다.

E3 seed 세 개는 예를 들어 `20260929, 777, 1234`로 동결한다. 이는 정책 생성의 반복이며 학습 seed가 아니다. 서로 다른 arm이 다른 실제 상태를 방문할 수 있으므로, 끝까지 동일 후보를 보았다고 쓰지 않는다. 정확히 같은 후보 비교는 E2에서 보장한다.

**10. 지표·통계·결론 기준**

| 영역 | 주 지표 | 반드시 함께 보고할 것 |
|---|---|---|
| E1 | k=4 공통 관측 frame의 의미 정확도 | k별 곡선, 변화 구간 정확도, parser 오류, abstain, 정보 포괄률 |
| E1 보조 | raw next-observation EM, 실행/실패 예측 F1 | 원문 출력 arm의 동일 protocol에만 EM 적용. 타입화와 전체 문자열 EM 비교 금지 |
| E2 | 후보 집합 내 decision regret | 후보가 모두 동점인 상태 비율, 선택 성공률, 허위 성공 예측 |
| E3 | ScienceWorld 최종 공식 점수 평균(0–100 척도) | 성공률, task별 점수, 실패 종료·시간 초과·반복·무효 행동 |
| ALFWorld | 실제 task success rate | family·seen/unseen별 결과 |
| 비용 | 성능 대비 총 추론 비용과 지연 | input/output token, GPU초, API 비용, p50/p95, peak VRAM, 캐시 hit |

ScienceWorld의 `done`은 성공뿐 아니라 실패·시간 제한도 포함할 수 있다. 성공은 해당 고정 버전의 공식 점수/성공 조건으로 판정하고 `done=True`를 성공으로 세지 않는다. 음수 실패 점수가 있으면 원점수와 0으로 clipping한 보고 점수를 모두 저장하고, 주 평균은 clipping 규칙을 사전 명시한다. [Python 환경 구현](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/scienceworld/scienceworld.py)

E2의 실제 효용은 `J_true(u) = (score_after − score_before)/100 − 0.01 × 실행 행동 수`로 정한다. 종료 조건은 실제 환경을 따른다. 최종 실패 여부는 별도 보고한다. 후보 집합 내 regret는 `max_u J_true(u) − J_true(선택한 u)`다. 전역 최적 정책에 대한 regret가 아니다. 점수 진전이 아직 없는 준비 행동의 가치는 H가 짧으면 과소평가될 수 있으므로 H=6 민감도 분석을 둔다.

파서가 성공한 출력만의 정확도를 주 지표로 쓰지 않는다. malformed와 예측 보류는 해당 평가 필드에서 오답으로 포함하고 coverage를 별도 보고한다. 원문에 실제로 언급되지 않은 정보의 `not_reported`는 정상 관측 라벨일 수 있으므로 변경/정보성 필드 지표를 함께 제시한다. 모든 모델이 “언급 없음”만 답해 높아지는 점수를 방지한다. frame 필드별 macro 평균과 frame 전체 일치율을 구분한다.

parser 자체는 dev에서 최소 200개 응답을 원문·arm·성공/실패별로 층화해 사람이 확인한다. gold 엔진 응답에서 정보 포괄률 95%, 모델 원문에서 명시 사실 추출 precision 98%를 초기 목표로 둔다. 부족하면 공통 표현을 수정하고 dev에서 재검사한다. 원문에 없는 의미를 LLM judge로 임의 보충해 주 지표를 만들지 않는다.

통계 단위는 전이 수가 아니라 **variation/scene**이다. 같은 variation의 prefix·seed·arm을 묶어 paired cluster bootstrap 10,000회로 평균 차이와 95% CI를 보고한다. ScienceWorld는 task별 층화 후 여섯 task의 macro 평균을 주 집계로 사용한다. 여섯 과제로 전체 과학 과제의 일반화를 입증했다고 쓰지 않는다.

E3의 일차 비교는 `T_JEV − G_SCHEMA` 최종 점수, 이차 확인 비교는 `T_QWEN − G_SCHEMA`로 사전 등록한다. 같은 결론에 묶는 복수의 확인 검정에는 Holm 보정을 적용한다. 나머지 여러 arm·horizon·하위 집단 결과는 탐색 분석임을 표시한다. 정책 seed를 늘린 것을 독립 환경 수 증가로 취급하지 않는다.

N≤60은 작은 차이까지 검출한다는 보장이 없다. dev의 paired 차이 분산으로 CI 폭과 검정력을 추정하고 test 개봉 전에 N≤120 확대 여부를 결정한다. 유의할 때까지 표본을 추가하지 않는다. JEV가 파싱 오류만 줄이고 내용·선택·실제 점수를 개선하지 못하면 결론도 **형식 안정성의 이득**으로 제한한다.

**11. 구현할 파일과 기존 코드의 재사용 범위**

아래 경로는 repo root 기준의 **제안 파일**이며 아직 구현되지 않았다.

```text
configs/sw_wm_v1.yaml              # arm, 입력 조건, seed, 예산, 동결 hash
src/env_adapters/scienceworld.py   # 관측/평가 정보 분리, reset+replay
src/env_adapters/alfworld.py       # 두 번째 환경, action protocol 변환
src/wm/observation_frame.py        # schema, entity 추적, parser, serializer
src/wm/word2world_backend.py       # 공개 next-observation 모델 어댑터
src/wm/structured_backend.py      # G_SCHEMA, 동일 Qwen T_QWEN 연결
src/agent/wm_lookahead.py          # 공통 후보, 가치 함수, 계획 예산
scripts/sw_wm_probe.py            # P0/P1 환경·난이도 점검
scripts/sw_wm_eval.py             # E1/E2/E3와 재개·집계
artifacts/sw_wm_v1/               # manifest, 예측 원문, 비용, 결과
```

현재 [recursive_forecaster.py](../src/wm/recursive_forecaster.py)의 backend 교체 구조, [llm_backends.py](../src/wm/llm_backends.py)의 Qwen/JEV 연결, [K5 rollout 평가](../scripts/kiis_k5_rollout.py)의 저장·재개·teacher-forced 개념을 재사용한다. 기존 `TaskSpec/state_schema`·효용·물체 변수는 새 환경에 맞게 교체한다. 기존 학습 adapter를 그대로 ScienceWorld용으로 취급하지 않는다.

현재 Qwen 선택지 표시는 `A`–`P` 16개에 한정된다. 새 환경의 위치·물체 수에 그대로 적용하면 부족할 수 있다. 공통 frame을 작은 유한 선택지와 entity별 질문으로 나누고, 더 긴 선택지가 필요한 경우 token 단위가 아닌 전체 선택지 log-likelihood 처리를 구현한다. JEV API의 현재 선택지·길이·batch 제한도 P0에서 확인한다. 편의상 정답 후보를 16개 안으로 좁히지 않는다.

`G_SCHEMA`에는 [XGrammar의 schema constrained decoding](https://github.com/mlc-ai/xgrammar/blob/main/docs/start/quick_start.md)을 우선 검토한다. 문법 제약은 의미 정답을 보장하지 않는다. 생성형에만 엔진 기반 상태 수선이나 물리 제약을 추가하지 않는다. 양쪽에 같은 invariant 검사를 적용하는 조건은 별도 ablation으로 둔다.

**12. 공개 코드와 환경에서 실제로 확인된 재현 문제**

| 확인 사항 | P0에서 할 일 |
|---|---|
| ScienceWorld 1.3.0은 객체 순서와 일부 물리 trajectory·명령 해석이 이전 버전과 달라질 수 있다고 명시 | 주 환경은 조사한 1.3 계열 commit/JAR로 고정. W2W native 재생은 원래 데이터와 호환되는 버전을 별도 확인하고, 같은 숫자로 합치지 않음 |
| W2W ScienceWorld 의존성은 `scienceworld` 버전을 고정하지 않음 | `pip install latest`로 원래 논문을 재현했다고 쓰지 않음. package·JAR·Java·prompt·data hash 기록 |
| 공개 초기 설명 수집기는 방 탐색에 `teleportAction`을 사용 | 실제 실행 환경의 simplification과 동일하다고 가정하지 않음. 저장된 초기 context와 실행 환경의 일치 점검 |
| W2W ALFWorld는 `put→move`, `help` 추가 protocol | 공식 ALFWorld와 W2W용 데이터/문법을 구분. 변환은 명시적 adapter로만 수행 |
| 공개 데이터 다운로드 코드에 기존 `~/.cache/alfworld` 제거 동작이 있음 | 스크립트를 그대로 실행하지 않고 격리된 data root로 다운로드 경로를 변경 |
| 공개 WM은 다음 관측에서 성공을 스스로 예측 | 원래 WM 성공률, WM2Real, 실제 closed-loop 성공률을 각각 보고 |

근거: [ScienceWorld 변경 안내](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/README.md), [W2W 의존성](https://github.com/X1AOX1A/Word2World/blob/e8dc240269fb222de3c242b09c70bfbf0b4ac1c9/AgentGym/agentenv-sciworld/pyproject.toml), [데이터 다운로드 코드](https://github.com/X1AOX1A/Word2World/blob/e8dc240269fb222de3c242b09c70bfbf0b4ac1c9/scripts/download_data/download_data.py).

최신 ScienceWorld에서의 `W2W-H`는 **초기 정보 변경과 환경 버전 이전을 포함할 수 있는 적용 실험**이다. 따라서 공개 모델의 약화를 타입화의 인과 효과로 해석하지 않는다. I 조건의 E4도 같은 최신 환경이면 버전 이전은 남는다. native 호환성까지 확인된 별도 재실행과 구별해 기록한다. 이 점 때문에 동일 Qwen의 `G_SCHEMA/T_QWEN` 통제가 필수다.

| 대상 | 조사한 revision | 확인 범위 |
|---|---|---|
| `allenai/ScienceWorld` | `e8216d6044e8e39be9fcb185e3b2dfb602584b52` | Apache-2.0, README·Python API |
| `alfworld/alfworld` | `aaba6870f86c5be6a08a491f32a50b906227bc3e` | MIT, README·텍스트 환경 구현 |
| `X1AOX1A/Word2World` | `e8dc240269fb222de3c242b09c70bfbf0b4ac1c9` | 초기화·전이·평가·데이터 코드. 루트 라이선스 미확인 |
| ScienceWorld W2W 7B | `6196b3531c5634a8229af6b2cd111e02a70c13b5` | Hugging Face model revision, 카드의 license는 `other` |
| ALFWorld W2W 7B | `8d4f069c4bf5a739763664897b26ca17c161f0da` | Hugging Face model revision, 카드의 license는 `other` |
| `maitrix-org/llm-reasoners` | `f94e5ac2cb9788c3d7d7dbf2173884ed4088e4b2` | Apache-2.0, backend/탐색 인터페이스 |

코드·가중치·데이터의 재사용 조건은 각각 기록한다. Word2World 하위 구성 요소의 라이선스가 루트 전체에 자동으로 적용된다고 가정하지 않는다. 재사용 가능한 범위가 확인되지 않으면 해당 코드의 직접 편입을 보류하고 문서화된 입출력에 맞춘 독립 adapter와 다른 대조군 구현을 진행한다. 공개 checkpoint를 사용할 수 없을 때는 “Word2World 재현 완료”라고 쓰지 않고 외부 기준선 미완료 항목으로 남긴다.

**13. 단계별 작업·완료 조건·일정**

| 단계 | 작업과 산출물 | 완료 기준 | 예상 작업량 |
|---|---|---|---|
| P0 재현 계약 | 환경/JAR·데이터·모델 lock, 24개 환경 probe, 원본 200전이 재생, 정보 누출 점검 | 같은 prefix를 3회 reset/replay했을 때 관측·점수·평가 상태 일치. W2W protocol 차이 목록 확보 | 2–3 작업일 |
| P1 표현·난이도 | frame/parser·entity 추적, 공통 정책과 후보, C/validity/oracle 진단 | §8 점검, dev 200응답 수작업 확인, 알려진 오류 해결 | 3–4 작업일 |
| P2 WM 연결 | G_FREE/G_SCHEMA/T_QWEN/JEV/W2W-H, 원문·비용 로깅 | 동일 입력/후보 검사, 각 arm smoke, OOM/길이 제한 확인 | 3–4 작업일 |
| P3 동결 | 데이터 manifest·프롬프트·질문·공통 가치 함수·통계·예산 확정 | test 미개봉 상태에서 N·arm·seed·중단 규칙 고정 | 1 작업일 |
| P4 ScienceWorld 평가 | E1→E2→E3→E4, 실패 사례와 CI·비용 표 | 실행 수·중복·config hash 검증, 실패 포함 결과표 | 준비·분석 2–3일 + 실측 계산 시간 |
| P5 ALFWorld 검증 | adapter·protocol 점검 후 동일 비교 | family·scene 분리 결과, ScienceWorld와 독립 결론 | 3–5일 + 실측 계산 시간 |

첫 구현 묶음은 **ScienceWorld adapter + 공통 입력 계약 + 공개 W2W 200전이 재생 + dev 난이도 보고서**다. 이 단계에서 월드모델이 도움이 될 여지가 있는지 확인하고 나서 본 실험 비용을 쓴다.

기존 K4v3가 사용하는 GPU를 중단하지 않는다. 문서·CPU 준비를 먼저 수행하고, 새 GPU 작업은 해당 실행 종료와 자원 확보 후 배치한다. [기존 확장 계획](../fullpaper/실험계획.md)의 KIIS 이후 일정과 연결하되, 새 환경 결과를 10월 2일 요약문까지 확보했다고 전제하지 않는다. 기존 v3의 조건·결과는 그대로 보존하고 새 연구와 출처를 구분한다.

**14. 계산 예산과 실험 중단 규칙**

현재 자원 기준은 RTX A5000 24GB 두 장이다. 무거운 프로세스는 GPU당 1개를 기본으로 한다. 7B BF16 가중치만 약 14GB이므로 긴 context·KV cache·batch를 포함해 24GB에 들어가는지 실제 측정해야 한다. 7B 정책과 7B WM을 한 GPU에 동시에 올릴 수 있다고 가정하지 않는다. 정책/가치 평가기 1장, 활성 WM 1장으로 순차 arm 실행을 우선한다.

API 요청 수만 맞추면 타입화의 여러 질문과 생성형의 긴 출력 비용이 달라진다. **같은 후보·깊이에서의 성능**과 **같은 총 추론 시간 상한에서의 성능**을 별도 비교한다. 시간 상한에는 후보 생성, WM, frame 처리, 가치 평가와 재시도를 모두 포함한다. JEV는 모델 버전·endpoint·실제 과금과 요청 처리 시간을 기록한다.

| 블록 | N=60에서의 대략적 상한·의미 |
|---|---|
| E1 | 360 rollout × 8 step × 5 WM = 14,400 가상 전이. teacher-forced 추가 시 최대 28,800 |
| E2 | 120 상태 × 4 후보 × 3 step × 5 WM = 7,200 가상 전이. 정답 환경 branch replay는 별도 CPU 비용 |
| E3 | 60 variation × 3 seed × 4 WM × 80 행동 × 4 후보 × 3 step = **691,200 가상 전이** 상한 |
| 별도 비용 | C_MEM 정책 실행, 가치 평가, 환경 reset/replay, I 조건, native 재생, parser 검증 |

“가상 전이 1개”는 API 요청 1개가 아니다. 타입화의 질문 수·batch 크기·입력 재전송을 별도 곱해야 한다. 예를 들어 E3 평균 전이 처리 0.5초라면 단순 합산 WM 시간만 96시간, 2초라면 384시간이다. 이는 측정값이나 완료 예상 시간이 아니며 정책·가치·JEV 대기 비용도 제외한 계산 예다. 실제 병렬화·캐시·조기 종료에 따라 달라진다.

P2에서 arm별 100개 전이와 5개 dev episode를 실행해 평균 질문 수, p50/p95, GPU 시간, token, 환경 replay 시간을 측정한다. 그 결과로 전체 비용표를 만든 뒤 P3에서 규모를 동결한다. 예산이 부족하면 **test 결과를 보기 전에** N≤60 또는 N≤120, seed 수, 추가 분석을 정한다. 주 모델의 성능을 보고 arm을 삭제하지 않는다. E1/E2만 완료된 단계에서는 실제 에이전트 성능 향상을 주장하지 않는다.

캐시는 model/revision, prompt, schema, 입력 이력, 행동, 환경 protocol, decode 설정을 키에 포함한다. arm 사이에서 예측을 공유하지 않는다. cache hit를 비용 0의 모델 추론으로 포장하지 않고 cold/warm 비용을 구분한다. 모델 오류·timeout·parser 실패와 환경 서버 장애는 다른 코드로 저장한다. 반복 환경 장애·복원 불일치가 생기면 해당 block을 멈춰 원인을 수정하고 조건 변경 영향을 받는 arm을 함께 다시 평가한다.

**15. 최종 산출물과 허용되는 결론**

완료 시 환경/데이터 lock 파일, arm별 프롬프트와 frame schema, 공개 기준선 변경 내역, 실행 원문·비용 로그, 재현 명령, 통계 notebook, 아래 결과표를 남긴다. 현재 빈칸을 추정 결과로 채우지 않는다.

| 결과표 | 내용 |
|---|---|
| 표 A | native 재생·버전·입력 조건·학습 이력·모델 크기·재사용 범위 |
| 표 B | E1 horizon별 의미 정확도·변화 구간·형식 오류·정보 포괄률 |
| 표 C | E2 동일 후보 regret·허위 성공·모든 후보 동점 비율 |
| 표 D | E3 실제 최종 점수·성공률·paired CI·task별 결과 |
| 표 E | 같은 탐색량 및 같은 시간 예산의 성능·비용 |
| 그림 | 자유 rollout 오류 곡선, 정확도와 선택 손실의 관계, 성능–비용 곡선 |

목표 결론은 미리 정해 두지 않는다. JEV가 높은 실제 점수를 얻더라도 모델·학습 차이를 명시하고, 동일 Qwen 비교에서만 질의 방식에 관한 제한된 인과 해석을 한다. ScienceWorld에서만 이득이면 해당 환경 범위로, 형식 오류만 줄면 형식 안정성으로, 공개 W2W가 더 좋으면 학습형 기준선의 우세로 보고한다. **새 환경에서 예측·선택·실제 성과를 연결해 무엇이 개선되는지 확인하는 것**이 이 연구의 완료 기준이다.

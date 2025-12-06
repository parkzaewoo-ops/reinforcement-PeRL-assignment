본 프로젝트는 강화학습(Reinforcement Learning)의 원리를 응용하여 LLM(Large Language Model)이 스스로의 실수를 학습해 점진적으로 성능을 개선하는 **자가 개선 프롬프트 튜닝(Self-Improving Prompt Tuning)** 시스템을 구현합니다.

특히 Text-to-SQL 문제를 대상으로, LLM이 생성한 SQL을 평가하고 이에 대한 자연어 피드백을 기반으로 **시스템 프롬프트(System Prompt)를 ‘자연어 가중치(Natural-Language Weights)’처럼 업데이트**하여 정책(policy)을 강화하는 새로운 방식의 Reinforcement Learning + Prompt Engineering 하이브리드 접근을 실험합니다.

## 1. 프로젝트 목표

- LLM이 **자연어 피드백을 활용해 스스로 학습**하는 자가 개선 구조 구축  
- SQL 생성 오류를 기반으로 **프롬프트를 자동 최적화**하는 시스템 설계  
- LangGraph 기반 **생성 → 평가 → 개선 루프** 구현  
- 강화학습의 정책(policy) 개념을 시스템 프롬프트로 확장하여 프롬프트를 **동적 가중치처럼 업데이트**  
- Text-to-SQL 성능의 점진적 향상 및 최종 일반화 성능 검증

## 2. 핵심 아이디어: 프롬프트를 '자연어 가중치'로

전통적인 머신러닝에서는 숫자 형태의 가중치(weights)를 업데이트하지만,  
본 프로젝트에서는 LLM의 행동 규칙인 **시스템 프롬프트를 학습 대상**으로 삼습니다.

## 3. 강화학습 프레임워크와의 연결

본 시스템은 강화학습의 구성 요소와 정확히 대응됩니다:

| 강화학습 요소 | 본 프로젝트에서의 대응 |
|---------------|-------------------------|
| Agent | SQL 쿼리를 생성하는 LLM (`generate_sql`) |
| Policy | 시스템 프롬프트 (자연어 규칙 집합) |
| Action | SQL 생성 |
| Environment | SQL 평가 LLM (`evaluate_sql`) |
| Reward | -1.0 ~ 1.0 범위의 점수 |
| Feedback | SQL 오류 분석 및 개선 가이드(자연어) |
| Policy Improvement | 프롬프트 업데이트 (`update_prompt`) |

## 4. 전체 워크플로우 (LangGraph 기반)

LangGraph를 활용한 자가 개선 루프는 다음과 같이 작동합니다:

1. **SQL 생성 (Generate SQL)**  
   - 시스템 프롬프트 기반으로 LLM이 SQL 출력

2. **평가 및 보상 계산 (Evaluate SQL)**  
   - SQL의 정확도를 정량 평가  
   - 잘못된 부분을 자연어로 상세 피드백 제공

3. **의사 결정 (Decider)**  
   - 보상 ≥ 1.0 → 성공 처리  
   - 보상 < 1.0 → 실패 처리 후 프롬프트 개선  
   - 실패 3회 이상 → 조기 중단 및 다음 데이터로 이동

4. **프롬프트 업데이트 (Update Prompt)**  
   - 실패 이력(`failure_history`) 기반 규칙 추가  
   - 동일한 오류가 반복되지 않도록 자연어로 정책 강화

5. **루프 반복**  
   - 개선된 정책으로 다시 SQL을 생성하며 점진적 향상

이 “생성 → 평가 → 개선” 루프는 전체 데이터셋에 대해 자동 반복되며,  
최종적으로 성능이 크게 향상된 **최적화된 시스템 프롬프트**가 만들어집니다.

## 5. 실험 설계 및 검증

### ✔ 데이터 분리
- 전체 데이터의 **80%를 학습**, **20%를 테스트**로 사용  
- 테스트 데이터는 학습 중 절대 보지 않음 → **일반화 성능 측정 가능**

### ✔ 학습 곡선(Learning Curve)
- 각 에피소드마다 reward 기록  
- 에이전트의 개선 경향을 시각적으로 확인

### ✔ 최종 테스트 평가
- 평균 보상 점수  
- Text-to-SQL 성공률(success rate)  
- 자동 생성된 최종 시스템 프롬프트 분석  

## 6. 디렉토리 구조

- `/code/` — SQL 생성, 평가, 프롬프트 업데이트 코드
- `/data/` — 데이터셋 및 부가 자료
- `/presentation/` — 실험 분석 및 발표 자료
- `requirements.txt` — 전체 실행 환경 패키지 버전
- `README.md` — 프로젝트 설명 문서

## 7. 주요 코드 구성 요소

| 파일 | 설명 |
|------|-------|
| `generate_sql.py` | SQL 생성 LLM 노드 |
| `evaluate_sql.py` | SQL 정확도 평가 및 보상 산출 |
| `update_prompt.py` | 자연어 피드백 기반 정책 업데이트 |
| `graph.py` | LangGraph 전체 워크플로우 정의 |
| `run.py` | 학습 및 테스트 실행 스크립트 |
| `presentation/` | 결과 시각화 및 발표 자료 |

## 8. 실행 환경 및 라이브러리 버전

- `/Python 3.10+/`
- `/langgraph==0.x/`
- `/langchain==0.x/`
- `/numpy==1.26.4/`
- `/torch==2.2.2/`
- `/matplotlib==3.8.3/`
- `/tqdm==4.66.1/`
- `/tenacity==8.x/`
- `/openai==1.x (또는 호환 API)/`

## 9. 프로젝트 의의

본 프로젝트는 다음의 가능성을 제시합니다.
- 프롬프트를 RL의 정책(policy)처럼 다루는 혁신적 관점
- 자연어 피드백을 활용한 정책 개선(Policy Improvement) 가능성
- LLM이 환경과 상호작용하며 스스로 발전하는 자가 개선(Self-Improving) 에이전트 구현
- Text-to-SQL 문제 해결에 있어 강화학습적 접근의 새로운 방향 제시
- 이는 향후 Auto-Improving Agents, 자율형 개발자 도구, AI 자기 학습 시스템 연구의 중요한 기반이 될 수 있습니다.

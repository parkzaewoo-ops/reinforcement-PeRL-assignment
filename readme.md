📘 README — reinforcement-PeRL-assignment

Prompt-engineering-Enhanced Reinforcement Learning (PeRL) 과제 실습

본 저장소는 PeRL(Prompt-engineering-Enhanced Reinforcement Learning) 실습을 위한 과제 리포지토리입니다.
기본적인 RL 구조 위에 프롬프트 엔지니어링 기법을 적용하여 LLM 기반 에이전트 또는 보상 구조를 개선하는 실험을 목표로 합니다.

본 과제에서는 RL 에이전트가 더 안정적으로 학습하거나 더 높은 보상을 달성할 수 있도록 프롬프트 최적화(prompt engineering) 아이디어를 RL 과정에 접목합니다.
(예: 보상 설명 프롬프트 강화, 행동 선택 reasoning 유도, 정책 학습 중 자연어 피드백 활용 등)

🧩 1. 프로젝트 개요

이 과제의 핵심 목표는 다음과 같습니다:

PeRL 개념 이해: 강화학습(RL)에 프롬프트 엔지니어링을 결합하는 실험적 접근

Baseline RL 에이전트 구현

프롬프트 기반 강화(PeRL) 적용 후 성능 변화 비교

실험 결과를 그래프/표로 정리하고 슬라이드로 보고

프롬프트 엔지니어링의 적용 방식 예시:

행동 정책 선택 시 LLM 기반 힌트(prompt-guided action suggestions)

reward shaping 문구를 LLM이 이해하기 쉬운 형태로 변환

자연어 설명(CoT)을 강화 루프 내부에서 활용

agent feedback(자연어 보상) 실험

이 저장소에 포함된 코드는 주어진 과제 범위 내에서 PeRL 개념을 구현한 단순 예시입니다.

📂 2. 디렉토리 구조
/
├── code/              # RL 및 PeRL 적용 코드
├── data/              # (있다면) 환경 설정, 로그 저장 위치
├── presentation/      # 실험 분석, 성능 비교 PPT 자료
├── requirements.txt   # 패키지 버전 명시
└── README.md          # 설명 문서

⚙️ 3. 사용 환경 및 라이브러리 버전

아래는 실험에 사용된 주요 라이브러리 버전입니다.

Python 3.10+
numpy==1.26.4
torch==2.2.2
gym==0.26.2
matplotlib==3.8.3
tqdm==4.66.1
openai==1.x (LLM 호출 시 사용 가능)


설치 방법:

pip install -r requirements.txt


또는 직접 설치:

pip install numpy torch gym matplotlib tqdm openai

🚀 4. 실행 방법
4.1 기본 RL 실험 실행
python code/main.py

4.2 PeRL 버전 실행

프롬프트 기반 강화 학습을 사용하는 스크립트:

python code/perl_main.py


혹은 특정 옵션을 조정할 수 있습니다:

python code/perl_main.py --episodes 500 --model "gpt-4o-mini"

🧠 5. 알고리즘 개요 (PeRL)
✔ Baseline RL

기본 강화학습 알고리즘으로 환경에서 보상을 최대화하도록 정책을 학습합니다.
예: Policy Gradient, Q-learning 등.

✔ PeRL (Prompt-engineering-Enhanced RL)

PeRL은 다음과 같은 아이디어들을 결합할 수 있습니다:

1) 행동 선택(prompt-guided action selection)

RL agent가 선택한 action을 프롬프트에 넣고 LLM 통해
"더 나은 조치가 있는지" 자연어로 힌트를 받음

예:

"현재 상태는 A입니다. 가능한 행동은 B, C, D입니다. 높은 보상을 얻기 위한 최선의 행동은?"

2) Reward reshaping prompt

보상 정보를 LLM에 재해석시키고 RL에 반영

3) CoT 기반 reasoning 추가

LLM이 행동의 이유를 설명하도록 하여 exploration 효율을 높임

4) Prompt-based evaluation

정량적 reward 외에 자연어 평가 점수를 LLM이 제공하도록 함

이러한 요소들을 RL 루프 내부에 넣어 정책 학습을 풍부하게 만드는 접근법이 PeRL입니다.

📊 6. 실험 내용 요약
✔ 실험 1: Baseline RL

reward curve 점진적 상승

특정 환경에서 일정 성능 도달

✔ 실험 2: PeRL 적용

LLM의 reasoning 지원으로 exploration 개선

reward 상승 속도 개선 가능성

학습 안정성 향상 패턴 관찰됨(환경에 따라 다름)

✔ 비교 분석

episode-reward 그래프 비교

variance 감소 여부

성공률 개선 여부

자세한 수치와 그래프는 presentation/ 폴더의 슬라이드 참고.

📁 7. 파일 설명
폴더/파일	내용
code/main.py	Baseline RL 학습 코드
code/perl_main.py	Prompt-engineering 기반 RL 학습 코드
presentation/	실험 결과 분석 및 과제 제출용 PPT
requirements.txt	설치해야 하는 패키지 버전 목록
⚠️ 8. 한계점

PeRL 방식은 LLM 응답 품질에 크게 의존

LLM 호출 비용 증가 가능

환경 종류에 따라 프롬프트의 효과가 제한적

재현성(reproducibility)이 낮을 수 있음(LLM 응답 stochasticity)

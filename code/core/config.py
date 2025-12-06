"""
Core Configuration Module
=========================
모델 설정, 경로, 환경 변수 등 전역 설정을 관리합니다.
"""

import os
import dspy
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt

# ============================================
# 경로 설정
# ============================================
BASE_DIR = "/data/workspace/sogang/sqlagent"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
SQLITE_PATH = os.path.join(DATA_DIR, "BIRD.sqlite")
SCHEMA_DIR = DATA_DIR

# 디렉토리 생성
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ============================================
# 모델 설정 클래스
# ============================================
class ModelConfig:
    """LLM 모델 설정을 관리하는 클래스"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Next-80B-A3B-Instruct",
        api_base: str = "http://211.47.56.81:7972/v1",
        api_key: str = "token-abc123",
        temperature: float = 0.6,
        timeout: float = 60.0,  # 120 → 60초로 (응답 없으면 빠르게 스킵)
        max_retries: int = 0,   # 1 → 0으로 (재시도 안 함, 바로 다음으로)
        max_tokens: int = 30000
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._model = None
    
    def get_model(self) -> dspy.LM:
        """DSPy LM 인스턴스 반환 (지연 초기화)"""
        if self._model is None:
            self._model = dspy.LM(
                model=f"openai/{self.model_name}",
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=self.temperature,
                request_kwargs={
                    "timeout": self.timeout,
                    "max_retries": self.max_retries,
                    "max_tokens": self.max_tokens
                }
            )
        return self._model
    
    def configure_dspy(self) -> dspy.LM:
        """DSPy 전역 설정에 모델 등록"""
        model = self.get_model()
        dspy.settings.configure(lm=model)
        print(f"✅ 모델 설정 완료: {self.model_name}")
        return model
    
    def __repr__(self) -> str:
        return (
            f"ModelConfig(\n"
            f"  model_name='{self.model_name}',\n"
            f"  api_base='{self.api_base}',\n"
            f"  temperature={self.temperature},\n"
            f"  max_tokens={self.max_tokens}\n"
            f")"
        )


# ============================================
# 최적화 설정 클래스
# ============================================
class OptimizerConfig:
    """최적화 파라미터 설정"""
    
    def __init__(
        self,
        optimizer_type: str = "copro",  # "copro" or "mipro"
        # COPRO 설정
        breadth: int = 10,
        depth: int = 3,
        # MIPROv2 설정
        num_candidates: int = 10,
        max_bootstrapped_demos: int = 3,
        max_labeled_demos: int = 4,
        # 공통 설정
        init_temperature: float = 0.6,
        num_threads: int = 4  # 8 → 4로 줄임 (SQLite 동시 접근 안정성)
    ):
        self.optimizer_type = optimizer_type
        self.breadth = breadth
        self.depth = depth
        self.num_candidates = num_candidates
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.init_temperature = init_temperature
        self.num_threads = num_threads
    
    def to_copro_kwargs(self) -> dict:
        """COPRO용 파라미터 딕셔너리"""
        return {
            "breadth": self.breadth,
            "depth": self.depth,
            "init_temperature": self.init_temperature,
            "num_threads": self.num_threads
        }
    
    def to_mipro_kwargs(self) -> dict:
        """MIPROv2용 파라미터 딕셔너리"""
        return {
            "num_candidates": self.num_candidates,
            "init_temperature": 1.4,  # MIPROv2 권장값
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
            "max_labeled_demos": self.max_labeled_demos,
            "num_threads": self.num_threads
        }
    
    def __repr__(self) -> str:
        return (
            f"OptimizerConfig(\n"
            f"  optimizer_type='{self.optimizer_type}',\n"
            f"  breadth={self.breadth}, depth={self.depth},\n"
            f"  num_candidates={self.num_candidates},\n"
            f"  num_threads={self.num_threads}\n"
            f")"
        )

DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_OPTIMIZER_CONFIG = OptimizerConfig()


def get_default_model() -> dspy.LM:
    """기본 모델 설정으로 DSPy 구성"""
    return DEFAULT_MODEL_CONFIG.configure_dspy()


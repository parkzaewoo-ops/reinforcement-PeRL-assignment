"""
Core Package for Text-to-SQL DSPy Agent
=======================================

이 패키지는 Text-to-SQL 작업을 위한 DSPy 기반 에이전트의 핵심 기능을 제공합니다.

모듈 구성:
- config: 설정, 경로, 모델 초기화
- signature: DSPy Signature 정의
- modules: DSPy Module 정의
- dataset: 데이터셋 로드 및 Example 생성
- metric: 메트릭 함수
- evaluation: 평가 함수
- optimizer: 최적화 함수 (COPRO, MIPROv2)
- logger: 성능 트래킹 및 로깅
- utiles: 유틸리티 함수

사용 예시:
    from core import (
        get_default_model,
        TextToSQLModule,
        load_bird_dataset,
        run_optimization_pipeline
    )
    
    # 모델 설정
    get_default_model()
    
    # 모듈 생성
    module = TextToSQLModule()
    
    # 데이터셋 로드
    train_examples, test_examples = load_bird_dataset(difficulty="challenging")
    
    # 최적화 실행
    optimized_module, results = run_optimization_pipeline(...)
"""

# Config
from .config import (
    BASE_DIR,
    RESULTS_DIR,
    DATA_DIR,
    SQLITE_PATH,
    ModelConfig,
    OptimizerConfig,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_OPTIMIZER_CONFIG,
    get_default_model,
)

# Signature
from .signature import (
    TextToSQLSignature,
    SimpleTextToSQLSignature,
    TextToSQLWithExamplesSignature,
    SQLErrorCorrectionSignature,
    get_signature,
    register_signature,
    SIGNATURE_REGISTRY,
)

# Modules
from .modules import (
    TextToSQLModule,
    SimpleTextToSQLModule,
    TextToSQLWithRetryModule,
    EnsembleTextToSQLModule,
    get_module,
    register_module,
    MODULE_REGISTRY,
)

# Dataset
from .dataset import (
    DatasetLoader,
    ExampleFactory,
    create_dspy_examples,
    load_bird_dataset,
    get_sample_examples,
)

# Metric
from .metric import (
    text_to_sql_metric,
    normalized_metric,
    binary_metric,
    execution_metric,
    MetricConfig,
    create_custom_metric,
    get_metric,
    register_metric,
    METRIC_REGISTRY,
    MAX_SCORE,
)

# Evaluation
from .evaluation import (
    evaluate_model,
    evaluate_with_dspy,
    compare_results,
)

# Optimizer
from .optimizer import (
    # New: with logging & early stopping
    OptimizationStep,
    OptimizationState,
    OptimizationLogger,
    optimize_with_copro_and_logging,
    optimize_with_mipro_and_logging,
    # Legacy
    optimize_with_copro,
    optimize_with_mipro,
    display_optimized_prompt,
    compare_prompts,
    run_optimization_pipeline,
    run_optimization_with_tracking,
)

# Logger
from .logger import (
    PerformanceTracker,
    StepMetrics,
    list_saved_results,
    load_and_display_history,
    display_latest_graph,
)

# Utilities
from .utiles import (
    get_db_schema,
    execute_sql,
    normalize_sql,
    compare_sql,
    compare_results as compare_sql_results,
    calculate_evidence_similarity,
)

__all__ = [
    # Config
    "BASE_DIR",
    "RESULTS_DIR",
    "DATA_DIR",
    "SQLITE_PATH",
    "ModelConfig",
    "OptimizerConfig",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_OPTIMIZER_CONFIG",
    "get_default_model",
    
    # Signature
    "TextToSQLSignature",
    "SimpleTextToSQLSignature",
    "TextToSQLWithExamplesSignature",
    "SQLErrorCorrectionSignature",
    "get_signature",
    "register_signature",
    "SIGNATURE_REGISTRY",
    
    # Modules
    "TextToSQLModule",
    "SimpleTextToSQLModule",
    "TextToSQLWithRetryModule",
    "EnsembleTextToSQLModule",
    "get_module",
    "register_module",
    "MODULE_REGISTRY",
    
    # Dataset
    "DatasetLoader",
    "ExampleFactory",
    "create_dspy_examples",
    "load_bird_dataset",
    "get_sample_examples",
    
    # Metric
    "text_to_sql_metric",
    "normalized_metric",
    "binary_metric",
    "execution_metric",
    "MetricConfig",
    "create_custom_metric",
    "get_metric",
    "register_metric",
    "METRIC_REGISTRY",
    "MAX_SCORE",
    
    # Evaluation
    "evaluate_model",
    "evaluate_with_dspy",
    "compare_results",
    
    # Optimizer
    "OptimizationStep",
    "OptimizationState",
    "OptimizationLogger",
    "optimize_with_copro_and_logging",
    "optimize_with_mipro_and_logging",
    "optimize_with_copro",
    "optimize_with_mipro",
    "display_optimized_prompt",
    "compare_prompts",
    "run_optimization_pipeline",
    "run_optimization_with_tracking",
    
    # Logger
    "PerformanceTracker",
    "StepMetrics",
    "list_saved_results",
    "load_and_display_history",
    "display_latest_graph",
    
    # Utilities
    "get_db_schema",
    "execute_sql",
    "normalize_sql",
    "compare_sql",
    "compare_sql_results",
    "calculate_evidence_similarity",
]

__version__ = "1.0.0"


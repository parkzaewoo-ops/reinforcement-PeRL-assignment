"""
Metric Functions for Text-to-SQL Evaluation
============================================
SQL 쿼리 평가를 위한 메트릭 함수들을 정의합니다.

보상 체계:
- Query와 DB 조회 결과가 완벽히 일치: 3.5점
- Query는 다르지만 DB 결과가 동일: 3.0점
- Evidence 유사도 기반: 0.5 ~ 2.5점
- 실행은 되지만 결과가 틀림: 0.5점
- 에러 발생: 0.0점
"""

import logging
import traceback
from typing import Any, Optional
from .utiles import execute_sql, compare_sql, compare_results, calculate_evidence_similarity

# 로거 설정
logger = logging.getLogger(__name__)

# 에러 로깅 활성화 플래그
VERBOSE_ERRORS = True  # True면 에러 시 상세 출력

# 점수 상수
MAX_SCORE = 3.5
PERFECT_MATCH_SCORE = 3.5
RESULT_MATCH_SCORE = 3.0
MIN_PARTIAL_SCORE = 0.5
MAX_PARTIAL_SCORE = 2.5
WRONG_RESULT_SCORE = 0.5
ERROR_SCORE = 0.0


def text_to_sql_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """
    Text-to-SQL 메트릭 함수 (원본 점수: 0 ~ 3.5)
    
    보상 체계:
    - Query + 결과 완벽 일치: 3.5점
    - Query 다르지만 결과 동일: 3.0점
    - Evidence 유사도 기반: 0.5 ~ 2.5점
    - 실행되지만 결과 틀림: 0.5점
    - 에러 발생: 0.0점
    
    Args:
        example: DSPy Example (gold_sql, gold_evidence 포함)
        prediction: DSPy Prediction (sql_query, evidence 포함)
        trace: DSPy trace (사용 안 함)
    
    Returns:
        점수 (0.0 ~ 3.5)
    """
    question = getattr(example, 'question', '')[:50]  # 디버깅용
    
    try:
        # prediction이 None인 경우 처리
        if prediction is None:
            if VERBOSE_ERRORS:
                print(f"❌ [METRIC] prediction이 None입니다. Q: {question}...")
            return ERROR_SCORE
        
        gold_sql = getattr(example, 'gold_sql', '')
        gold_evidence = getattr(example, 'gold_evidence', '')
        pred_sql = getattr(prediction, 'sql_query', '')
        pred_evidence = getattr(prediction, 'evidence', '')
        
        # pred_sql이 None이거나 빈 문자열인 경우
        if not pred_sql:
            if VERBOSE_ERRORS:
                print(f"❌ [METRIC] pred_sql이 비어있습니다. Q: {question}...")
            return ERROR_SCORE
        
        # 1. 예측된 SQL 실행
        pred_success, pred_result = execute_sql(pred_sql)
        
        # 에러 발생 시 상세 로깅
        if not pred_success:
            if VERBOSE_ERRORS:
                error_msg = str(pred_result)[:100]
                # DB 연결 에러는 크리티컬 - 반드시 출력
                if "unable to open database" in error_msg.lower() or "no such table" in error_msg.lower():
                    print(f"🚨 [CRITICAL] DB 연결/테이블 에러! {error_msg}")
                else:
                    print(f"⚠️ [METRIC] pred_sql 실행 실패: {error_msg}")
            return ERROR_SCORE
        
        # 2. Gold SQL 실행
        gold_success, gold_result = execute_sql(gold_sql)
        if not gold_success:
            if VERBOSE_ERRORS:
                error_msg = str(gold_result)[:100]
                if "unable to open database" in error_msg.lower():
                    print(f"🚨 [CRITICAL] Gold SQL DB 연결 에러! {error_msg}")
            return WRONG_RESULT_SCORE  # Gold SQL 에러지만 예측은 실행됨
        
        # 3. SQL 및 결과 비교
        sql_match = compare_sql(pred_sql, gold_sql)
        result_match = compare_results(pred_result, gold_result)
        
        # 4. 보상 계산
        if sql_match and result_match:
            return PERFECT_MATCH_SCORE
        elif result_match:
            return RESULT_MATCH_SCORE
        else:
            # 결과가 다름 - evidence 유사도로 부분 점수
            evidence_sim = calculate_evidence_similarity(pred_evidence, gold_evidence)
            if evidence_sim > 0.1:
                # evidence_sim (0~1) → 0.5 ~ 2.5로 스케일링
                return MIN_PARTIAL_SCORE + evidence_sim * 2.0
            else:
                return WRONG_RESULT_SCORE
                
    except Exception as e:
        # 예외 발생 시 상세 로깅
        if VERBOSE_ERRORS:
            print(f"🚨 [METRIC ERROR] 예외 발생: {type(e).__name__}: {str(e)[:100]}")
            if "database" in str(e).lower() or "sqlite" in str(e).lower():
                print(f"   ⚠️ DB 관련 에러입니다. DB 경로/연결을 확인하세요.")
                traceback.print_exc()
        return ERROR_SCORE


def normalized_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """
    정규화된 메트릭 (0~1 범위) - DSPy 최적화용
    
    DSPy의 COPRO/MIPROv2/Evaluate는 0~1 범위의 메트릭을 기대합니다.
    원본 점수(0~3.5)를 0~1로 정규화합니다.
    
    Args:
        example: DSPy Example
        prediction: DSPy Prediction
        trace: DSPy trace
    
    Returns:
        정규화된 점수 (0.0 ~ 1.0)
    """
    try:
        score = text_to_sql_metric(example, prediction, trace)
        return score / MAX_SCORE
    except Exception as e:
        if VERBOSE_ERRORS:
            print(f"🚨 [NORMALIZED_METRIC ERROR] {type(e).__name__}: {str(e)[:100]}")
        return 0.0


def binary_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """
    이진 메트릭 (0 또는 1) - 결과 일치 여부만 판단
    
    Args:
        example: DSPy Example
        prediction: DSPy Prediction
        trace: DSPy trace
    
    Returns:
        1.0 (결과 일치) 또는 0.0 (불일치)
    """
    score = text_to_sql_metric(example, prediction, trace)
    return 1.0 if score >= RESULT_MATCH_SCORE else 0.0


def execution_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """
    실행 성공 메트릭 - SQL이 에러 없이 실행되는지만 판단
    
    Args:
        example: DSPy Example
        prediction: DSPy Prediction
        trace: DSPy trace
    
    Returns:
        1.0 (실행 성공) 또는 0.0 (실행 실패)
    """
    pred_sql = getattr(prediction, 'sql_query', '')
    success, _ = execute_sql(pred_sql)
    return 1.0 if success else 0.0


class MetricConfig:
    """메트릭 설정 클래스"""
    
    def __init__(
        self,
        perfect_match_score: float = 3.5,
        result_match_score: float = 3.0,
        min_partial_score: float = 0.5,
        max_partial_score: float = 2.5,
        wrong_result_score: float = 0.5,
        error_score: float = 0.0
    ):
        self.perfect_match = perfect_match_score
        self.result_match = result_match_score
        self.min_partial = min_partial_score
        self.max_partial = max_partial_score
        self.wrong_result = wrong_result_score
        self.error = error_score
        self.max_score = perfect_match_score


def create_custom_metric(config: MetricConfig):
    """커스텀 메트릭 함수 생성"""
    def custom_metric(example: Any, prediction: Any, trace: Any = None) -> float:
        gold_sql = getattr(example, 'gold_sql', '')
        gold_evidence = getattr(example, 'gold_evidence', '')
        pred_sql = getattr(prediction, 'sql_query', '')
        pred_evidence = getattr(prediction, 'evidence', '')
        
        pred_success, pred_result = execute_sql(pred_sql)
        if not pred_success:
            return config.error
        
        gold_success, gold_result = execute_sql(gold_sql)
        if not gold_success:
            return config.wrong_result
        
        sql_match = compare_sql(pred_sql, gold_sql)
        result_match = compare_results(pred_result, gold_result)
        
        if sql_match and result_match:
            return config.perfect_match
        elif result_match:
            return config.result_match
        else:
            evidence_sim = calculate_evidence_similarity(pred_evidence, gold_evidence)
            if evidence_sim > 0.1:
                return config.min_partial + evidence_sim * (config.max_partial - config.min_partial)
            else:
                return config.wrong_result
    
    return custom_metric


# 메트릭 레지스트리
METRIC_REGISTRY = {
    "default": text_to_sql_metric,
    "normalized": normalized_metric,
    "binary": binary_metric,
    "execution": execution_metric,
}


def get_metric(name: str = "normalized"):
    """이름으로 메트릭 함수 반환"""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]


def register_metric(name: str, metric_fn):
    """새로운 메트릭을 레지스트리에 등록"""
    METRIC_REGISTRY[name] = metric_fn
    print(f"✅ Metric '{name}' 등록 완료")


print("✅ 메트릭 함수 로드 완료")

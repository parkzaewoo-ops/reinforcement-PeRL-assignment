"""
DSPy Signature Definitions
==========================
Text-to-SQL 작업을 위한 DSPy Signature 정의
새로운 Signature를 추가하여 확장 가능합니다.
"""

import dspy


class TextToSQLSignature(dspy.Signature):
    """You are an expert SQL developer. Given a natural language question and database schema, 
    generate a correct SQL query that answers the question. 
    Think step by step about the table relationships and required columns."""
    
    # Input fields
    question = dspy.InputField(desc="자연어로 된 데이터베이스 질의 질문")
    table_schema = dspy.InputField(desc="데이터베이스 테이블 스키마 (CREATE TABLE 문)")
    hint = dspy.InputField(desc="질문 해석을 위한 힌트/증거", default="")
    
    # Output fields
    reasoning = dspy.OutputField(desc="SQL 쿼리를 작성하기 위한 단계별 추론 과정")
    sql_query = dspy.OutputField(desc="최종 SQL 쿼리 (SELECT 문)")
    evidence = dspy.OutputField(desc="쿼리 결과를 설명하는 증거 문장")


class SimpleTextToSQLSignature(dspy.Signature):
    """You are an expert SQL developer. Given a natural language question and database schema, 
    generate a correct SQL query that answers the question.
    Focus on table relationships, column names, and SQL syntax."""
    
    question = dspy.InputField(desc="Natural language question about the database")
    table_schema = dspy.InputField(desc="Database schema (CREATE TABLE statements)")
    hint = dspy.InputField(desc="Optional hint for query interpretation", default="")
    
    sql_query = dspy.OutputField(desc="The final SQL query (SELECT statement)")


class TextToSQLWithExamplesSignature(dspy.Signature):
    """You are an expert SQL developer. Generate accurate SQL queries based on the question, 
    schema, and provided examples. Pay attention to column names and table relationships."""
    
    question = dspy.InputField(desc="자연어로 된 데이터베이스 질의 질문")
    table_schema = dspy.InputField(desc="데이터베이스 테이블 스키마")
    hint = dspy.InputField(desc="질문 해석을 위한 힌트", default="")
    examples = dspy.InputField(desc="유사한 질문-SQL 쌍 예시", default="")
    
    reasoning = dspy.OutputField(desc="단계별 추론 과정")
    sql_query = dspy.OutputField(desc="최종 SQL 쿼리")
    confidence = dspy.OutputField(desc="쿼리 정확도에 대한 신뢰도 (0-100)")


class SQLErrorCorrectionSignature(dspy.Signature):
    """Given a failed SQL query and error message, generate a corrected SQL query."""
    
    question = dspy.InputField(desc="원본 질문")
    table_schema = dspy.InputField(desc="데이터베이스 스키마")
    failed_sql = dspy.InputField(desc="실패한 SQL 쿼리")
    error_message = dspy.InputField(desc="에러 메시지")
    
    analysis = dspy.OutputField(desc="에러 원인 분석")
    corrected_sql = dspy.OutputField(desc="수정된 SQL 쿼리")


# Signature 레지스트리 (확장용)
SIGNATURE_REGISTRY = {
    "default": TextToSQLSignature,
    "simple": SimpleTextToSQLSignature,
    "with_examples": TextToSQLWithExamplesSignature,
    "error_correction": SQLErrorCorrectionSignature,
}


def get_signature(name: str = "default") -> type:
    """이름으로 Signature 클래스 반환"""
    if name not in SIGNATURE_REGISTRY:
        raise ValueError(f"Unknown signature: {name}. Available: {list(SIGNATURE_REGISTRY.keys())}")
    return SIGNATURE_REGISTRY[name]


def register_signature(name: str, signature_class: type):
    """새로운 Signature를 레지스트리에 등록"""
    SIGNATURE_REGISTRY[name] = signature_class
    print(f"✅ Signature '{name}' 등록 완료")


import sqlite3
import re
import pandas as pd
from typing import Tuple, Any
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading

SQLITE_PATH = "/data/workspace/sogang/sqlagent/data/BIRD.sqlite"

# 쿼리 실행 타임아웃 (초) - 복잡한 쿼리를 위해 30초로 증가
SQL_EXECUTION_TIMEOUT = 30


def get_db_schema(db_id: str) -> str:
    """데이터베이스에서 스키마 정보를 추출"""
    # 읽기 전용 연결 (동시 접근 가능)
    con = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True, check_same_thread=False)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema_parts = []
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = cursor.fetchall()
        col_defs = []
        for col in columns:
            col_id, col_name, col_type, not_null, default_val, pk = col
            col_def = f"    {col_name} {col_type}"
            if pk:
                col_def += " PRIMARY KEY"
            if not_null:
                col_def += " NOT NULL"
            col_defs.append(col_def)
        schema_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);")    
    con.close()
    return "\n\n".join(schema_parts)


def _execute_sql_internal(sql: str) -> Tuple[bool, Any]:
    """내부 SQL 실행 함수 - 읽기 전용 모드 (Lock 없이 동시 접근 가능)"""
    con = None
    try:
        # 읽기 전용 연결 (uri 모드) - 동시에 여러 연결 가능
        con = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True, check_same_thread=False)
        result = pd.read_sql_query(sql, con)
        return True, result
    finally:
        # 항상 연결 닫기
        if con:
            try:
                con.close()
            except:
                pass


def execute_sql(sql: str, timeout: int = SQL_EXECUTION_TIMEOUT) -> Tuple[bool, Any]:
    """SQL 쿼리 실행 및 결과 반환 (타임아웃 적용)
    
    Returns:
        (success: bool, result: DataFrame or error_message)
    """
    con = None
    try:
        sql = sql.strip()
        if sql.startswith("```"):
            sql = re.sub(r'^```\w*\n?', '', sql)
            sql = re.sub(r'\n?```$', '', sql)
        
        # 빈 쿼리 체크
        if not sql:
            return False, "Empty SQL query"
        
        # 직접 실행 (ThreadPoolExecutor 제거 - 파일 핸들러 누수 방지)
        con = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True, check_same_thread=False, timeout=timeout)
        result = pd.read_sql_query(sql, con)
        return True, result
    except Exception as e:
        return False, str(e)
    finally:
        # 항상 연결 닫기
        if con:
            try:
                con.close()
            except:
                pass


def normalize_result(df: pd.DataFrame) -> str:
    """결과를 정규화하여 비교 가능한 문자열로 변환"""
    if df.empty:
        return ""
    # 컬럼 이름 소문자로, 정렬 후 문자열 변환
    df_normalized = df.copy()
    df_normalized.columns = [str(c).lower().strip() for c in df_normalized.columns]
    df_normalized = df_normalized.sort_values(by=list(df_normalized.columns)).reset_index(drop=True)
    return df_normalized.to_string(index=False)


def normalize_sql(sql: str) -> str:
    """SQL 정규화 (공백, 대소문자 등)"""
    sql = sql.strip().lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*\(\s*', '(', sql)
    sql = re.sub(r'\s*\)\s*', ')', sql)
    return sql


def compare_sql(pred_sql: str, gold_sql: str) -> bool:
    """두 SQL이 동일한지 비교"""
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


def compare_results(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> bool:
    """두 결과가 동일한지 비교"""
    try:
        return normalize_result(pred_df) == normalize_result(gold_df)
    except:
        return False

def calculate_evidence_similarity(pred_evidence: str, gold_evidence: str) -> float:
    """evidence 문장 유사도 계산 (0~1)"""
    if not pred_evidence or not gold_evidence:
        return 0.0
    return SequenceMatcher(None, pred_evidence.lower(), gold_evidence.lower()).ratio()

print("유틸리티 함수 로드 완료")


if __name__ == "__main__":
    sample_schema = get_db_schema("california_schools")
    print(sample_schema[:1000] + "...")

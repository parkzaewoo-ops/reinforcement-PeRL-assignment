"""
Dataset Loading and Example Creation
=====================================
데이터셋 로드 및 DSPy Example 생성을 담당합니다.
"""

import dspy
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from .utiles import get_db_schema


class DatasetLoader:
    """데이터셋 로더 클래스"""
    
    def __init__(
        self,
        dataset_name: str = "birdsql/bird_sql_dev_20251106",
        split_name: str = "dev_20251106"
    ):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self._df = None
    
    def load(self) -> pd.DataFrame:
        """데이터셋 로드"""
        if self._df is None:
            ds = load_dataset(self.dataset_name)
            self._df = ds[self.split_name].to_pandas()
            print(f"✅ 데이터셋 로드 완료: {len(self._df)}개 샘플")
        return self._df
    
    def filter_by_difficulty(self, difficulty: str) -> pd.DataFrame:
        """난이도로 필터링"""
        df = self.load()
        filtered = df[df['difficulty'] == difficulty].copy()
        print(f"📊 '{difficulty}' 난이도: {len(filtered)}개 샘플")
        return filtered
    
    def split_data(
        self,
        df: pd.DataFrame = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """학습/테스트 데이터 분할"""
        if df is None:
            df = self.load()
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        print(f"📊 Train: {len(train_df)}개, Test: {len(test_df)}개")
        return train_df, test_df


class ExampleFactory:
    """DSPy Example 생성 팩토리"""
    
    def __init__(self, db_id: str = "bird"):
        self.db_id = db_id
        self._schema = None
    
    @property
    def schema(self) -> str:
        """스키마 캐싱"""
        if self._schema is None:
            self._schema = get_db_schema(self.db_id)
        return self._schema
    
    def create_example(
        self,
        question: str,
        gold_sql: str,
        evidence: str = "",
        hint: str = ""
    ) -> dspy.Example:
        """단일 Example 생성"""
        return dspy.Example(
            question=question,
            table_schema=self.schema,
            hint=hint if hint else evidence,
            gold_sql=gold_sql,
            gold_evidence=evidence
        ).with_inputs('question', 'table_schema', 'hint')
    
    def create_examples_from_df(
        self, 
        df: pd.DataFrame,
        question_col: str = "question",
        sql_col: str = "SQL",
        evidence_col: str = "evidence"
    ) -> List[dspy.Example]:
        """DataFrame에서 Example 리스트 생성"""
        examples = []
        
        for _, row in df.iterrows():
            example = self.create_example(
                question=row[question_col],
                gold_sql=row[sql_col],
                evidence=row[evidence_col] if pd.notna(row.get(evidence_col)) else "",
                hint=row[evidence_col] if pd.notna(row.get(evidence_col)) else ""
            )
            examples.append(example)
        
        print(f"✅ {len(examples)}개 Example 생성 완료")
        return examples


def create_dspy_examples(
    df: pd.DataFrame, 
    db_id: str = "bird"
) -> List[dspy.Example]:
    """
    DataFrame에서 DSPy Example 리스트 생성 (레거시 호환)
    
    Args:
        df: 데이터프레임 (question, SQL, evidence 컬럼 필요)
        db_id: 데이터베이스 ID
    
    Returns:
        DSPy Example 리스트
    """
    factory = ExampleFactory(db_id=db_id)
    return factory.create_examples_from_df(df)


def load_bird_dataset(
    difficulty: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    BIRD 데이터셋을 로드하고 DSPy Example로 변환
    
    Args:
        difficulty: 난이도 필터 ('simple', 'moderate', 'challenging', None)
        test_size: 테스트 세트 비율 (전체 대비)
        random_state: 랜덤 시드
    
    Returns:
        (train_examples, test_examples)
    
    데이터 분할:
        전체 → train (80%) + test (20%)
    """
    loader = DatasetLoader()
    
    if difficulty:
        df = loader.filter_by_difficulty(difficulty)
    else:
        df = loader.load()
    
    # train vs test 분할
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    print(f"📊 데이터셋 분할:")
    print(f"   Train: {len(train_df)}개 (최적화용)")
    print(f"   Test: {len(test_df)}개 (최종 평가용)")
    
    factory = ExampleFactory()
    train_examples = factory.create_examples_from_df(train_df)
    test_examples = factory.create_examples_from_df(test_df)
    
    return train_examples, test_examples


def load_bird_dataset_legacy(
    difficulty: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    BIRD 데이터셋 로드 (레거시 - train/test만)
    
    Returns:
        (train_examples, test_examples)
    """
    loader = DatasetLoader()
    
    if difficulty:
        df = loader.filter_by_difficulty(difficulty)
    else:
        df = loader.load()
    
    train_df, test_df = loader.split_data(df, test_size, random_state)
    
    factory = ExampleFactory()
    train_examples = factory.create_examples_from_df(train_df)
    test_examples = factory.create_examples_from_df(test_df)
    
    return train_examples, test_examples


# 빠른 사용을 위한 편의 함수
def get_sample_examples(n: int = 10, difficulty: str = "challenging") -> List[dspy.Example]:
    """샘플 Example 반환 (테스트용)"""
    loader = DatasetLoader()
    df = loader.filter_by_difficulty(difficulty)
    
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    
    factory = ExampleFactory()
    return factory.create_examples_from_df(df)


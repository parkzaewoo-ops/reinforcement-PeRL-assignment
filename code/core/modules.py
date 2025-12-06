"""
DSPy Module Definitions
=======================
Text-to-SQL 작업을 위한 DSPy Module 정의
새로운 모듈을 추가하여 확장 가능합니다.
"""

import dspy
from typing import Optional
from .signature import (
    TextToSQLSignature,
    SimpleTextToSQLSignature,
    TextToSQLWithExamplesSignature,
    SQLErrorCorrectionSignature,
    get_signature
)


class TextToSQLModule(dspy.Module):
    """Text-to-SQL 변환 모듈 (기본 ChainOfThought)"""
    
    def __init__(self, signature_name: str = "default"):
        super().__init__()
        signature_class = get_signature(signature_name)
        self.cot = dspy.ChainOfThought(signature_class)
    
    def forward(self, question: str, table_schema: str, hint: str = "") -> dspy.Prediction:
        prediction = self.cot(
            question=question,
            table_schema=table_schema,
            hint=hint
        )
        return dspy.Prediction(
            reasoning=getattr(prediction, 'reasoning', ''),
            sql_query=getattr(prediction, 'sql_query', ''),
            evidence=getattr(prediction, 'evidence', '')
        )


class SimpleTextToSQLModule(dspy.Module):
    """간단한 Text-to-SQL 모듈 (sql_query만 출력 - 최적화에 효과적)"""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SimpleTextToSQLSignature)
    
    def forward(self, question: str, table_schema: str, hint: str = "") -> dspy.Prediction:
        prediction = self.predict(
            question=question,
            table_schema=table_schema,
            hint=hint
        )
        return dspy.Prediction(
            sql_query=getattr(prediction, 'sql_query', ''),
            reasoning=getattr(prediction, 'reasoning', '')  # CoT 내부 reasoning
        )


class TextToSQLWithRetryModule(dspy.Module):
    """에러 발생 시 자동 수정을 시도하는 모듈"""
    
    def __init__(self, max_retries: int = 2):
        super().__init__()
        self.cot = dspy.ChainOfThought(TextToSQLSignature)
        self.error_correction = dspy.ChainOfThought(SQLErrorCorrectionSignature)
        self.max_retries = max_retries
    
    def forward(
        self, 
        question: str, 
        table_schema: str, 
        hint: str = "",
        execute_fn: Optional[callable] = None
    ) -> dspy.Prediction:
        """
        Args:
            execute_fn: SQL 실행 함수 (success, result) 튜플 반환
        """
        # 첫 번째 시도
        prediction = self.cot(
            question=question,
            table_schema=table_schema,
            hint=hint
        )
        
        sql_query = getattr(prediction, 'sql_query', '')
        
        # 실행 함수가 제공된 경우 검증 및 재시도
        if execute_fn is not None:
            for retry in range(self.max_retries):
                success, result = execute_fn(sql_query)
                if success:
                    break
                
                # 에러 수정 시도
                correction = self.error_correction(
                    question=question,
                    table_schema=table_schema,
                    failed_sql=sql_query,
                    error_message=str(result)
                )
                sql_query = getattr(correction, 'corrected_sql', sql_query)
        
        return dspy.Prediction(
            reasoning=getattr(prediction, 'reasoning', ''),
            sql_query=sql_query,
            evidence=getattr(prediction, 'evidence', '')
        )


class EnsembleTextToSQLModule(dspy.Module):
    """여러 모듈의 결과를 앙상블하는 모듈"""
    
    def __init__(self, num_samples: int = 3):
        super().__init__()
        self.num_samples = num_samples
        self.modules = [
            dspy.ChainOfThought(TextToSQLSignature) 
            for _ in range(num_samples)
        ]
    
    def forward(
        self, 
        question: str, 
        table_schema: str, 
        hint: str = "",
        voting_fn: Optional[callable] = None
    ) -> dspy.Prediction:
        """
        Args:
            voting_fn: SQL 결과 투표 함수 (가장 좋은 쿼리 선택)
        """
        predictions = []
        for module in self.modules:
            pred = module(
                question=question,
                table_schema=table_schema,
                hint=hint
            )
            predictions.append(pred)
        
        # 투표 함수가 없으면 첫 번째 결과 반환
        if voting_fn is None:
            best = predictions[0]
        else:
            sql_queries = [getattr(p, 'sql_query', '') for p in predictions]
            best_idx = voting_fn(sql_queries)
            best = predictions[best_idx]
        
        return dspy.Prediction(
            reasoning=getattr(best, 'reasoning', ''),
            sql_query=getattr(best, 'sql_query', ''),
            evidence=getattr(best, 'evidence', '')
        )


# Module 레지스트리 (확장용)
MODULE_REGISTRY = {
    "default": TextToSQLModule,
    "simple": SimpleTextToSQLModule,
    "with_retry": TextToSQLWithRetryModule,
    "ensemble": EnsembleTextToSQLModule,
}


def get_module(name: str = "default", **kwargs) -> dspy.Module:
    """이름으로 Module 인스턴스 반환"""
    if name not in MODULE_REGISTRY:
        raise ValueError(f"Unknown module: {name}. Available: {list(MODULE_REGISTRY.keys())}")
    return MODULE_REGISTRY[name](**kwargs)


def register_module(name: str, module_class: type):
    """새로운 Module을 레지스트리에 등록"""
    MODULE_REGISTRY[name] = module_class
    print(f"✅ Module '{name}' 등록 완료")


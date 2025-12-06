#!/usr/bin/env python3
"""
Text-to-SQL Inference Script
============================

저장된 모델을 로드하여 자연어 질문을 SQL로 변환합니다.

사용법:
    # 단일 질문 추론
    python inference.py "How many students are there?"
    
    # 최적화된 모델 사용
    python inference.py "How many students?" --model optimized_model.json
    
    # 힌트와 함께 사용
    python inference.py "What is the average score?" --hint "score는 점수를 의미"
    
    # 대화형 모드
    python inference.py --interactive
    
    # 배치 추론 (CSV 파일)
    python inference.py --batch questions.csv --output results.csv
"""

import argparse
import json
import os
import sys
import pandas as pd
from typing import Optional, Dict, List

from core import (
    get_default_model,
    TextToSQLModule,
    get_db_schema,
    execute_sql,
    ModelConfig,
)


class SQLInferenceEngine:
    """SQL 추론 엔진"""
    
    def __init__(self, model_path: Optional[str] = None, db_id: str = "bird"):
        """
        Args:
            model_path: 저장된 모델 경로 (None이면 베이스라인 사용)
            db_id: 데이터베이스 ID
        """
        self.db_id = db_id
        self.model_path = model_path
        self._module = None
        self._schema = None
        self._initialized = False
    
    def initialize(self):
        """모델 초기화 (지연 로딩)"""
        if self._initialized:
            return
        
        # DSPy 모델 설정
        get_default_model()
        
        # 모듈 생성
        self._module = TextToSQLModule()
        
        # 저장된 모델 로드
        if self.model_path and os.path.exists(self.model_path):
            self._module.load(self.model_path)
            print(f"✅ 모델 로드 완료: {self.model_path}")
        else:
            print("📝 베이스라인 모델 사용")
        
        # 스키마 캐싱
        self._schema = get_db_schema(self.db_id)
        
        self._initialized = True
    
    def predict(
        self, 
        question: str, 
        hint: str = "",
        execute: bool = False
    ) -> Dict:
        """
        단일 질문에 대한 SQL 생성
        
        Args:
            question: 자연어 질문
            hint: 추가 힌트/컨텍스트
            execute: SQL 실행 여부
        
        Returns:
            예측 결과 딕셔너리
        """
        self.initialize()
        
        # 추론 실행
        prediction = self._module(
            question=question,
            table_schema=self._schema,
            hint=hint
        )
        
        result = {
            "question": question,
            "hint": hint,
            "sql_query": getattr(prediction, 'sql_query', ''),
            "reasoning": getattr(prediction, 'reasoning', ''),
            "evidence": getattr(prediction, 'evidence', ''),
        }
        
        # SQL 실행 (선택사항)
        if execute:
            success, query_result = execute_sql(result['sql_query'])
            result['execution_success'] = success
            if success:
                result['query_result'] = query_result.to_dict('records')
            else:
                result['execution_error'] = str(query_result)
        
        return result
    
    def predict_batch(
        self, 
        questions: List[Dict],
        execute: bool = False
    ) -> List[Dict]:
        """
        배치 추론
        
        Args:
            questions: [{"question": "...", "hint": "..."}, ...]
            execute: SQL 실행 여부
        
        Returns:
            예측 결과 리스트
        """
        results = []
        total = len(questions)
        
        for i, q in enumerate(questions, 1):
            question = q.get('question', q) if isinstance(q, dict) else q
            hint = q.get('hint', '') if isinstance(q, dict) else ''
            
            print(f"\n[{i}/{total}] {question[:50]}...")
            
            try:
                result = self.predict(question, hint, execute)
                results.append(result)
            except Exception as e:
                results.append({
                    "question": question,
                    "error": str(e)
                })
        
        return results


def run_single_inference(args):
    """단일 질문 추론"""
    engine = SQLInferenceEngine(model_path=args.model)
    
    result = engine.predict(
        question=args.question,
        hint=args.hint or "",
        execute=args.execute
    )
    
    print("\n" + "="*60)
    print("🎯 추론 결과")
    print("="*60)
    print(f"\n📝 질문: {result['question']}")
    
    if result.get('hint'):
        print(f"💡 힌트: {result['hint']}")
    
    print(f"\n🔍 추론 과정:")
    print("-"*40)
    print(result['reasoning'])
    
    print(f"\n💾 생성된 SQL:")
    print("-"*40)
    print(result['sql_query'])
    
    if result.get('evidence'):
        print(f"\n📋 Evidence:")
        print(result['evidence'])
    
    if args.execute:
        print(f"\n⚡ 실행 결과:")
        print("-"*40)
        if result.get('execution_success'):
            if result.get('query_result'):
                df = pd.DataFrame(result['query_result'])
                print(df.to_string(index=False))
            else:
                print("(빈 결과)")
        else:
            print(f"❌ 에러: {result.get('execution_error')}")
    
    return result


def run_interactive_mode(args):
    """대화형 모드"""
    engine = SQLInferenceEngine(model_path=args.model)
    
    print("\n" + "="*60)
    print("🤖 Text-to-SQL 대화형 모드")
    print("="*60)
    print("질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("힌트를 추가하려면: question | hint 형식으로 입력하세요.")
    print("-"*60)
    
    while True:
        try:
            user_input = input("\n📝 질문: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 종료합니다.")
                break
            
            # 힌트 파싱 (question | hint 형식)
            if '|' in user_input:
                parts = user_input.split('|', 1)
                question = parts[0].strip()
                hint = parts[1].strip()
            else:
                question = user_input
                hint = ""
            
            result = engine.predict(question, hint, execute=args.execute)
            
            print(f"\n💾 SQL:")
            print(result['sql_query'])
            
            if args.execute and result.get('execution_success'):
                print(f"\n⚡ 결과:")
                if result.get('query_result'):
                    df = pd.DataFrame(result['query_result'])
                    print(df.head(10).to_string(index=False))
                    if len(result['query_result']) > 10:
                        print(f"... ({len(result['query_result'])}개 행)")
                        
        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 에러: {e}")


def run_batch_inference(args):
    """배치 추론 (CSV 파일)"""
    if not os.path.exists(args.batch):
        print(f"❌ 파일을 찾을 수 없습니다: {args.batch}")
        return
    
    # CSV 로드
    df = pd.read_csv(args.batch)
    
    if 'question' not in df.columns:
        print("❌ CSV 파일에 'question' 컬럼이 필요합니다.")
        return
    
    questions = df.to_dict('records')
    
    print(f"\n📂 {len(questions)}개 질문 로드됨")
    
    engine = SQLInferenceEngine(model_path=args.model)
    results = engine.predict_batch(questions, execute=args.execute)
    
    # 결과 저장
    output_path = args.output or args.batch.replace('.csv', '_results.csv')
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 결과 저장 완료: {output_path}")
    
    # 요약 출력
    success_count = sum(1 for r in results if 'error' not in r)
    print(f"📊 성공: {success_count}/{len(results)}")


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py "How many students are there?"
  python inference.py "What is the average score?" --execute
  python inference.py --interactive
  python inference.py --batch questions.csv --output results.csv
        """
    )
    
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="자연어 질문"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="저장된 모델 경로 (없으면 베이스라인 사용)"
    )
    
    parser.add_argument(
        "--hint",
        type=str,
        default="",
        help="추가 힌트/컨텍스트"
    )
    
    parser.add_argument(
        "--execute", "-e",
        action="store_true",
        help="생성된 SQL 실행"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=str,
        default=None,
        help="배치 추론용 CSV 파일 경로"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="배치 결과 저장 경로"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="결과를 JSON으로 출력"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.interactive:
        run_interactive_mode(args)
    elif args.batch:
        run_batch_inference(args)
    elif args.question:
        result = run_single_inference(args)
        if args.json:
            print("\n" + json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("질문을 입력하거나 --interactive 또는 --batch 옵션을 사용하세요.")
        print("도움말: python inference.py --help")


if __name__ == "__main__":
    main()


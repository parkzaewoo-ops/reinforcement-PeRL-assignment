#!/usr/bin/env python3
"""
Text-to-SQL DSPy Optimizer - Main Script
=========================================

DSPy를 사용한 Text-to-SQL 모델 최적화 메인 스크립트입니다.

사용법:
    # 기본 실행 (COPRO 최적화)
    python run_optimizer.py
    
    # MIPROv2 최적화
    python run_optimizer.py --optimizer mipro
    
    # 커스텀 설정
    python run_optimizer.py --optimizer copro --breadth 5 --depth 2
    
    # 모델만 평가
    python run_optimizer.py --mode evaluate --model-path optimized_text2sql_model.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
import dspy

# 에러 시 스킵하도록 설정
dspy.configure(experimental=True)

# Core imports
from core import (
    # Config
    get_default_model,
    RESULTS_DIR,
    ModelConfig,
    OptimizerConfig,
    # Modules
    TextToSQLModule,
    SimpleTextToSQLModule,  # 간단한 모듈 (sql_query만 출력)
    # Dataset
    load_bird_dataset,
    # Evaluation
    evaluate_model,
    # Optimizer
    run_optimization_pipeline,
    # Logger
    PerformanceTracker,
)


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL DSPy Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimizer.py --optimizer copro
  python run_optimizer.py --optimizer mipro --num-threads 20
  python run_optimizer.py --mode evaluate --model-path model.json
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="optimize",
        choices=["optimize", "evaluate", "compare"],
        help="실행 모드 (기본값: optimize)"
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="copro",
        choices=["copro", "mipro"],
        help="최적화 방법 (기본값: copro)"
    )
    
    # COPRO 설정
    parser.add_argument(
        "--breadth",
        type=int,
        default=5,
        help="COPRO breadth - 최소 2 이상 (기본값: 5)"
    )
    
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="COPRO depth (기본값: 1)"
    )
    
    # MIPROv2 설정
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help="MIPROv2 후보 수 (기본값: 10)"
    )
    
    parser.add_argument(
        "--max-demos",
        type=int,
        default=3,
        help="MIPROv2 최대 데모 수 (기본값: 3)"
    )
    
    # 모듈 설정
    parser.add_argument(
        "--module",
        type=str,
        default="simple",
        choices=["default", "simple"],
        help="사용할 모듈 (default: 복잡한 출력, simple: sql_query만 - 최적화에 효과적)"
    )
    
    # 공통 설정
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="병렬 처리 스레드 수 (기본값: 8, 너무 높으면 타임아웃 문제 발생)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="초기 temperature (기본값: 0.6)"
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        default="challenging",
        choices=["simple", "moderate", "challenging", "all"],
        help="데이터셋 난이도 필터 (기본값: challenging)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="테스트 데이터 비율 (기본값: 0.2)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=f"optimized_text2sql_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="모델 저장/로드 경로"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="최적화된 모델 저장하지 않음"
    )
    
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=5,
        help="평가에 사용할 샘플 수 (기본값: 5)"
    )
    parser.add_argument(
        "--use-early-stopping",
        action="store_true",
        help="Early Stopping 사용 (기본값: True)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early Stopping 파일럿 (기본값: 3)"
    )
    return parser.parse_args()


def save_module(module, path: str):
    """모듈 저장"""
    module.save(path)
    print(f"✅ 모델 저장 완료: {path}")


def load_module(path: str) -> TextToSQLModule:
    """모듈 로드"""
    module = TextToSQLModule()
    module.load(path)
    print(f"✅ 모델 로드 완료: {path}")
    return module


def run_optimization(args):
    """최적화 실행"""
    print("="*60)
    print("🚀 Text-to-SQL DSPy Optimizer")
    print("="*60)
    
    # 모델 설정
    get_default_model()
    
    # 데이터셋 로드 (train/test 분할)
    difficulty = None if args.difficulty == "all" else args.difficulty
    train_examples, test_examples = load_bird_dataset(
        difficulty=difficulty,
        test_size=args.test_size
    )
    
    print(f"\n📊 데이터셋:")
    print(f"   Train: {len(train_examples)}개 (최적화용)")
    print(f"   Test: {len(test_examples)}개 (최종 평가용)")
    
    # 모듈 생성 (simple: sql_query만 출력 → 최적화에 효과적)
    if args.module == "simple":
        module = SimpleTextToSQLModule()
        print(f"\n📦 모듈: SimpleTextToSQLModule (sql_query만 출력)")
    else:
        module = TextToSQLModule()
        print(f"\n📦 모듈: TextToSQLModule (reasoning + sql_query + evidence)")
    
    # 트래커 생성
    tracker = PerformanceTracker(f"bird_text2sql_{args.optimizer}_optimization")
    
    # 최적화 설정
    config = OptimizerConfig(
        optimizer_type=args.optimizer,
        breadth=args.breadth,
        depth=args.depth,
        num_candidates=args.num_candidates,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos + 1,
        init_temperature=args.temperature,
        num_threads=args.num_threads
    )
    
    print(f"\n🔧 최적화 설정:")
    print(f"   Optimizer: {args.optimizer.upper()}")
    print(config)
    
    # 최적화 실행 (test_examples로 단계별 평가)
    optimized_module, final_results = run_optimization_pipeline(
        module=module,
        train_examples=train_examples,
        test_examples=test_examples,    # 단계별 평가 & 최종 평가용
        tracker=tracker,
        optimizer_type=args.optimizer,
        config=config,
        save_intermediate=True,
        use_early_stopping=True,
        patience=args.patience
    )
    
    print("\n✅ 최적화 완료!")
    print(f"📁 결과 저장 경로: {RESULTS_DIR}")
    
    # 추가 평가
    print("\n" + "="*60)
    print("📝 최종 평가")
    print("="*60)
    
    # 베이스라인도 같은 모듈 타입 사용
    if args.module == "simple":
        baseline_module = SimpleTextToSQLModule()
    else:
        baseline_module = TextToSQLModule()
    
    baseline_results = evaluate_model(
        baseline_module, 
        test_examples[:args.eval_samples], 
        verbose=True
    )
    
    optimized_results = evaluate_model(
        optimized_module, 
        test_examples[:args.eval_samples], 
        verbose=True
    )
    
    improvement = optimized_results['avg_score'] - baseline_results['avg_score']
    print(f"\n📈 개선율: {improvement:+.3f} 점")
    print(f"베이스라인: {baseline_results['avg_score']:.3f} → 최적화: {optimized_results['avg_score']:.3f}")
    
    # 모델 저장
    if not args.no_save:
        save_path = os.path.join(os.path.dirname(__file__),f"optimized_{args.optimizer}_{args.model_path}")
        save_module(optimized_module, save_path)
    
    return optimized_module, final_results


def run_evaluation(args):
    """모델 평가"""
    print("="*60)
    print("📝 Text-to-SQL 모델 평가")
    print("="*60)
    
    # 모델 설정
    get_default_model()
    
    # 데이터셋 로드
    difficulty = None if args.difficulty == "all" else args.difficulty
    _, test_examples = load_bird_dataset(
        difficulty=difficulty,
        test_size=args.test_size
    )
    
    # 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
    
    if os.path.exists(model_path):
        module = load_module(model_path)
    else:
        print(f"⚠️ 모델 파일이 없습니다: {model_path}")
        print("   베이스라인 모델로 평가합니다.")
        if args.module == "simple":
            module = SimpleTextToSQLModule()
        else:
            module = TextToSQLModule()
    
    # 평가
    results = evaluate_model(
        module, 
        test_examples[:args.eval_samples], 
        verbose=True
    )
    
    return results

def run_comparison(args):
    """베이스라인 vs 최적화 모델 비교"""
    print("="*60)
    print("📊 모델 비교: Baseline vs Optimized")
    print("="*60)
    # 모델 설정
    get_default_model()
    
    # 데이터셋 로드
    difficulty = None if args.difficulty == "all" else args.difficulty
    _, test_examples = load_bird_dataset(
        difficulty=difficulty,
        test_size=args.test_size
    )
    # 베이스라인 모델 (같은 모듈 타입 사용)
    if args.module == "simple":
        baseline_module = SimpleTextToSQLModule()
    else:
        baseline_module = TextToSQLModule()
    # 최적화 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
    
    if os.path.exists(model_path):
        optimized_module = load_module(model_path)
    else:
        print(f"⚠️ 최적화 모델 파일이 없습니다: {model_path}")
        return
    
    # 평가
    print("\n🔵 베이스라인 평가:")
    baseline_results = evaluate_model(
        baseline_module, 
        # test_examples[:args.eval_samples], 
        test_examples,
        verbose=True
    )
    
    print("\n🟢 최적화 모델 평가:")
    optimized_results = evaluate_model(
        optimized_module, 
        # test_examples[:args.eval_samples], 
        test_examples,
        verbose=True
    )
    
    # 비교
    improvement = optimized_results['avg_score'] - baseline_results['avg_score']
    
    print("\n" + "="*60)
    print("📈 성능 비교 요약")
    print("="*60)
    print(f"베이스라인 평균 점수: {baseline_results['avg_score']:.3f}")
    print(f"최적화 평균 점수: {optimized_results['avg_score']:.3f}")
    print(f"개선율: {improvement:+.3f} ({improvement/baseline_results['avg_score']*100:+.1f}%)")
    
    return {
        'baseline': baseline_results,
        'optimized': optimized_results,
        'improvement': improvement
    }


def main():
    args = parse_args()
    
    if args.mode == "optimize":
        run_optimization(args)
    elif args.mode == "evaluate":
        run_evaluation(args)
    elif args.mode == "compare":
        run_comparison(args)
    else:
        print(f"알 수 없는 모드: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()


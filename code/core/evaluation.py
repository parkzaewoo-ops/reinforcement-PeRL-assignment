"""
Evaluation Functions
====================
모델 평가를 위한 함수들을 정의합니다.
"""

import time
import dspy
from dspy.evaluate import Evaluate
from typing import List, Dict, Any, Optional

from .metric import text_to_sql_metric, normalized_metric, MAX_SCORE
from .logger import PerformanceTracker


def evaluate_model(
    module: dspy.Module, 
    examples: List[dspy.Example], 
    verbose: bool = True
) -> dict:
    """
    모델 평가 및 상세 결과 반환
    
    Args:
        module: 평가할 DSPy 모듈
        examples: 평가 데이터셋
        verbose: 상세 출력 여부
    
    Returns:
        평가 결과 딕셔너리
    """
    results = {
        'total': len(examples),
        'perfect_match': 0,      # 3.5점
        'result_match': 0,       # 3.0점
        'partial_match': 0,      # 0.5 ~ 2.5점
        'wrong_result': 0,       # 0.5점
        'error': 0,              # 0점
        'scores': [],
        'details': []
    }
    
    for i, example in enumerate(examples):
        try:
            prediction = module(
                question=example.question,
                table_schema=example.table_schema,
                hint=getattr(example, 'hint', '')
            )
            
            score = text_to_sql_metric(example, prediction)
            results['scores'].append(score)
            
            # 카테고리 분류
            if score >= 3.4:
                results['perfect_match'] += 1
                category = "✅ Perfect"
            elif score >= 2.9:
                results['result_match'] += 1
                category = "🟢 Result Match"
            elif score >= 0.6:
                results['partial_match'] += 1
                category = f"🟡 Partial ({score:.2f})"
            elif score >= 0.4:
                results['wrong_result'] += 1
                category = "🟠 Wrong Result"
            else:
                results['error'] += 1
                category = "❌ Error"
            
            if verbose and i < 5:
                print(f"\n[{i+1}] {category}")
                print(f"Q: {example.question[:80]}...")
                print(f"Pred SQL: {getattr(prediction, 'sql_query', '')[:80]}...")
                print(f"Score: {score}")
            
            results['details'].append({
                'question': example.question,
                'pred_sql': getattr(prediction, 'sql_query', ''),
                'gold_sql': example.gold_sql,
                'score': score,
                'category': category
            })
            
        except Exception as e:
            results['scores'].append(0)
            results['error'] += 1
            results['details'].append({
                'question': example.question,
                'error': str(e),
                'score': 0,
                'category': "❌ Error"
            })
    
    # 통계 계산
    results['avg_score'] = sum(results['scores']) / len(results['scores']) if results['scores'] else 0
    results['avg_normalized'] = results['avg_score'] / MAX_SCORE
    
    _print_evaluation_summary(results)
    return results


def evaluate_with_dspy(
    module: dspy.Module, 
    examples: List[dspy.Example],
    tracker: Optional[PerformanceTracker] = None,
    step_name: str = "",
    num_threads: int = 1,
    display_progress: bool = True
) -> dict:
    """
    DSPy Evaluate를 사용한 평가 + 트래킹
    
    Args:
        module: 평가할 DSPy 모듈
        examples: 평가 데이터셋
        tracker: PerformanceTracker 인스턴스
        step_name: 스텝 이름
        num_threads: 병렬 처리 스레드 수
        display_progress: 진행 상황 표시 여부
    
    Returns:
        평가 결과 딕셔너리
    """
    # 모듈 검증
    if module is None:
        raise ValueError("평가할 모듈이 None입니다. 최적화가 제대로 완료되지 않았을 수 있습니다.")
    
    start_time = time.time()
    
    # DSPy Evaluate 생성
    evaluator = Evaluate(
        devset=examples,
        metric=normalized_metric,
        num_threads=num_threads,
        display_progress=display_progress,
        provide_traceback=True  # 상세 에러 traceback 표시
    )
    
    # 평가 실행
    try:
        eval_result = evaluator(module)
    except Exception as e:
        print(f"⚠️ DSPy Evaluate 에러: {e}")
        # Fallback: 수동 평가
        return _fallback_evaluation(module, examples, step_name)
    elapsed = time.time() - start_time
    
    # DSPy 결과에서 평균 점수 추출 (0~1 범위로 정규화)
    dspy_avg_score = 0.0
    raw_score = 0.0
    
    if hasattr(eval_result, 'score'):
        raw_score = float(eval_result.score) if eval_result.score is not None else 0.0
    elif isinstance(eval_result, (int, float)):
        raw_score = float(eval_result)
    
    # DSPy 점수 정규화: 항상 0~1 범위로 변환
    # - 0~1: 이미 정규화됨
    # - 1~100: 백분율
    # - 100+: 총점 (샘플 수로 나눔)
    if raw_score > 100:
        # 총점인 경우 (예: 17.3 / 47 = 0.368)
        dspy_avg_score = raw_score / len(examples) if len(examples) > 0 else 0.0
    elif raw_score > 1.0:
        # 백분율인 경우 (예: 36.8 → 0.368)
        dspy_avg_score = raw_score / 100.0
    else:
        # 이미 정규화됨 (0~1)
        dspy_avg_score = raw_score
    
    # 최종 범위 제한 (0~1)
    dspy_avg_score = max(0.0, min(1.0, dspy_avg_score))
    
    # 상세 결과 분석
    results = _analyze_evaluation_results(module, examples, eval_result, dspy_avg_score)
    results['elapsed_time'] = elapsed
    
    # 트래킹 로그
    if tracker:
        tracker.log_step(results, step_name)
    
    _print_dspy_evaluation_summary(results, step_name)
    return results


def _analyze_evaluation_results(
    module: dspy.Module, 
    examples: List[dspy.Example], 
    eval_result: Any,
    dspy_avg_score: float = 0.0
) -> dict:
    """평가 결과 분석"""
    results = {
        'total': len(examples),
        'perfect_match': 0,
        'result_match': 0,
        'partial_match': 0,
        'wrong_result': 0,
        'error': 0,
        'scores': [],
        'details': []
    }
    
    # 개별 결과 추출 시도
    individual_results = None
    if hasattr(eval_result, 'results') and eval_result.results:
        individual_results = eval_result.results
    elif hasattr(eval_result, 'outputs') and eval_result.outputs:
        individual_results = eval_result.outputs
    
    parsed_successfully = False
    
    if individual_results:
        for i, result_item in enumerate(individual_results):
            # 다양한 DSPy 버전 호환
            if hasattr(result_item, 'example'):
                example = result_item.example
            elif i < len(examples):
                example = examples[i]
            else:
                continue
            
            # prediction/output 추출
            output = None
            if hasattr(result_item, 'prediction'):
                output = result_item.prediction
            elif hasattr(result_item, 'output'):
                output = result_item.output
            
            # score 추출
            norm_score = 0
            if hasattr(result_item, 'score'):
                norm_score = result_item.score if result_item.score is not None else 0
            elif hasattr(result_item, 'metric'):
                norm_score = result_item.metric if result_item.metric is not None else 0
            
            if norm_score > 0:
                parsed_successfully = True
            
            original_score = norm_score * MAX_SCORE
            results['scores'].append(original_score)
            
            category = _categorize_score(original_score)
            _update_category_count(results, original_score)
            
            results['details'].append({
                'question': str(getattr(example, 'question', ''))[:100],
                'pred_sql': str(getattr(output, 'sql_query', ''))[:100] if output else '',
                'score': original_score,
                'category': category
            })
    
    # DSPy 평균 점수가 있고, 개별 파싱이 실패한 경우 DSPy 점수 사용
    if dspy_avg_score > 0 and not parsed_successfully:
        # DSPy가 보고한 평균 점수 사용 (0~1 범위)
        avg_score = dspy_avg_score * MAX_SCORE
        results['avg_score'] = avg_score
        results['avg_normalized'] = dspy_avg_score
        
        # 대략적인 분포 추정 (점수 기반)
        # 예: 36.8% → 약 17개 성공, 30개 실패
        total = len(examples)
        estimated_success = int(total * dspy_avg_score)
        estimated_success = max(0, min(estimated_success, total))  # 0~total 범위 제한
        
        results['partial_match'] = estimated_success
        results['error'] = max(0, total - estimated_success)  # 음수 방지
        results['scores'] = [avg_score] * total
        
        print(f"   ℹ️ DSPy 평균 점수 사용: {dspy_avg_score:.3f} ({dspy_avg_score*100:.1f}%)")
        return results
    
    # 개별 결과가 있으면 평균 계산
    if results['scores']:
        calculated_avg = sum(results['scores']) / len(results['scores'])
        # DSPy 점수가 더 신뢰할 수 있으면 사용
        if dspy_avg_score > 0 and abs(calculated_avg - dspy_avg_score * MAX_SCORE) > 0.5:
            results['avg_score'] = dspy_avg_score * MAX_SCORE
            results['avg_normalized'] = dspy_avg_score
        else:
            results['avg_score'] = calculated_avg
            results['avg_normalized'] = calculated_avg / MAX_SCORE
    else:
        # 점수 없으면 DSPy 점수 사용
        results['avg_score'] = dspy_avg_score * MAX_SCORE
        results['avg_normalized'] = dspy_avg_score
    
    return results


def _fallback_evaluation(
    module: dspy.Module,
    examples: List[dspy.Example],
    step_name: str = ""
) -> dict:
    """DSPy Evaluate 실패 시 수동 평가"""
    print("🔄 Fallback 평가 모드로 전환...")
    
    results = {
        'total': len(examples),
        'perfect_match': 0,
        'result_match': 0,
        'partial_match': 0,
        'wrong_result': 0,
        'error': 0,
        'scores': [],
        'details': [],
        'elapsed_time': 0
    }
    
    start_time = time.time()
    
    for i, example in enumerate(examples):
        try:
            output = module(
                question=example.question,
                table_schema=example.table_schema,
                hint=getattr(example, 'hint', '')
            )
            norm_score = normalized_metric(example, output)
        except Exception as e:
            print(f"  ⚠️ Example {i+1} 에러: {str(e)[:50]}")
            output = None
            norm_score = 0
        
        original_score = norm_score * MAX_SCORE
        results['scores'].append(original_score)
        
        _update_category_count(results, original_score)
        
        results['details'].append({
            'question': str(example.question)[:100],
            'pred_sql': str(getattr(output, 'sql_query', ''))[:100] if output else '',
            'score': original_score,
            'category': _categorize_score(original_score)
        })
    
    results['elapsed_time'] = time.time() - start_time
    results['avg_score'] = sum(results['scores']) / len(results['scores']) if results['scores'] else 0
    results['avg_normalized'] = results['avg_score'] / MAX_SCORE
    
    _print_dspy_evaluation_summary(results, step_name)
    return results


def _categorize_score(score: float) -> str:
    """점수를 카테고리로 분류"""
    if score >= 3.4:
        return "Perfect"
    elif score >= 2.9:
        return "Result Match"
    elif score >= 0.6:
        return "Partial"
    elif score >= 0.4:
        return "Wrong Result"
    else:
        return "Error"


def _update_category_count(results: dict, score: float):
    """카테고리별 카운트 업데이트"""
    if score >= 3.4:
        results['perfect_match'] += 1
    elif score >= 2.9:
        results['result_match'] += 1
    elif score >= 0.6:
        results['partial_match'] += 1
    elif score >= 0.4:
        results['wrong_result'] += 1
    else:
        results['error'] += 1


def _print_evaluation_summary(results: dict):
    """평가 결과 요약 출력"""
    total = results['total']
    print(f"\n{'='*50}")
    print(f"📊 평가 결과 요약")
    print(f"{'='*50}")
    print(f"총 샘플: {total}")
    print(f"✅ Perfect Match (3.5점): {results['perfect_match']} ({results['perfect_match']/total*100:.1f}%)")
    print(f"🟢 Result Match (3.0점): {results['result_match']} ({results['result_match']/total*100:.1f}%)")
    print(f"🟡 Partial Match (0.5 ~ 2.5점): {results['partial_match']} ({results['partial_match']/total*100:.1f}%)")
    print(f"🟠 Wrong Result (0.5점): {results['wrong_result']} ({results['wrong_result']/total*100:.1f}%)")
    print(f"❌ Error (0점): {results['error']} ({results['error']/total*100:.1f}%)")
    print(f"\n평균 점수: {results['avg_score']:.3f}")
    print(f"정규화 점수: {results['avg_normalized']:.3f}")


def _print_dspy_evaluation_summary(results: dict, step_name: str):
    """DSPy 평가 결과 요약 출력"""
    total = results['total']
    elapsed = results.get('elapsed_time', 0)
    
    print(f"\n{'='*50}")
    print(f"📊 DSPy Evaluate 결과 - {step_name}")
    print(f"{'='*50}")
    print(f"✅ Perfect Match (3.5): {results['perfect_match']} ({results['perfect_match']/total*100:.1f}%)")
    print(f"🟢 Result Match (3.0): {results['result_match']} ({results['result_match']/total*100:.1f}%)")
    print(f"🟡 Partial Match (0.5 ~ 2.5): {results['partial_match']} ({results['partial_match']/total*100:.1f}%)")
    print(f"🟠 Wrong Result (0.5): {results['wrong_result']} ({results['wrong_result']/total*100:.1f}%)")
    print(f"❌ Error (0.0): {results['error']} ({results['error']/total*100:.1f}%)")
    print(f"\n⏱️  실행 시간: {elapsed:.1f}초")
    print(f"🎯 평균 점수: {results['avg_score']:.3f}")
    print(f"📊 정규화 점수: {results['avg_normalized']:.3f}")


def compare_results(
    baseline_results: dict, 
    optimized_results: dict,
    show_improvement: bool = True
) -> dict:
    """베이스라인과 최적화 결과 비교"""
    comparison = {
        'baseline': baseline_results,
        'optimized': optimized_results,
        'improvement': {}
    }
    
    for key in ['avg_score', 'perfect_match', 'result_match', 'partial_match']:
        baseline_val = baseline_results.get(key, 0)
        optimized_val = optimized_results.get(key, 0)
        comparison['improvement'][key] = optimized_val - baseline_val
    
    if show_improvement:
        print("\n" + "="*60)
        print("📈 성능 비교: Baseline vs Optimized")
        print("="*60)
        print(f"평균 점수: {baseline_results['avg_score']:.3f} → {optimized_results['avg_score']:.3f} "
              f"({comparison['improvement']['avg_score']:+.3f})")
        print(f"Perfect Match: {baseline_results['perfect_match']} → {optimized_results['perfect_match']} "
              f"({comparison['improvement']['perfect_match']:+d})")
    
    return comparison


print("✅ 평가 함수 로드 완료")


import dspy
from dspy.evaluate import Evaluate
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
from typing import Optional, Tuple, Any, List, Dict
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from core.utiles import get_db_schema, execute_sql, compare_sql, compare_results, calculate_evidence_similarity
from core.logger import PerformanceTracker, StepMetrics
from core.logger import list_saved_results, load_and_display_history, display_latest_graph

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 결과 저장 경로
RESULTS_DIR = "/data/workspace/sogang/sqlagent/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
SQLITE_PATH = "/data/workspace/sogang/sqlagent/data/BIRD.sqlite"
SCHEMA_DIR = "/data/workspace/sogang/sqlagent/data"
ds = load_dataset("birdsql/bird_sql_dev_20251106")
df = ds['dev_20251106'].to_pandas()
hard_df = df[df['difficulty'] == 'challenging'].copy()
train_df, test_df = train_test_split(hard_df, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct"
api_key = "token-abc123"
model = dspy.LM(
    model=f"openai/{model_name}", 
    api_base="http://211.47.56.81:7972/v1", 
    api_key=api_key, 
    temperature=0.6,
    request_kwargs={
        "timeout": 60.0,
        "max_retries": 3,
        "max_tokens": 30000
    }
)
dspy.settings.configure(lm=model)
print(f"모델 설정 완료: {model_name}")

class TextToSQLSignature(dspy.Signature):
    """You are an expert SQL developer. Given a natural language question and database schema, 
    generate a correct SQL query that answers the question. 
    Think step by step about the table relationships and required columns."""
    
    # Input fields (MIPROv2가 이 Signature의 docstring을 System Prompt로 최적화함)
    question = dspy.InputField(desc="자연어로 된 데이터베이스 질의 질문")
    table_schema = dspy.InputField(desc="데이터베이스 테이블 스키마 (CREATE TABLE 문)")
    hint = dspy.InputField(desc="질문 해석을 위한 힌트/증거", default="")
    
    # Output fields
    reasoning = dspy.OutputField(desc="SQL 쿼리를 작성하기 위한 단계별 추론 과정")
    sql_query = dspy.OutputField(desc="최종 SQL 쿼리 (SELECT 문)")
    evidence = dspy.OutputField(desc="쿼리 결과를 설명하는 증거 문장")


class TextToSQLModule(dspy.Module):
    """Text-to-SQL 변환 모듈 (최적화 대상)"""
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(TextToSQLSignature)
    
    def forward(self, question: str, table_schema: str, hint: str = "") -> dspy.Prediction:
        prediction = self.cot(
            question=question,
            table_schema=table_schema,
            hint=hint
        )
        return dspy.Prediction(
            reasoning=prediction.reasoning,
            sql_query=prediction.sql_query,
            evidence=prediction.evidence
        )


print("✅ Signature & Module 정의 완료")
print("   📝 MIPROv2는 Signature의 docstring을 System Prompt로 최적화합니다.")
# ============================================
# 7. DSPy Example 데이터셋 준비
# ============================================

def create_dspy_examples(df: pd.DataFrame) -> List[dspy.Example]:
    examples = []
    schema = get_db_schema("bird")    
    for _, row in df.iterrows():
        example = dspy.Example(
            question=row['question'],
            table_schema=schema,
            hint=row['evidence'] if pd.notna(row['evidence']) else "",
            gold_sql=row['SQL'],
            gold_evidence=row['evidence'] if pd.notna(row['evidence']) else "",
        ).with_inputs('question', 'table_schema', 'hint')
        examples.append(example)
    return examples


train_examples = create_dspy_examples(train_df)
test_examples = create_dspy_examples(test_df)

print(f"Train examples: {len(train_examples)}")
print(f"Test examples: {len(test_examples)}")
print(f"\n샘플 example:")
print(f"Question: {train_examples[0].question[:100]}...")
print(f"Gold SQL: {train_examples[0].gold_sql[:100]}...")
# ============================================
# 6. 보상 체계 기반 커스텀 메트릭 함수
# ============================================
"""
보상 체계:
- Query와 DB 조회 결과가 완벽히 일치: 2.0점
- Query는 다르지만 DB 결과가 동일: 1.5점
- Query/결과 다르지만 evidence 유사도 기반: 0.0 ~ 1.0점
- 실행은 되지만 결과가 틀림: -0.5점
- 에러 발생: -1.0점
"""

def text_to_sql_metric(example, prediction, trace=None) -> float:
    """
    보상 체계:
    - Query + 결과 완벽 일치: 3.5점
    - Query 다르지만 결과 동일: 3.0점
    - Evidence 유사도 기반: 0.5 ~ 2.5점
    - 실행되지만 결과 틀림: 0.5점
    - 에러 발생: 0.0점
    """
    gold_sql = example.gold_sql
    gold_evidence = example.gold_evidence if hasattr(example, 'gold_evidence') else ""
    pred_sql = prediction.sql_query if hasattr(prediction, 'sql_query') else ""
    pred_evidence = prediction.evidence if hasattr(prediction, 'evidence') else ""
    
    # 1. 예측된 SQL 실행
    pred_success, pred_result = execute_sql(pred_sql)
    
    # 에러 발생 시 0.0점
    if not pred_success:
        return 0.0
    
    # 2. Gold SQL 실행
    gold_success, gold_result = execute_sql(gold_sql)
    if not gold_success:
        return 0.5  # Gold SQL 에러지만 예측은 실행됨
    
    sql_match = compare_sql(pred_sql, gold_sql)
    result_match = compare_results(pred_result, gold_result)
    
    # 3. 보상 계산
    if sql_match and result_match:
        # Query + 결과 완벽 일치
        return 3.5
    elif result_match:
        # Query는 다르지만 결과 동일
        return 3.0
    else:
        # 결과가 다름 - evidence 유사도로 부분 점수 (0.5 ~ 2.5)
        evidence_sim = calculate_evidence_similarity(pred_evidence, gold_evidence)
        if evidence_sim > 0.1:
            # evidence_sim (0~1) → 0.5 ~ 2.5로 스케일링
            return 0.5 + evidence_sim * 2.0
        else:
            # 실행되지만 결과 틀림
            return 0.5

def normalized_metric(example, prediction, trace=None) -> float:
    """
    정규화된 메트릭 (0~1 범위로 변환) - DSPy 최적화용
    
    DSPy의 COPRO/MIPROv2/Evaluate는 0~1 범위의 메트릭을 기대합니다.
    0~3.5 점수를 0~1로 정규화합니다.
    """
    score = text_to_sql_metric(example, prediction, trace)
    # 0~3.5 → 0~1 정규화
    return score / 3.5


print("메트릭 함수 정의 완료")
print("   - text_to_sql_metric(): 원본 점수 (0 ~ 3.5)")
print("   - normalized_metric(): DSPy 최적화용 정규화 (0 ~ 1)")
# ============================================
# 8. 평가 함수
# ============================================

def evaluate_model(module: dspy.Module, examples: List[dspy.Example], 
                   verbose: bool = True) -> dict:
    """모델 평가 및 상세 결과 반환"""
    
    results = {
        'total': len(examples),
        'perfect_match': 0,      # 3.5점
        'result_match': 0,       # 3점
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
                hint=example.hint
            )
            
            score = text_to_sql_metric(example, prediction)
            results['scores'].append(score)
            
            # 새로운 보상 체계 (0 ~ 3.5) 기준 카테고리 분류
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
            
            if verbose and i < 5:  # 처음 5개만 출력
                print(f"\n[{i+1}] {category}")
                print(f"Q: {example.question[:80]}...")
                print(f"Pred SQL: {prediction.sql_query[:80]}...")
                print(f"Score: {score}")
            
            results['details'].append({
                'question': example.question,
                'pred_sql': prediction.sql_query,
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
    results['avg_normalized'] = results['avg_score'] / 3.5  # 0~3.5 → 0~1 정규화
    
    print(f"\n{'='*50}")
    print(f"📊 평가 결과 요약")
    print(f"{'='*50}")
    print(f"총 샘플: {results['total']}")
    print(f"✅ Perfect Match (3.5점): {results['perfect_match']} ({results['perfect_match']/results['total']*100:.1f}%)")
    print(f"🟢 Result Match (3점): {results['result_match']} ({results['result_match']/results['total']*100:.1f}%)")
    print(f"🟡 Partial Match (0.5 ~ 2.5점): {results['partial_match']} ({results['partial_match']/results['total']*100:.1f}%)")
    print(f"🟠 Wrong Result (0.5점): {results['wrong_result']} ({results['wrong_result']/results['total']*100:.1f}%)")
    print(f"❌ Error (0점): {results['error']} ({results['error']/results['total']*100:.1f}%)")
    print(f"\n평균 점수: {results['avg_score']:.3f}")
    print(f"정규화 점수: {results['avg_normalized']:.3f}")
    
    return results


print("평가 함수 정의 완료")
# ============================================
# 9. DSPy 최적화 파이프라인 (COPRO / MIPROv2)
# ============================================
"""
🔥 두 가지 최적화 방법:

1️⃣ COPRO (Collaborative Prompt Optimization)
   - Signature의 설명문(instruction)을 자동 개선
   - 용도: 설명문만 최적화하고자 할 때

2️⃣ MIPROv2 (Mixed Instruction and PRompt Optimization v2)
   - 베이지안 최적화로 설명문 + Few-shot 예제 모두 최적화
   - 용도: Zero-shot으로 시작하거나 예제가 200개 이상인 경우
"""

def optimize_with_copro(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    metric_fn,
    breadth: int = 10,
    depth: int = 1,
    init_temperature: float = 0.6,
    num_threads: int = 20
) -> dspy.Module:
    """
    COPRO로 최적화 - Signature 설명문(Instruction)만 자동 개선
    
    COPRO가 최적화하는 것:
    ✅ Instruction (System Prompt) - 설명문 자동 개선
    Args:
        breadth: 각 단계에서 생성할 후보 수
        depth: 최적화 반복 횟수
        init_temperature: 초기 temperature (다양성 조절)
    """
    try:
        from dspy.teleprompt import COPRO
        
        print("🚀 COPRO 최적화 시작...")
        print("   📝 최적화 대상:")
        print("      ✅ Signature 설명문 (Instruction)")
        print("      ❌ Few-shot demonstrations (최적화 안함)")
        print(f"   🔧 설정: breadth={breadth}, depth={depth}")
        
        optimizer = COPRO(
            metric=metric_fn,
            breadth=breadth,
            depth=depth,
            init_temperature=init_temperature,
            verbose=True
        )
        
        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            eval_kwargs=dict(num_threads=num_threads, display_progress=True)
        )
        
        # 최적화된 프롬프트 출력
        print("\n" + "="*60)
        print("🎯 COPRO 최적화된 Instruction 확인")
        print("="*60)
        display_optimized_prompt(optimized_module)
        
        print("\n✅ COPRO 최적화 완료!")
        return optimized_module
        
    except ImportError as e:
        print(f"⚠️ COPRO를 사용할 수 없습니다: {e}")
        raise
    except Exception as e:
        print(f"⚠️ COPRO 오류: {e}")
        raise


def optimize_with_mipro(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    metric_fn,
    num_candidates: int = 10,
    init_temperature: float = 1.4,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 4,
    num_threads: int = 20
) -> dspy.Module:
    """
    MIPROv2로 최적화 - 베이지안 최적화로 설명문 + Few-shot 모두 최적화
    
    MIPROv2가 최적화하는 것:
    ✅ Instruction (System Prompt) - 자연어 지시문 자동 생성/최적화
    ✅ Few-shot demonstrations - 최적의 예시 선택
    
    권장 사용 시점:
    - Zero-shot으로 시작할 때
    - 예제가 200개 이상인 경우
    
    Args:
        num_candidates: 베이지안 최적화 후보 수
        init_temperature: 초기 temperature
        max_bootstrapped_demos: 부트스트랩 예시 수
        max_labeled_demos: 라벨링된 예시 수
    """
    try:
        from dspy.teleprompt import MIPROv2
        
        print("🚀 MIPROv2 최적화 시작...")
        print("   📝 최적화 대상:")
        print("      ✅ System Prompt (Instruction) - 베이지안 최적화")
        print("      ✅ Few-shot demonstrations")
        print(f"   🔧 설정: candidates={num_candidates}, demos={max_bootstrapped_demos}+{max_labeled_demos}")
        
        optimizer = MIPROv2(
            metric=metric_fn,
            auto=None,  # 수동 설정 모드 (num_candidates/num_trials 사용 시 필수)
            num_candidates=num_candidates,
            init_temperature=init_temperature,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_threads=num_threads,
            verbose=True
        )
        
        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            num_trials=num_candidates  # auto=None일 때 num_trials 필수
        )
        
        # 최적화된 프롬프트 출력
        print("\n" + "="*60)
        print("🎯 MIPROv2 최적화된 프롬프트 확인")
        print("="*60)
        display_optimized_prompt(optimized_module)
        
        print("\n✅ MIPROv2 최적화 완료!")
        return optimized_module
        
    except ImportError as e:
        print(f"⚠️ MIPROv2를 사용할 수 없습니다: {e}")
        raise
    except Exception as e:
        print(f"⚠️ MIPROv2 오류: {e}")
        raise


def display_optimized_prompt(module: dspy.Module):
    """최적화된 모듈의 프롬프트 출력"""
    try:
        # ChainOfThought 내부의 predict 모듈 접근
        if hasattr(module, 'cot'):
            predictor = module.cot
        elif hasattr(module, 'predict'):
            predictor = module.predict
        else:
            predictor = module
        
        # Extended Signature에서 instructions 추출
        if hasattr(predictor, 'extended_signature'):
            sig = predictor.extended_signature
            if hasattr(sig, 'instructions'):
                print(f"\n📋 최적화된 Instruction:")
                print("-" * 50)
                print(sig.instructions)
                print("-" * 50)
        
        # Signature에서 직접 추출 시도
        if hasattr(predictor, 'signature'):
            sig = predictor.signature
            if hasattr(sig, 'instructions'):
                print(f"\n📋 Signature Instruction:")
                print("-" * 50)
                print(sig.instructions)
                print("-" * 50)
        
        # Demos (Few-shot examples) 출력
        if hasattr(predictor, 'demos') and predictor.demos:
            print(f"\n📚 최적화된 Few-shot Demos ({len(predictor.demos)}개):")
            for i, demo in enumerate(predictor.demos[:3]):  # 처음 3개만
                print(f"\n  [Demo {i+1}]")
                if hasattr(demo, 'question'):
                    print(f"  Q: {str(demo.question)[:100]}...")
                if hasattr(demo, 'sql_query'):
                    print(f"  SQL: {str(demo.sql_query)[:100]}...")
        else:
            print("\n📚 Few-shot Demos: 없음 (Zero-shot)")
        
    except Exception as e:
        print(f"프롬프트 출력 오류: {e}")


def compare_prompts(baseline_module: dspy.Module, optimized_module: dspy.Module):
    """베이스라인과 최적화된 모듈의 프롬프트 비교"""
    print("\n" + "="*60)
    print("📊 프롬프트 비교: Baseline vs Optimized")
    print("="*60)
    
    print("\n🔵 [Baseline]")
    display_optimized_prompt(baseline_module)
    
    print("\n🟢 [Optimized]")
    display_optimized_prompt(optimized_module)


print("✅ 최적화 함수 정의 완료")
print("=" * 50)
print("🔧 사용 가능한 최적화 방법:")
print("   1️⃣ optimize_with_copro()  - Instruction만 최적화")
print("   2️⃣ optimize_with_mipro()  - Instruction + Few-shot 모두 최적화 ⭐")
print("=" * 50)
# ============================================
# 9-1. DSPy Evaluate 기반 평가 함수 (트래킹 통합)
# ============================================

def evaluate_with_dspy(
    module: dspy.Module, 
    examples: List[dspy.Example],
    tracker: PerformanceTracker = None,
    step_name: str = "",
    num_threads: int = 1,
    display_progress: bool = True
) -> dict:
    """
    DSPy Evaluate를 사용한 평가 + 트래킹 (새로운 API 호환)
    
    Args:
        module: 평가할 DSPy 모듈
        examples: 평가 데이터셋
        tracker: PerformanceTracker 인스턴스
        step_name: 스텝 이름
        num_threads: 병렬 처리 스레드 수
        display_progress: 진행 상황 표시 여부
    
    Returns:
        dict: 평가 결과
    """
    start_time = time.time()
    
    # DSPy Evaluate 생성 (normalized_metric 사용 - 0~1 범위)
    evaluator = Evaluate(
        devset=examples,
        metric=normalized_metric,  # 정규화된 메트릭 사용!
        num_threads=num_threads,
        display_progress=display_progress
    )
    
    # 평가 실행 - 새로운 API는 EvaluationResult 객체 반환
    eval_result = evaluator(module)
    
    elapsed = time.time() - start_time
    
    # EvaluationResult에서 결과 추출
    # 새로운 API: eval_result.score (평균), eval_result.results (상세)
    if hasattr(eval_result, 'score'):
        avg_score = eval_result.score
    else:
        avg_score = eval_result if isinstance(eval_result, (int, float)) else 0
    
    # 상세 결과 분석
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
    
    # 새로운 API에서 개별 결과 추출
    if hasattr(eval_result, 'results') and eval_result.results:
        for i, result_item in enumerate(eval_result.results):
            example = result_item.example if hasattr(result_item, 'example') else examples[i]
            output = result_item.prediction if hasattr(result_item, 'prediction') else None
            norm_score = result_item.score if hasattr(result_item, 'score') else 0
            
            # 정규화된 점수(0~1)를 원본 점수(0~3.5)로 복원
            original_score = norm_score * 3.5
            results['scores'].append(original_score)
            
            # 카테고리 분류
            # 새로운 보상 체계 (0 ~ 3.5) 기준 카테고리 분류
            if original_score >= 3.4:
                results['perfect_match'] += 1
                category = "Perfect"
            elif original_score >= 2.9:
                results['result_match'] += 1
                category = "Result Match"
            elif original_score >= 0.6:
                results['partial_match'] += 1
                category = "Partial"
            elif original_score >= 0.4:
                results['wrong_result'] += 1
                category = "Wrong Result"
            else:
                results['error'] += 1
                category = "Error"
            
            results['details'].append({
                'question': str(getattr(example, 'question', ''))[:100],
                'pred_sql': str(getattr(output, 'sql_query', ''))[:100] if output else '',
                'score': original_score,
                'category': category
            })
    else:
        # fallback: 수동 평가
        for i, example in enumerate(examples):
            try:
                output = module(
                    question=example.question,
                    table_schema=example.table_schema,
                    hint=getattr(example, 'hint', '')
                )
                norm_score = normalized_metric(example, output)  # 정규화된 메트릭 사용
            except Exception as e:
                output = None
                norm_score = 0
            
            # 정규화된 점수(0~1)를 원본 점수(0~3.5)로 복원
            original_score = norm_score * 3.5
            results['scores'].append(original_score)
            
            # 새로운 보상 체계 (0 ~ 3.5) 기준 카테고리 분류
            if original_score >= 3.4:
                results['perfect_match'] += 1
                category = "Perfect"
            elif original_score >= 2.9:
                results['result_match'] += 1
                category = "Result Match"
            elif original_score >= 0.6:
                results['partial_match'] += 1
                category = "Partial"
            elif original_score >= 0.4:
                results['wrong_result'] += 1
                category = "Wrong Result"
            else:
                results['error'] += 1
                category = "Error"
            
            results['details'].append({
                'question': str(example.question)[:100],
                'pred_sql': str(getattr(output, 'sql_query', ''))[:100] if output else '',
                'score': original_score,
                'category': category
            })
    
    results['avg_score'] = sum(results['scores']) / len(results['scores']) if results['scores'] else 0
    results['avg_normalized'] = results['avg_score'] / 3.5  # 0~3.5 → 0~1 정규화
    results['elapsed_time'] = elapsed
    
    # 트래킹 로그
    if tracker:
        tracker.log_step(results, step_name)
    
    # 결과 출력
    print(f"\n{'='*50}")
    print(f"📊 DSPy Evaluate 결과 - {step_name}")
    print(f"{'='*50}")
    print(f"✅ Perfect Match (3.5): {results['perfect_match']} ({results['perfect_match']/results['total']*100:.1f}%)")
    print(f"🟢 Result Match (3.0): {results['result_match']} ({results['result_match']/results['total']*100:.1f}%)")
    print(f"🟡 Partial Match (0.5 ~ 2.5): {results['partial_match']} ({results['partial_match']/results['total']*100:.1f}%)")
    print(f"🟠 Wrong Result (0.5): {results['wrong_result']} ({results['wrong_result']/results['total']*100:.1f}%)")
    print(f"❌ Error (0.0): {results['error']} ({results['error']/results['total']*100:.1f}%)")
    print(f"\n⏱️  실행 시간: {elapsed:.1f}초")
    print(f"🎯 평균 점수: {results['avg_score']:.3f}")
    print(f"📊 정규화 점수: {results['avg_normalized']:.3f}")
    
    return results


def run_optimization_with_tracking(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    test_examples: List[dspy.Example],
    tracker: PerformanceTracker,
    optimizer_type: str = "mipro",  # 기본값을 mipro로 변경!
    save_intermediate: bool = True,
    breadth: int = 10,
    depth: int = 3,
    init_temperature: float = 0.6,
    num_threads: int = 10,
) -> Tuple[dspy.Module, dict]:
    """
    최적화 과정을 트래킹하며 실행
    
    Args:
        module: 최적화할 모듈
        train_examples: 학습 데이터
        test_examples: 테스트 데이터
        tracker: PerformanceTracker
        optimizer_type: "mipro" (System Prompt 최적화) 또는 "bootstrap" (Few-shot만)
        save_intermediate: 중간 결과 저장 여부
    
    Returns:
        (optimized_module, final_results)
    """
    # 베이스라인 모듈 저장 (나중에 비교용)
    baseline_module = module
    
    tracker.start()
    
    # Step 1: 베이스라인 평가
    print("\n" + "🔵"*25)
    print("Step 1: 베이스라인 평가")
    print("🔵"*25)
    baseline_results = evaluate_with_dspy(
        module, test_examples, tracker, "Baseline"
    )
    
    if save_intermediate:
        tracker.plot_metrics()
    
    # Step 2: 최적화 실행
    print("\n" + "🟡"*25)
    print(f"Step 2: 최적화 실행 ({optimizer_type.upper()})")
    print("🟡"*25)
    
    optimization_start = time.time()
    
    if optimizer_type == "copro":
        # COPRO: Instruction(설명문)만 최적화
        optimized_module = optimize_with_copro(
            module=module,
            train_examples=train_examples,
            metric_fn=normalized_metric,
            breadth=breadth,
            depth=depth,
            num_threads=num_threads,
            init_temperature=init_temperature
        )
    elif optimizer_type == "mipro":
        # MIPROv2: Instruction + Few-shot 모두 최적화 (베이지안)
        optimized_module = optimize_with_mipro(
            module=module,
            train_examples=train_examples,
            metric_fn=normalized_metric,
            num_candidates=10,
            init_temperature=1.4,
            max_bootstrapped_demos=3,
            max_labeled_demos=4,
            num_threads=20
        )
    else:
        raise ValueError(f"알 수 없는 optimizer_type: {optimizer_type}. 'copro' 또는 'mipro'를 사용하세요.")
    
    optimization_time = time.time() - optimization_start
    print(f"\n⏱️  최적화 시간: {optimization_time:.1f}초")
    
    # Step 3: 최적화 후 평가
    print("\n" + "🟢"*25)
    print("Step 3: 최적화 후 평가")
    print("🟢"*25)
    optimized_results = evaluate_with_dspy(
        optimized_module, test_examples, tracker, "Optimized"
    )
    
    # Step 4: 프롬프트 비교
    print("\n" + "🔮"*25)
    print("Step 4: 프롬프트 변화 확인")
    print("🔮"*25)
    compare_prompts(baseline_module, optimized_module)
    # 최종 그래프 저장
    tracker.plot_metrics()
    tracker.save_history()
    tracker.summary()
    return optimized_module, optimized_results


print("✅ DSPy Evaluate 기반 평가 함수 정의 완료")

if __name__ == "__main__":
        # ============================================
    # 10-1. 트래킹이 포함된 전체 파이프라인 실행
    # ============================================
    """
    🔧 최적화 방법 선택:
    - "copro": Instruction(설명문)만 최적화
    - "mipro": Instruction + Few-shot 모두 최적화 (베이지안)
    """

    # ⚙️ 설정
    OPTIMIZER_TYPE = "copro"  # "copro" 또는 "mipro" 선택!
    TRAIN_SIZE = 5           # 학습 데이터 수
    TEST_SIZE = 10            # 테스트 데이터 수

    # 새 트래커 생성
    tracker = PerformanceTracker(f"bird_text2sql_{OPTIMIZER_TYPE}_optimization")

    # 새 모듈 생성
    module_for_optimization = TextToSQLModule()
    print(f"🚀 최적화 방법: {OPTIMIZER_TYPE.upper()}")
    # print(f"📊 학습 데이터: {TRAIN_SIZE}개, 테스트 데이터: {TEST_SIZE}개")
    print("="*50)

    optimized_module, final_results = run_optimization_with_tracking(
        module=module_for_optimization,
        train_examples=train_examples,
        test_examples=test_examples,
        tracker=tracker,
        optimizer_type=OPTIMIZER_TYPE,
        save_intermediate=True,
        breadth=3,
        depth=1,
        init_temperature=0.6,
        num_threads=10
    )

    print("\n✅ 전체 파이프라인 완료!")
    print(f"📁 결과 저장 경로: {RESULTS_DIR}")

    # ============================================
    # 10. 베이스라인 평가 (최적화 전)
    # ============================================

    # 기본 모듈 생성
    baseline_module = TextToSQLModule()

    # 테스트 샘플 몇 개로 베이스라인 평가
    print("📝 베이스라인 모델 평가 (최적화 전)")
    print("=" * 50)

    # 처음 10개만 빠른 테스트
    baseline_results = evaluate_model(
        baseline_module, 
        test_examples[:5], 
        verbose=True
    )

    print("📝 최적화된 모델 평가")
    print("=" * 50)
    optimized_results = evaluate_model(
        optimized_module, 
        test_examples[:5], 
        verbose=True
    )

    improvement = optimized_results['avg_score'] - baseline_results['avg_score']
    print(f"\n📈 개선율: {improvement:+.3f} 점")
    print(f"베이스라인: {baseline_results['avg_score']:.3f} → 최적화: {optimized_results['avg_score']:.3f}")
    import json

    SAVE_PATH = "/data/workspace/sogang/sqlagent/optimized_text2sql_model.json"

    def save_optimized_module(module: dspy.Module, path: str):
        """최적화된 모듈 저장"""
        module.save(path)
        print(f"✅ 모델 저장 완료: {path}")


    def load_optimized_module(path: str) -> dspy.Module:
        """저장된 모듈 로드"""
        module = TextToSQLModule()
        module.load(path)
        print(f"✅ 모델 로드 완료: {path}")
        return module


    # 최적화된 모델 저장
    save_optimized_module(optimized_module, SAVE_PATH)

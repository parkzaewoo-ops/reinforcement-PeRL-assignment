"""
DSPy Optimizer Functions
========================
COPRO, MIPROv2 등 DSPy 최적화 함수들을 정의합니다.

🔥 두 가지 최적화 방법:
1️⃣ COPRO (Collaborative Prompt Optimization)
   - Signature의 설명문(instruction)을 자동 개선
   - 용도: 설명문만 최적화하고자 할 때

2️⃣ MIPROv2 (Mixed Instruction and PRompt Optimization v2)
   - 베이지안 최적화로 설명문 + Few-shot 예제 모두 최적화
   - 용도: Zero-shot으로 시작하거나 예제가 200개 이상인 경우

🆕 Early Stopping & Step Logging:
   - 각 최적화 단계마다 성능 로깅
   - 성능이 향상되지 않으면 early stopping
   - 가장 좋은 프롬프트 자동 보존
"""

import time
import copy
import json
import os
import dspy
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict, Any
from .metric import normalized_metric, MAX_SCORE
from .evaluation import evaluate_with_dspy
from .logger import PerformanceTracker
from .config import OptimizerConfig, RESULTS_DIR


# ============================================
# Optimization State & Logging
# ============================================

@dataclass
class OptimizationStep:
    """단일 최적화 단계 정보"""
    step: int
    score: float
    normalized_score: float
    instruction: str
    num_demos: int
    timestamp: str
    is_best: bool = False
    improvement: float = 0.0


@dataclass
class OptimizationState:
    """최적화 상태 관리"""
    best_score: float = 0.0
    best_module: Optional[dspy.Module] = None
    best_step: int = 0
    steps_without_improvement: int = 0
    history: List[OptimizationStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    # 자동 저장 설정
    save_dir: str = field(default_factory=lambda: RESULTS_DIR)
    experiment_name: str = "optimization"
    auto_save: bool = True
    
    def update(self, step: int, score: float, module: dspy.Module, instruction: str = "", num_demos: int = 0) -> bool:
        """
        상태 업데이트 및 개선 여부 반환
        
        Returns:
            True if improved, False otherwise
        """
        normalized = score / MAX_SCORE if score > 1 else score
        improvement = score - self.best_score
        # 첫 번째 업데이트이거나 점수가 개선되면 best로 설정
        is_best = (self.best_module is None) or (score > self.best_score)
        
        step_info = OptimizationStep(
            step=step,
            score=score,
            normalized_score=normalized,
            instruction=instruction[:200] + "..." if len(instruction) > 200 else instruction,
            num_demos=num_demos,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            is_best=is_best,
            improvement=improvement
        )
        self.history.append(step_info)
        
        if is_best:
            self.best_score = score
            self.best_module = copy.deepcopy(module)
            self.best_step = step
            self.steps_without_improvement = 0
            
            # Best 모델 자동 저장
            if self.auto_save and self.best_module is not None:
                self._save_best_module(step)
            
            return True
        else:
            self.steps_without_improvement += 1
            return False
    
    def _save_best_module(self, step: int):
        """Best 모델 저장"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"best_model_step{step}_{timestamp}.json"
            save_path = os.path.join(self.save_dir, filename)
            
            self.best_module.save(save_path)
            print(f"   💾 Best 모델 저장: {filename} (score: {self.best_score:.3f})")
        except Exception as e:
            print(f"   ⚠️ 모델 저장 실패: {e}")
    
    def should_stop(self, patience: int) -> bool:
        """Early stopping 조건 확인"""
        return self.steps_without_improvement >= patience
    
    def get_elapsed_time(self) -> float:
        """경과 시간 (초)"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "best_score": self.best_score,
            "best_step": self.best_step,
            "total_steps": len(self.history),
            "elapsed_time": self.get_elapsed_time(),
            "history": [
                {
                    "step": s.step,
                    "score": s.score,
                    "normalized_score": s.normalized_score,
                    "instruction": s.instruction,
                    "num_demos": s.num_demos,
                    "timestamp": s.timestamp,
                    "is_best": s.is_best,
                    "improvement": s.improvement
                }
                for s in self.history
            ]
        }
    
    def save(self, path: str = None):
        """상태를 JSON으로 저장"""
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(RESULTS_DIR, f"optimization_state_{timestamp}.json")
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"💾 최적화 상태 저장: {path}")
        return path


class OptimizationLogger:
    """최적화 과정 로거"""
    
    def __init__(self, experiment_name: str = "optimization"):
        self.experiment_name = experiment_name
        self.logs: List[Dict] = []
        self.previous_instruction: str = ""  # 이전 instruction 저장
        self.original_instruction: str = ""  # 최초 instruction 저장
    
    def set_original_instruction(self, instruction: str):
        """원본 instruction 저장 (최적화 시작 시 호출)"""
        self.original_instruction = instruction
        self.previous_instruction = instruction
        print(f"\n{'📝'*25}")
        print("📋 원본 Instruction:")
        print(f"{'📝'*25}")
        print(f"{instruction}")
        print(f"{'📝'*25}\n")
    
    def log_instruction_change(self, step: int, new_instruction: str):
        """Instruction 변경 확인 및 로깅"""
        changed = new_instruction != self.previous_instruction
        
        print(f"\n{'🔍'*25}")
        print(f"📝 Step {step} Instruction 변경 확인")
        print(f"{'🔍'*25}")
        
        if changed:
            print(f"✅ Instruction 변경됨!")
            print(f"\n[이전]")
            print(f"{self.previous_instruction[:300]}{'...' if len(self.previous_instruction) > 300 else ''}")
            print(f"\n[현재]")
            print(f"{new_instruction[:300]}{'...' if len(new_instruction) > 300 else ''}")
            
            # 원본과 비교
            if new_instruction != self.original_instruction:
                print(f"\n📊 원본 대비 변경: ✅ 예")
            else:
                print(f"\n📊 원본 대비 변경: ❌ 아니오 (원본으로 복귀)")
        else:
            print(f"❌ Instruction 변경 없음 (이전과 동일)")
        
        print(f"{'🔍'*25}\n")
        
        self.previous_instruction = new_instruction
        return changed
    
    def log_final_instruction_comparison(self, final_instruction: str):
        """최종 instruction과 원본 비교"""
        print(f"\n{'🎯'*25}")
        print("📋 Instruction 최종 비교")
        print(f"{'🎯'*25}")
        
        if final_instruction != self.original_instruction:
            print(f"✅ Instruction이 최적화됨!")
            print(f"\n[원본]")
            print(f"{self.original_instruction}")
            print(f"\n[최적화됨]")
            print(f"{final_instruction}")
        else:
            print(f"⚠️ Instruction이 변경되지 않음 (원본과 동일)")
            print(f"\n[Instruction]")
            print(f"{final_instruction}")
        
        print(f"{'🎯'*25}\n")
    
    def log_step(self, step: int, score: float, is_best: bool, details: Dict = None):
        """단계별 로깅"""
        log_entry = {
            "step": step,
            "score": score,
            "is_best": is_best,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **(details or {})
        }
        self.logs.append(log_entry)
        
        # 콘솔 출력
        best_marker = "⭐ BEST" if is_best else ""
        print(f"\n{'='*60}")
        print(f"📊 Step {step} 결과 {best_marker}")
        print(f"{'='*60}")
        print(f"   점수: {score:.4f}")
        
        if details:
            if details.get('num_demos'):
                print(f"   Demos: {details['num_demos']}개")
            if details.get('improvement'):
                print(f"   개선: {details['improvement']:+.4f}")
    
    def log_early_stop(self, step: int, patience: int, best_step: int):
        """Early stopping 로깅"""
        print(f"\n{'🛑'*20}")
        print(f"⚠️  Early Stopping at Step {step}")
        print(f"   {patience}회 연속 개선 없음")
        print(f"   Best Step: {best_step}")
        print(f"{'🛑'*20}")
    
    def log_final(self, state: OptimizationState):
        """최종 결과 로깅"""
        elapsed = state.get_elapsed_time()
        
        print(f"\n{'🎯'*20}")
        print(f"✅ 최적화 완료")
        print(f"{'🎯'*20}")
        print(f"   총 단계: {len(state.history)}")
        print(f"   Best Step: {state.best_step}")
        print(f"   Best Score: {state.best_score:.4f}")
        print(f"   소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        
        # 단계별 점수 변화 시각화
        print(f"\n📈 점수 변화:")
        for s in state.history:
            bar = "█" * int(s.score * 10)
            best_mark = " ⭐" if s.is_best else ""
            print(f"   Step {s.step}: {bar} {s.score:.3f}{best_mark}")


# ============================================
# Optimizer with Early Stopping
# ============================================

def optimize_with_copro_and_logging(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    test_examples: List[dspy.Example],
    metric_fn: Callable = None,
    breadth: int = 10,
    depth: int = 3,
    init_temperature: float = 0.9,
    num_threads: int = 16,
    patience: int = 2,
    min_improvement: float = 0.01,
    tracker: 'PerformanceTracker' = None
) -> Tuple[dspy.Module, OptimizationState]:
    """
    COPRO 최적화 + 단계별 로깅 + Early Stopping
    
    Args:
        module: 최적화할 DSPy 모듈
        train_examples: 학습 데이터
        test_examples: 테스트 데이터 (각 단계 평가용)
        metric_fn: 메트릭 함수
        breadth: 각 단계에서 생성할 후보 수
        depth: 최대 최적화 반복 횟수
        init_temperature: 초기 temperature
        num_threads: 병렬 처리 스레드 수
        patience: early stopping patience (연속 N회 개선 없으면 중단)
        min_improvement: 최소 개선 임계값
        tracker: PerformanceTracker (그래프/CSV 저장용)
    
    Returns:
        (best_module, optimization_state)
    """
    try:
        from dspy.teleprompt import COPRO
        
        if metric_fn is None:
            metric_fn = normalized_metric
        
        state = OptimizationState(
            experiment_name="copro",
            auto_save=True
        )
        logger = OptimizationLogger("copro")
        
        # breadth 최소값 검증
        if breadth < 2:
            print(f"⚠️ breadth={breadth}는 너무 작습니다. 최소값 2로 설정합니다.")
            breadth = 2
        
        print("🚀 COPRO 최적화 시작 (Early Stopping 활성화)")
        print("   📝 최적화 대상: Signature 설명문 (Instruction)")
        print(f"   🔧 설정: breadth={breadth}, depth={depth}, patience={patience}")
        print("="*60)
        
        # 원본 instruction 저장 및 출력
        original_instruction = _get_instruction(module)
        logger.set_original_instruction(original_instruction)
        
        # 초기 평가 (베이스라인)
        print("\n📊 Depth 0: 베이스라인 평가")
        baseline_results = evaluate_with_dspy(module, test_examples, None, "Baseline (Test)", num_threads=num_threads)
        baseline_score = baseline_results['avg_score']
        
        # Baseline에는 Train 점수 없음 (아직 최적화 전)
        baseline_results['train_score'] = 0
        baseline_results['train_normalized'] = 0
        
        # Tracker에 수동 로깅
        if tracker:
            tracker.log_step(baseline_results, "Baseline (Test)")
        
        instruction = _get_instruction(module)
        state.update(0, baseline_score, module, instruction, 0)
        logger.log_step(0, baseline_score, True, {"instruction": instruction})
        
        # 그래프 저장
        if tracker:
            tracker.plot_metrics()
        
        current_module = copy.deepcopy(module)
        
        # 단계별 최적화 (Depth)
        for step in range(1, depth + 1):
            print(f"\n{'🔄'*20}")
            print(f"Depth {step}/{depth}: COPRO 최적화 중...")
            print(f"{'🔄'*20}")
            
            step_start = time.time()
            
            # COPRO 1단계 실행 (track_stats=True로 점수 캡처)
            optimizer = COPRO(
                metric=metric_fn,
                breadth=breadth,
                depth=1,  # 한 단계씩만
                init_temperature=init_temperature,
                verbose=True,
                track_stats=True  # 점수 추적 활성화!
            )
            
            try:
                optimized_module = optimizer.compile(
                    current_module,
                    trainset=train_examples,
                    eval_kwargs=dict(num_threads=num_threads, display_progress=True)
                )
            except Exception as e:
                print(f"⚠️ Step {step} 최적화 실패: {e}")
                continue
            
            step_time = time.time() - step_start
            
            # Train 점수: COPRO가 캡처한 results_best에서 추출
            train_score = 0.0
            train_score_normalized = 0.0
            
            if hasattr(optimized_module, 'results_best'):
                # results_best: {predictor_id: {"max": [...], "average": [...], ...}}
                for pred_id, stats in optimized_module.results_best.items():
                    if stats.get('max') and len(stats['max']) > 0:
                        # 마지막 depth의 최고 점수 사용
                        raw_score = stats['max'][-1]
                        
                        # 점수 정규화: 다양한 형식 처리
                        # - 0~1: 이미 정규화됨
                        # - 1~100: 백분율 (예: 29.2)
                        # - 100+: 총점 합계 (예: 50.79)
                        if raw_score > 100:
                            # 총점인 경우: train 샘플 수로 나눔
                            normalized = raw_score / len(train_examples) if len(train_examples) > 0 else 0
                        elif raw_score > 1.0:
                            # 백분율인 경우 (예: 29.2 → 0.292)
                            normalized = raw_score / 100.0
                        else:
                            # 이미 정규화됨 (0~1)
                            normalized = raw_score
                        
                        # 범위 제한 (0~1)
                        normalized = max(0.0, min(1.0, normalized))
                        
                        if normalized > train_score_normalized:
                            train_score_normalized = normalized
                
                # 정규화 점수 → 원본 점수 (0~3.5)
                train_score = train_score_normalized * 3.5  # MAX_SCORE
                print(f"   📊 Train Best Score (from COPRO): {train_score_normalized:.3f} ({train_score_normalized*100:.1f}%) → {train_score:.3f}")
            else:
                print(f"   ⚠️ COPRO results_best를 찾을 수 없음")
            
            # 테스트 평가 (Test 데이터 사용)
            print(f"\n📊 Depth {step} 테스트 평가...")
            test_results = evaluate_with_dspy(optimized_module, test_examples, None, f"Depth {step} (Test)", num_threads=num_threads)
            
            # Train 점수 추가
            test_results['train_score'] = train_score
            test_results['train_normalized'] = train_score_normalized
            
            # Tracker에 수동 로깅 (train 점수 포함)
            if tracker:
                tracker.log_step(test_results, f"Depth {step} (Test)")
            
            score = test_results['avg_score']
            
            instruction = _get_instruction(optimized_module)
            num_demos = _get_num_demos(optimized_module)
            
            # Instruction 변경 확인 로깅
            logger.log_instruction_change(step, instruction)
            
            # 상태 업데이트
            improved = state.update(step, score, optimized_module, instruction, num_demos)
            
            logger.log_step(step, score, improved, {
                "num_demos": num_demos,
                "improvement": score - baseline_score,
                "step_time": step_time
            })
            
            # 그래프 업데이트 (tracker.log_step에서 이미 데이터 추가됨)
            if tracker:
                tracker.plot_metrics()
            
            # Early Stopping 체크
            if state.should_stop(patience):
                logger.log_early_stop(step, patience, state.best_step)
                break
            
            # 다음 단계를 위해 현재 모듈 업데이트
            current_module = optimized_module
        
        # 최종 결과
        logger.log_final(state)
        state.save()
        
        # 최종 instruction 비교 출력
        final_instruction = _get_instruction(state.best_module)
        logger.log_final_instruction_comparison(final_instruction)
        
        print("\n" + "="*60)
        print("🎯 최적화된 Instruction 확인")
        print("="*60)
        display_optimized_prompt(state.best_module)
        
        return state.best_module, state
        
    except ImportError as e:
        print(f"⚠️ COPRO를 사용할 수 없습니다: {e}")
        raise
    except Exception as e:
        print(f"⚠️ COPRO 오류: {e}")
        raise


def optimize_with_mipro_and_logging(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    test_examples: List[dspy.Example],
    metric_fn: Callable = None,
    num_candidates: int = 10,
    init_temperature: float = 1.4,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 4,
    num_threads: int = 16,
    patience: int = 3,
    eval_every: int = 2,
    tracker: 'PerformanceTracker' = None
) -> Tuple[dspy.Module, OptimizationState]:
    """
    MIPROv2 최적화 + 단계별 로깅 + Early Stopping
    
    MIPROv2는 내부적으로 베이지안 최적화를 수행하므로,
    후보들을 평가하면서 best를 추적합니다.
    
    Args:
        module: 최적화할 DSPy 모듈
        train_examples: 학습 데이터
        test_examples: 테스트 데이터
        metric_fn: 메트릭 함수
        num_candidates: 베이지안 최적화 후보 수
        init_temperature: 초기 temperature
        max_bootstrapped_demos: 부트스트랩 예시 수
        max_labeled_demos: 라벨링된 예시 수
        num_threads: 병렬 처리 스레드 수
        patience: early stopping patience
        eval_every: N 후보마다 테스트 평가
    
    Returns:
        (best_module, optimization_state)
    """
    try:
        from dspy.teleprompt import MIPROv2
        
        if metric_fn is None:
            metric_fn = normalized_metric
        
        state = OptimizationState(
            experiment_name="mipro",
            auto_save=True
        )
        logger = OptimizationLogger("mipro")
        
        print("🚀 MIPROv2 최적화 시작 (Early Stopping 활성화)")
        print("   📝 최적화 대상: Instruction + Few-shot Demos")
        print(f"   🔧 설정: candidates={num_candidates}, patience={patience}")
        print("="*60)
        
        # 원본 instruction 저장 및 출력
        original_instruction = _get_instruction(module)
        logger.set_original_instruction(original_instruction)
        
        # 초기 평가
        print("\n📊 Depth 0: 베이스라인 평가")
        baseline_results = evaluate_with_dspy(module, test_examples, tracker, "Baseline (Test)", num_threads=num_threads)
        baseline_score = baseline_results['avg_score']
        
        instruction = _get_instruction(module)
        state.update(0, baseline_score, module, instruction, 0)
        logger.log_step(0, baseline_score, True, {})
        
        if tracker:
            tracker.plot_metrics()
        
        # MIPROv2 실행
        print(f"\n{'🔄'*20}")
        print("MIPROv2 최적화 실행 중...")
        print(f"{'🔄'*20}")
        
        optimizer = MIPROv2(
            metric=metric_fn,
            auto=None,  # 수동 설정 모드 (num_candidates/num_trials 사용 시 필수)
            num_candidates=num_candidates,  # 베이지안 최적화 후보 수 추가!
            init_temperature=init_temperature,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_threads=num_threads,
            verbose=True
        )
        
        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            num_trials=num_candidates  # 시도 횟수 명시
        )
        
        # 최종 평가
        print("\n📊 Depth 1: 최종 테스트 평가...")
        final_results = evaluate_with_dspy(optimized_module, test_examples, tracker, "Final (Test)", num_threads=num_threads)
        final_score = final_results['avg_score']
        
        instruction = _get_instruction(optimized_module)
        num_demos = _get_num_demos(optimized_module)
        
        # Instruction 변경 확인 로깅
        logger.log_instruction_change(1, instruction)
        
        improved = state.update(1, final_score, optimized_module, instruction, num_demos)
        logger.log_step(1, final_score, improved, {
            "num_demos": num_demos,
            "improvement": final_score - baseline_score
        })
        
        if tracker:
            tracker.plot_metrics()
        
        # 최종 결과
        logger.log_final(state)
        state.save()
        
        # 최종 instruction 비교 출력
        final_instruction = _get_instruction(state.best_module)
        logger.log_final_instruction_comparison(final_instruction)
        
        print("\n" + "="*60)
        print("🎯 MIPROv2 최적화된 프롬프트 확인")
        print("="*60)
        display_optimized_prompt(state.best_module)
        
        return state.best_module, state
        
    except ImportError as e:
        print(f"⚠️ MIPROv2를 사용할 수 없습니다: {e}")
        raise
    except Exception as e:
        print(f"⚠️ MIPROv2 오류: {e}")
        raise


# ============================================
# Helper Functions
# ============================================

def _get_instruction(module: dspy.Module) -> str:
    """모듈에서 instruction 추출 (DSPy 3.x 호환)"""
    try:
        # 1. predictor 찾기 (cot 또는 predict)
        if hasattr(module, 'cot'):
            predictor = module.cot
        elif hasattr(module, 'predict'):
            predictor = module.predict
        else:
            predictor = module
        
        # 2. DSPy 3.x: ChainOfThought.predict.signature.instructions
        #    ChainOfThought는 내부에 predict 속성을 가지고 있음
        if hasattr(predictor, 'predict') and hasattr(predictor.predict, 'signature'):
            sig = predictor.predict.signature
            if hasattr(sig, 'instructions') and sig.instructions:
                return str(sig.instructions)
        
        # 3. 직접 signature 접근 시도
        if hasattr(predictor, 'signature'):
            sig = predictor.signature
            if hasattr(sig, 'instructions') and sig.instructions:
                return str(sig.instructions)
        
        # 4. extended_signature 시도
        if hasattr(predictor, 'extended_signature'):
            sig = predictor.extended_signature
            if hasattr(sig, 'instructions') and sig.instructions:
                return str(sig.instructions)
        
        # 5. __dict__에서 predict 찾기 (ChainOfThought의 경우)
        if hasattr(predictor, '__dict__'):
            for key, val in predictor.__dict__.items():
                if hasattr(val, 'signature'):
                    sig = val.signature
                    if hasattr(sig, 'instructions') and sig.instructions:
                        return str(sig.instructions)
        
        return "(instruction을 찾을 수 없음)"
    except Exception as e:
        return f"(instruction 추출 오류: {e})"


def _get_num_demos(module: dspy.Module) -> int:
    """모듈에서 demo 수 추출"""
    try:
        if hasattr(module, 'cot'):
            predictor = module.cot
        elif hasattr(module, 'predict'):
            predictor = module.predict
        else:
            predictor = module
        
        if hasattr(predictor, 'demos') and predictor.demos:
            return len(predictor.demos)
        return 0
    except:
        return 0


# ============================================
# Legacy Functions (기존 호환)
# ============================================

def optimize_with_copro(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    metric_fn: Callable = None,
    breadth: int = 10,
    depth: int = 1,
    init_temperature: float = 0.6,
    num_threads: int = 16
) -> dspy.Module:
    """COPRO 최적화 (레거시 - 로깅 없음)"""
    try:
        from dspy.teleprompt import COPRO
        
        if metric_fn is None:
            metric_fn = normalized_metric
        
        # breadth 최소값 검증
        if breadth < 2:
            breadth = 2
        
        print("🚀 COPRO 최적화 시작...")
        print(f"   🔧 설정: breadth={breadth}, depth={depth}")
        
        optimizer = COPRO(
            metric=metric_fn,
            breadth=breadth,
            depth=depth,
            init_temperature=init_temperature,
            verbose=False
        )
        
        optimized_module = optimizer.compile(
            module,
            trainset=train_examples,
            eval_kwargs=dict(num_threads=num_threads, display_progress=True)
        )
        
        display_optimized_prompt(optimized_module)
        print("\n✅ COPRO 최적화 완료!")
        return optimized_module
        
    except Exception as e:
        print(f"⚠️ COPRO 오류: {e}")
        raise


def optimize_with_mipro(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    metric_fn: Callable = None,
    num_candidates: int = 10,
    init_temperature: float = 1.4,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 4,
    num_threads: int = 16
) -> dspy.Module:
    """MIPROv2 최적화 (레거시 - 로깅 없음)"""
    try:
        from dspy.teleprompt import MIPROv2
        
        if metric_fn is None:
            metric_fn = normalized_metric
        
        print("🚀 MIPROv2 최적화 시작...")
        print(f"   🔧 설정: candidates={num_candidates}")
        
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
        
        display_optimized_prompt(optimized_module)
        print("\n✅ MIPROv2 최적화 완료!")
        return optimized_module
        
    except Exception as e:
        print(f"⚠️ MIPROv2 오류: {e}")
        raise


def display_optimized_prompt(module: dspy.Module):
    """최적화된 모듈의 프롬프트 출력"""
    try:
        if hasattr(module, 'cot'):
            predictor = module.cot
        elif hasattr(module, 'predict'):
            predictor = module.predict
        else:
            predictor = module
        
        if hasattr(predictor, 'extended_signature'):
            sig = predictor.extended_signature
            if hasattr(sig, 'instructions'):
                print(f"\n📋 최적화된 Instruction:")
                print("-" * 50)
                print(sig.instructions)
                print("-" * 50)
        
        if hasattr(predictor, 'signature'):
            sig = predictor.signature
            if hasattr(sig, 'instructions'):
                print(f"\n📋 Signature Instruction:")
                print("-" * 50)
                print(sig.instructions)
                print("-" * 50)
        
        if hasattr(predictor, 'demos') and predictor.demos:
            print(f"\n📚 Few-shot Demos ({len(predictor.demos)}개):")
            for i, demo in enumerate(predictor.demos[:3]):
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


# ============================================
# Main Pipeline (Updated)
# ============================================

def run_optimization_pipeline(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    test_examples: List[dspy.Example] = None,
    tracker: PerformanceTracker = None,
    optimizer_type: str = "mipro",
    config: Optional[OptimizerConfig] = None,
    save_intermediate: bool = True,
    use_early_stopping: bool = True,
    patience: int = 5
) -> Tuple[dspy.Module, dict]:
    """
    최적화 파이프라인 실행 (로깅 + Early Stopping 지원)
    
    Args:
        module: 최적화할 모듈
        train_examples: 학습 데이터 (COPRO/MIPROv2 최적화용)
        test_examples: 테스트 데이터 (단계별 평가 + 최종 평가용)
        tracker: PerformanceTracker
        optimizer_type: "copro" 또는 "mipro"
        config: OptimizerConfig (선택사항)
        save_intermediate: 중간 결과 저장 여부
        use_early_stopping: Early Stopping 사용 여부
        patience: Early Stopping patience
    
    Returns:
        (optimized_module, final_results)
    
    데이터 사용:
        - train_examples: COPRO/MIPROv2 프롬프트 최적화
        - test_examples: 단계별 평가 + 최종 성능 평가
    """
    if config is None:
        config = OptimizerConfig(optimizer_type=optimizer_type)
    
    baseline_module = module
    tracker.start()
    
    print("\n" + "🟡"*25)
    print(f"최적화 실행 ({optimizer_type.upper()})")
    if use_early_stopping:
        print(f"   Early Stopping: patience={patience} (Test 기반)")
    print("🟡"*25)
    
    optimization_start = time.time()
    
    if use_early_stopping:
        # Early Stopping 버전 사용 (test_examples로 평가)
        if optimizer_type == "copro":
            kwargs = config.to_copro_kwargs()
            optimized_module, opt_state = optimize_with_copro_and_logging(
                module=module,
                train_examples=train_examples,
                test_examples=test_examples,  # Test 사용!
                metric_fn=normalized_metric,
                patience=patience,
                tracker=tracker,  # PerformanceTracker 전달!
                **kwargs
            )
        elif optimizer_type == "mipro":
            kwargs = config.to_mipro_kwargs()
            optimized_module, opt_state = optimize_with_mipro_and_logging(
                module=module,
                train_examples=train_examples,
                test_examples=test_examples,  # Test 사용!
                metric_fn=normalized_metric,
                patience=patience,
                tracker=tracker,  # PerformanceTracker 전달!
                **kwargs
            )
        else:
            raise ValueError(f"알 수 없는 optimizer_type: {optimizer_type}")
    else:
        # 기존 버전 사용
        if optimizer_type == "copro":
            kwargs = config.to_copro_kwargs()
            optimized_module = optimize_with_copro(
                module=module,
                train_examples=train_examples,
                metric_fn=normalized_metric,
                **kwargs
            )
        elif optimizer_type == "mipro":
            kwargs = config.to_mipro_kwargs()
            optimized_module = optimize_with_mipro(
                module=module,
                train_examples=train_examples,
                metric_fn=normalized_metric,
                **kwargs
            )
        else:
            raise ValueError(f"알 수 없는 optimizer_type: {optimizer_type}")
    
    optimization_time = time.time() - optimization_start
    print(f"\n⏱️  최적화 시간: {optimization_time:.1f}초")
    
    # 최종 평가 (Test 데이터 - Unseen)
    print("\n" + "🟢"*25)
    print("최종 평가 (Test - Unseen Data)")
    print("🟢"*25)
    optimized_results = evaluate_with_dspy(
        optimized_module, test_examples, tracker, "Final (Test)"
    )
    
    # 프롬프트 비교
    print("\n" + "🔮"*25)
    print("프롬프트 변화 확인")
    print("🔮"*25)
    compare_prompts(baseline_module, optimized_module)
    
    # 최종 그래프 저장
    tracker.plot_metrics()
    tracker.save_history()
    tracker.summary()
    
    return optimized_module, optimized_results


# 레거시 호환 함수
def run_optimization_with_tracking(
    module: dspy.Module,
    train_examples: List[dspy.Example],
    test_examples: List[dspy.Example],
    tracker: PerformanceTracker,
    optimizer_type: str = "mipro",
    save_intermediate: bool = True,
    breadth: int = 10,
    depth: int = 3,
    init_temperature: float = 0.6,
    num_threads: int = 10,
) -> Tuple[dspy.Module, dict]:
    """레거시 호환 함수"""
    config = OptimizerConfig(
        optimizer_type=optimizer_type,
        breadth=breadth,
        depth=depth,
        init_temperature=init_temperature,
        num_threads=num_threads
    )
    
    return run_optimization_pipeline(
        module=module,
        train_examples=train_examples,
        test_examples=test_examples,
        tracker=tracker,
        optimizer_type=optimizer_type,
        config=config,
        save_intermediate=save_intermediate,
        use_early_stopping=True,
        patience=2
    )


print("✅ 최적화 함수 로드 완료")
print("=" * 50)
print("🔧 사용 가능한 최적화 방법:")
print("   1️⃣ optimize_with_copro_and_logging()  - COPRO + 로깅 + Early Stopping ⭐")
print("   2️⃣ optimize_with_mipro_and_logging()  - MIPROv2 + 로깅 + Early Stopping ⭐")
print("   3️⃣ optimize_with_copro()  - COPRO (레거시)")
print("   4️⃣ optimize_with_mipro()  - MIPROv2 (레거시)")
print("=" * 50)

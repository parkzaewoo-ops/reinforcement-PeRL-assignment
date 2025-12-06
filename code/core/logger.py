from dataclasses import dataclass
from typing import List, Dict
import time
import datetime
import matplotlib.pyplot as plt
import json
import dspy
from datetime import datetime
import os
import pandas as pd

RESULTS_DIR = "/data/workspace/sogang/sqlagent/results"


@dataclass
class StepMetrics:
    """각 스텝의 메트릭을 저장하는 클래스"""
    step: int
    timestamp: str
    avg_score: float
    normalized_score: float
    perfect_match: int
    result_match: int
    partial_match: int
    wrong_result: int
    error_count: int
    total_samples: int
    elapsed_time: float  # 초 단위
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Train 점수 (COPRO 내부 평가)
    train_score: float = 0.0
    train_normalized: float = 0.0


class PerformanceTracker:
    """성능, 시간, 토큰 사용량을 트래킹하는 클래스"""
    
    def __init__(self, experiment_name: str = "text2sql"):
        self.experiment_name = experiment_name
        self.metrics_history: List[StepMetrics] = []
        self.start_time = None
        self.current_step = 0
        # 세션 ID: 실험 시작 시 생성되어 해당 세션 내에서 동일하게 유지
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def start(self):
        """트래킹 시작"""
        self.start_time = time.time()
        self.current_step = 0
        # 새 세션 시작 시 session_id 갱신
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"🚀 트래킹 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📁 세션 ID: {self.session_id}")
        
    def get_token_usage(self) -> Dict[str, int]:
        """DSPy LM에서 토큰 사용량 추출"""
        try:
            lm = dspy.settings.lm
            if hasattr(lm, 'history') and lm.history:
                total_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0
                
                for entry in lm.history:
                    if isinstance(entry, dict):
                        usage = entry.get('usage', {})
                        if isinstance(usage, dict):
                            total_tokens += usage.get('total_tokens', 0)
                            prompt_tokens += usage.get('prompt_tokens', 0)
                            completion_tokens += usage.get('completion_tokens', 0)
                
                return {
                    'total_tokens': total_tokens,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens
                }
        except:
            pass
        return {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
    
    def log_step(self, results: dict, step_name: str = ""):
        """각 스텝의 결과를 로깅 + CSV 저장"""
        # Step 0부터 시작 (먼저 현재 step 사용, 나중에 증가)
        current = self.current_step
        elapsed = time.time() - self.start_time if self.start_time else 0
        token_usage = self.get_token_usage()
        
        metrics = StepMetrics(
            step=current,  # 0부터 시작
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            avg_score=results.get('avg_score', 0),
            normalized_score=results.get('avg_normalized', 0),
            perfect_match=results.get('perfect_match', 0),
            result_match=results.get('result_match', 0),
            partial_match=results.get('partial_match', 0),
            wrong_result=results.get('wrong_result', 0),
            error_count=results.get('error', 0),
            total_samples=results.get('total', 0),
            elapsed_time=elapsed,
            total_tokens=token_usage['total_tokens'],
            prompt_tokens=token_usage['prompt_tokens'],
            completion_tokens=token_usage['completion_tokens'],
            train_score=results.get('train_score', 0),
            train_normalized=results.get('train_normalized', 0)
        )
        
        self.metrics_history.append(metrics)
        self.current_step += 1  # 다음 step을 위해 증가
        
        # 각 epoch마다 CSV 저장
        self._save_step_to_csv(metrics, step_name)
        
        print(f"\n📊 Depth {current} {step_name}")
        print(f"   ⏱️  경과 시간: {elapsed:.1f}초")
        print(f"   🎯 평균 점수: {metrics.avg_score:.3f}")
        print(f"   🪙 총 토큰: {metrics.total_tokens:,}")
        print(f"   📈 정규화 점수: {metrics.normalized_score:.3f}")
        
        return metrics
    
    def _save_step_to_csv(self, metrics: StepMetrics, step_name: str = ""):
        """각 step을 CSV 파일에 추가 (세션별)"""
        csv_path = f"{RESULTS_DIR}/{self.experiment_name}_{self.session_id}_epochs.csv"
        
        # 데이터 행 준비
        row_data = {
            'epoch': metrics.step,
            'step_name': step_name,
            'timestamp': metrics.timestamp,
            'avg_score': round(metrics.avg_score, 4),
            'normalized_score': round(metrics.normalized_score, 4),
            'train_score': round(metrics.train_score, 4),
            'train_normalized': round(metrics.train_normalized, 4),
            'perfect_match': metrics.perfect_match,
            'result_match': metrics.result_match,
            'partial_match': metrics.partial_match,
            'wrong_result': metrics.wrong_result,
            'error_count': metrics.error_count,
            'total_samples': metrics.total_samples,
            'success_rate': round((metrics.perfect_match + metrics.result_match + metrics.partial_match) / max(metrics.total_samples, 1) * 100, 2),
            'error_rate': round(metrics.error_count / max(metrics.total_samples, 1) * 100, 2),
            'elapsed_time': round(metrics.elapsed_time, 2),
            'total_tokens': metrics.total_tokens,
            'prompt_tokens': metrics.prompt_tokens,
            'completion_tokens': metrics.completion_tokens
        }
        
        # 파일 존재 여부 확인
        file_exists = os.path.exists(csv_path)
        
        # CSV에 추가
        df = pd.DataFrame([row_data])
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
        
        print(f"   📄 CSV 저장: {os.path.basename(csv_path)}")
    
    def save_history_csv(self, save_path: str = None) -> str:
        """전체 히스토리를 CSV로 저장 (세션별)"""
        if not self.metrics_history:
            print("⚠️ 저장할 메트릭이 없습니다.")
            return None
        
        if save_path is None:
            save_path = f"{RESULTS_DIR}/{self.experiment_name}_{self.session_id}_full_history.csv"
        
        rows = []
        for m in self.metrics_history:
            rows.append({
                'epoch': m.step,
                'timestamp': m.timestamp,
                'avg_score': round(m.avg_score, 4),
                'normalized_score': round(m.normalized_score, 4),
                'perfect_match': m.perfect_match,
                'result_match': m.result_match,
                'partial_match': m.partial_match,
                'wrong_result': m.wrong_result,
                'error_count': m.error_count,
                'total_samples': m.total_samples,
                'success_rate': round((m.perfect_match + m.result_match + m.partial_match) / max(m.total_samples, 1) * 100, 2),
                'error_rate': round(m.error_count / max(m.total_samples, 1) * 100, 2),
                'elapsed_time': round(m.elapsed_time, 2),
                'total_tokens': m.total_tokens
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"📊 전체 히스토리 CSV 저장: {save_path}")
        return save_path
    
    def plot_metrics(self, save_path: str = None) -> str:
        """성능 메트릭을 시각화하여 PNG로 저장 (라인플롯)"""
        if not self.metrics_history:
            print("⚠️ 저장된 메트릭이 없습니다.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Optimization Tracking: {self.experiment_name}', fontsize=16, fontweight='bold')
        
        steps = [m.step for m in self.metrics_history]
        
        # 1. 평균 점수 변화 (라인플롯) - Val vs Train
        ax1 = axes[0, 0]
        avg_scores = [m.avg_score for m in self.metrics_history]
        train_scores = [m.train_score for m in self.metrics_history]
        
        # Val 점수 (기본)
        ax1.plot(steps, avg_scores, '-o', linewidth=2.5, markersize=10, 
                 label='Val Score (0~3.5)', color='#3498db')
        ax1.fill_between(steps, avg_scores, alpha=0.2, color='#3498db')
        
        # Train 점수 (COPRO 내부 평가) - 0이 아닌 경우만 표시
        if any(s > 0 for s in train_scores):
            ax1.plot(steps, train_scores, '--s', linewidth=2, markersize=8, 
                     label='Train Score (0~3.5)', color='#9b59b6')
            ax1.fill_between(steps, train_scores, alpha=0.1, color='#9b59b6')
        
        # Best 지점 표시
        if avg_scores:
            best_idx = avg_scores.index(max(avg_scores))
            ax1.scatter([steps[best_idx]], [avg_scores[best_idx]], 
                       s=200, color='gold', marker='*', zorder=5, label='Best (Val)')
            ax1.annotate(f'Best: {avg_scores[best_idx]:.3f}', 
                        xy=(steps[best_idx], avg_scores[best_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='#e67e22')
        
        ax1.set_xlabel('Depth', fontsize=12)
        ax1.set_ylabel('Average Score', fontsize=12)
        ax1.set_title('Val vs Train Score per Depth', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(steps)
        
        # 2. 상태별 카운트 변화 (라인플롯) - Error vs Partial vs Perfect
        ax2 = axes[0, 1]
        
        error_counts = [m.error_count for m in self.metrics_history]
        partial_counts = [m.partial_match for m in self.metrics_history]
        result_match_counts = [m.result_match for m in self.metrics_history]
        perfect_counts = [m.perfect_match for m in self.metrics_history]
        
        ax2.plot(steps, error_counts, 'r-s', linewidth=2, markersize=8, 
                 label='Error', color='#e74c3c')
        ax2.plot(steps, partial_counts, 'y-^', linewidth=2, markersize=8, 
                 label='Partial Match', color='#f39c12')
        ax2.plot(steps, result_match_counts, 'g-d', linewidth=2, markersize=8, 
                 label='Result Match', color='#27ae60')
        ax2.plot(steps, perfect_counts, 'b-*', linewidth=2, markersize=10, 
                 label='Perfect Match', color='#2ecc71')
        
        ax2.set_xlabel('Depth', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('📊 Status Count per Depth', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xticks(steps)
        
        # 3. 성공률 추이 (Error 감소, Success 증가)
        ax3 = axes[1, 0]
        
        total_samples = [m.total_samples for m in self.metrics_history]
        success_rates = []
        error_rates = []
        
        for i, m in enumerate(self.metrics_history):
            total = m.total_samples if m.total_samples > 0 else 1
            success = m.perfect_match + m.result_match + m.partial_match
            success_rates.append(success / total * 100)
            error_rates.append(m.error_count / total * 100)
        
        ax3.plot(steps, success_rates, 'g-o', linewidth=2.5, markersize=8, 
                 label='Success Rate (%)', color='#27ae60')
        ax3.plot(steps, error_rates, 'r-s', linewidth=2.5, markersize=8, 
                 label='Error Rate (%)', color='#e74c3c')
        
        ax3.fill_between(steps, success_rates, alpha=0.2, color='#27ae60')
        ax3.fill_between(steps, error_rates, alpha=0.2, color='#e74c3c')
        
        ax3.set_xlabel('Depth', fontsize=12)
        ax3.set_ylabel('Rate (%)', fontsize=12)
        ax3.set_title('📉 Success vs Error Rate per Depth', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xticks(steps)
        ax3.set_ylim(0, 100)
        
        # 4. 정규화 점수 및 실행 시간
        ax4 = axes[1, 1]
        
        normalized_scores = [m.normalized_score for m in self.metrics_history]
        times = [m.elapsed_time for m in self.metrics_history]
        
        # 듀얼 축: 정규화 점수 + 시간
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(steps, normalized_scores, 'b-o', linewidth=2.5, markersize=8, 
                         label='Normalized Score (0~1)', color='#3498db')
        line2 = ax4_twin.plot(steps, times, 'g--s', linewidth=2, markersize=6, 
                              label='Elapsed Time (s)', color='#1abc9c', alpha=0.7)
        
        ax4.set_xlabel('Depth', fontsize=12)
        ax4.set_ylabel('Normalized Score', fontsize=12, color='#3498db')
        ax4_twin.set_ylabel('Time (seconds)', fontsize=12, color='#1abc9c')
        ax4.set_title('⏱️ Score & Execution Time per Depth', fontsize=14, fontweight='bold')
        
        # 범례 합치기
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='best', fontsize=10)
        
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xticks(steps)
        
        plt.tight_layout()
        
        # PNG 저장 (세션 내에서 덮어씌우기)
        if save_path is None:
            save_path = f"{RESULTS_DIR}/{self.experiment_name}_{self.session_id}_metrics.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📈 그래프 저장 완료: {save_path}")
        return save_path
    
    def save_history(self, save_path: str = None) -> str:
        """메트릭 히스토리를 JSON으로 저장 (세션별)"""
        if save_path is None:
            save_path = f"{RESULTS_DIR}/{self.experiment_name}_{self.session_id}_history.json"
        
        history_data = []
        for m in self.metrics_history:
            history_data.append({
                'step': m.step,
                'timestamp': m.timestamp,
                'avg_score': m.avg_score,
                'normalized_score': m.normalized_score,
                'perfect_match': m.perfect_match,
                'result_match': m.result_match,
                'partial_match': m.partial_match,
                'wrong_result': m.wrong_result,
                'error_count': m.error_count,
                'total_samples': m.total_samples,
                'elapsed_time': m.elapsed_time,
                'total_tokens': m.total_tokens,
                'prompt_tokens': m.prompt_tokens,
                'completion_tokens': m.completion_tokens
            })
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 히스토리 저장 완료: {save_path}")
        return save_path
    
    def summary(self):
        """최종 요약 출력 + CSV 저장"""
        if not self.metrics_history:
            print("⚠️ 저장된 메트릭이 없습니다.")
            return
        
        first = self.metrics_history[0]
        last = self.metrics_history[-1]
        
        # Best epoch 찾기
        best_idx = 0
        best_score = 0
        for i, m in enumerate(self.metrics_history):
            if m.avg_score > best_score:
                best_score = m.avg_score
                best_idx = i
        
        print("\n" + "="*60)
        print("📊 최종 성능 요약")
        print("="*60)
        print(f"총 Depth: {len(self.metrics_history)}")
        print(f"총 실행 시간: {last.elapsed_time:.1f}초 ({last.elapsed_time/60:.1f}분)")
        print(f"총 토큰 사용: {last.total_tokens:,}")
        print(f"\n점수 변화:")
        print(f"  시작 (Depth 0): {first.avg_score:.3f}")
        print(f"  최종 (Depth {len(self.metrics_history) - 1}): {last.avg_score:.3f}")
        print(f"  Best (Depth {best_idx}): {best_score:.3f} ⭐")
        print(f"  개선율: {best_score - first.avg_score:+.3f}")
        print("="*60)
        
        # 전체 히스토리 CSV 저장
        self.save_history_csv()


# 글로벌 트래커 인스턴스
tracker = PerformanceTracker("bird_text2sql")
print("✅ PerformanceTracker 클래스 로드 완료")


# ============================================
# 10-3. 저장된 결과 확인 및 시각화
# ============================================

import glob

def list_saved_results():
    """저장된 결과 파일 목록 출력"""
    print(f"\n📁 저장된 결과 ({RESULTS_DIR}):")
    print("="*50)
    
    # PNG 파일
    png_files = glob.glob(f"{RESULTS_DIR}/*.png")
    if png_files:
        print("\n📊 그래프 파일 (.png):")
        for f in sorted(png_files):
            print(f"   - {os.path.basename(f)}")
    
    # JSON 파일
    json_files = glob.glob(f"{RESULTS_DIR}/*.json")
    if json_files:
        print("\n📋 히스토리 파일 (.json):")
        for f in sorted(json_files):
            print(f"   - {os.path.basename(f)}")
    
    if not png_files and not json_files:
        print("   (저장된 파일 없음)")


def load_and_display_history(json_path: str):
    """저장된 히스토리 파일 로드 및 표시"""
    with open(json_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    df = pd.DataFrame(history)
    print(f"\n📊 히스토리 로드: {os.path.basename(json_path)}")
    print(df.to_string(index=False))
    return df


def display_latest_graph():
    """최신 그래프를 IPython에서 표시"""
    try:
        from IPython.display import Image, display
        
        png_files = glob.glob(f"{RESULTS_DIR}/*.png")
        if png_files:
            latest = max(png_files, key=os.path.getmtime)
            print(f"\n🖼️ 최신 그래프: {os.path.basename(latest)}")
            display(Image(filename=latest))
        else:
            print("저장된 그래프가 없습니다.")
    except ImportError:
        print("IPython 환경이 아닙니다. 그래프 파일을 직접 열어주세요.")


# 저장된 결과 확인
list_saved_results()

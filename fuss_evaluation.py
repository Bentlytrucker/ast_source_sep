#!/usr/bin/env python3
"""
개선된 FUSS 데이터셋 음원 분리 성능 측정 스크립트

이 스크립트는:
1. FUSS 데이터셋의 실제 구조를 파악하여 참조 소스와 분리된 소스를 올바르게 매칭
2. mir_eval을 사용한 정확한 SIR, SDR, SAR 지표 계산
3. 다양한 클래스 조합에 대한 성능 평가
4. 상세한 결과 분석 및 시각화

사용법:
python fuss_evaluation_improved.py --target_classes "speech,music" --num_samples 5
"""

import os
import sys
import argparse
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
import tarfile
from urllib.parse import urlparse
import glob
import re

# mir_eval 라이브러리 import
try:
    from mir_eval.separation import bss_eval_sources
except ImportError:
    print("mir_eval 라이브러리가 설치되지 않았습니다. 설치 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mir_eval"])
    from mir_eval.separation import bss_eval_sources

class FUSSDatasetAnalyzer:
    """FUSS 데이터셋 구조 분석 클래스"""
    
    def __init__(self, root_dir: str = "./fuss_data"):
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "fuss_dev"
        self.ssdata_dir = self.data_dir / "ssdata"
        self.ssdata_reverb_dir = self.data_dir / "ssdata_reverb"
    
    def analyze_dataset_structure(self):
        """데이터셋 구조 분석"""
        print("=== FUSS 데이터셋 구조 분석 ===")
        
        for split in ['train', 'validation', 'eval']:
            split_dir = self.ssdata_dir / split
            if split_dir.exists():
                print(f"\n{split} 스플릿:")
                print(f"  디렉토리: {split_dir}")
                
                # 파일 개수 확인
                wav_files = list(split_dir.glob("*.wav"))
                print(f"  WAV 파일 개수: {len(wav_files)}")
                
                # 파일명 패턴 분석
                if wav_files:
                    sample_files = wav_files[:5]
                    print(f"  샘플 파일명:")
                    for f in sample_files:
                        print(f"    {f.name}")
                
                # sources 디렉토리 확인
                sources_dir = split_dir / "sources"
                if sources_dir.exists():
                    source_files = list(sources_dir.glob("*.wav"))
                    print(f"  소스 파일 개수: {len(source_files)}")
                    
                    if source_files:
                        sample_sources = source_files[:5]
                        print(f"  샘플 소스 파일명:")
                        for f in sample_sources:
                            print(f"    {f.name}")
    
    def get_mixture_source_pairs(self, split: str = 'eval', num_samples: int = 10) -> List[Dict]:
        """혼합 오디오와 참조 소스 쌍 반환"""
        pairs = []
        split_dir = self.ssdata_dir / split
        sources_dir = split_dir / "sources"
        
        if not split_dir.exists() or not sources_dir.exists():
            print(f"스플릿 '{split}' 또는 sources 디렉토리가 존재하지 않습니다.")
            return pairs
        
        # 혼합 오디오 파일들
        mixture_files = list(split_dir.glob("*.wav"))
        source_files = list(sources_dir.glob("*.wav"))
        
        print(f"혼합 오디오: {len(mixture_files)}개")
        print(f"참조 소스: {len(source_files)}개")
        
        # 파일명 기반으로 매칭
        for mixture_file in mixture_files[:num_samples]:
            mixture_name = mixture_file.stem
            
            # 해당 혼합 오디오에 대응하는 소스 파일들 찾기
            corresponding_sources = []
            for source_file in source_files:
                source_name = source_file.stem
                # 파일명 패턴 매칭 (예: mixture_001.wav -> source_001_*.wav)
                if mixture_name in source_name or source_name.startswith(mixture_name):
                    corresponding_sources.append(source_file)
            
            if corresponding_sources:
                pair = {
                    'mixture': mixture_file,
                    'sources': corresponding_sources,
                    'mixture_name': mixture_name
                }
                pairs.append(pair)
        
        print(f"매칭된 혼합-소스 쌍: {len(pairs)}개")
        return pairs

class SourceSeparationEvaluator:
    """음원 분리 성능 평가 클래스"""
    
    def __init__(self, separator_script: str = "separator.py"):
        self.separator_script = separator_script
    
    def separate_audio(self, input_audio: Path, output_dir: Path) -> bool:
        """오디오 분리 수행"""
        try:
            # separator.py 스크립트 실행
            cmd = [
                sys.executable, self.separator_script,
                "--input", str(input_audio),
                "--output", str(output_dir),
                "--passes", "3",
                "--no_debug"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✓ 분리 완료: {input_audio.name}")
                return True
            else:
                print(f"  ✗ 분리 실패: {input_audio.name}")
                print(f"    에러: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ✗ 분리 중 오류 발생: {e}")
            return False
    
    def load_audio(self, file_path: Path) -> np.ndarray:
        """오디오 파일 로드"""
        try:
            audio, sr = sf.read(file_path)
            return audio
        except Exception as e:
            print(f"오디오 로드 실패 {file_path}: {e}")
            return np.array([])
    
    def align_audio_lengths(self, audios: List[np.ndarray]) -> List[np.ndarray]:
        """오디오 길이 맞추기 (가장 짧은 길이로 자르기)"""
        if not audios:
            return audios
        
        min_length = min(len(audio) for audio in audios)
        return [audio[:min_length] for audio in audios]
    
    def calculate_metrics(self, reference_sources: List[np.ndarray], 
                         estimated_sources: List[np.ndarray]) -> Dict[str, float]:
        """SIR, SDR, SAR 지표 계산"""
        try:
            # 오디오 길이 맞추기
            reference_sources = self.align_audio_lengths(reference_sources)
            estimated_sources = self.align_audio_lengths(estimated_sources)
            
            if not reference_sources or not estimated_sources:
                return {'SDR': 0.0, 'SIR': 0.0, 'SAR': 0.0}
            
            # mir_eval을 사용한 성능 지표 계산
            sdr, sir, sar, _ = bss_eval_sources(
                np.array(reference_sources), 
                np.array(estimated_sources)
            )
            
            return {
                'SDR': float(np.mean(sdr)),
                'SIR': float(np.mean(sir)),
                'SAR': float(np.mean(sar))
            }
        except Exception as e:
            print(f"지표 계산 중 오류: {e}")
            return {'SDR': 0.0, 'SIR': 0.0, 'SAR': 0.0}
    
    def match_sources(self, reference_sources: List[Path], 
                     estimated_sources: List[Path]) -> List[Tuple[Path, Path]]:
        """참조 소스와 분리된 소스 매칭"""
        matches = []
        
        # 간단한 매칭 전략: 파일 개수가 같으면 순서대로 매칭
        if len(reference_sources) == len(estimated_sources):
            matches = list(zip(reference_sources, estimated_sources))
        else:
            # 파일 개수가 다르면 파일명 기반으로 매칭 시도
            for ref_source in reference_sources:
                ref_name = ref_source.stem.lower()
                best_match = None
                best_score = 0
                
                for est_source in estimated_sources:
                    est_name = est_source.stem.lower()
                    # 간단한 문자열 유사도 계산
                    common_chars = sum(1 for c in ref_name if c in est_name)
                    score = common_chars / max(len(ref_name), len(est_name))
                    
                    if score > best_score:
                        best_score = score
                        best_match = est_source
                
                if best_match and best_score > 0.3:  # 30% 이상 유사도
                    matches.append((ref_source, best_match))
        
        return matches

class FUSSEvaluationPipeline:
    """FUSS 평가 파이프라인"""
    
    def __init__(self, root_dir: str = "./fuss_data"):
        self.analyzer = FUSSDatasetAnalyzer(root_dir)
        self.evaluator = SourceSeparationEvaluator()
        self.results = []
    
    def run_evaluation(self, num_samples: int = 5, split: str = 'eval'):
        """평가 실행"""
        print("=== FUSS 데이터셋 음원 분리 성능 평가 시작 ===")
        
        # 1. 데이터셋 구조 분석
        self.analyzer.analyze_dataset_structure()
        
        # 2. 혼합-소스 쌍 가져오기
        pairs = self.analyzer.get_mixture_source_pairs(split, num_samples)
        
        if not pairs:
            print("평가할 데이터가 없습니다.")
            return
        
        # 3. 각 쌍에 대해 평가 수행
        for i, pair in enumerate(pairs):
            print(f"\n--- 샘플 {i+1}/{len(pairs)} 평가 중 ---")
            print(f"혼합 오디오: {pair['mixture_name']}")
            print(f"참조 소스 개수: {len(pair['sources'])}")
            
            # 분리 수행
            output_dir = Path(f"separation_output_{i}")
            output_dir.mkdir(exist_ok=True)
            
            success = self.evaluator.separate_audio(pair['mixture'], output_dir)
            
            if success:
                # 분리된 파일들 가져오기
                separated_files = list(output_dir.glob("*.wav"))
                print(f"  분리된 파일 개수: {len(separated_files)}")
                
                if separated_files:
                    # 소스 매칭
                    matches = self.evaluator.match_sources(pair['sources'], separated_files)
                    print(f"  매칭된 소스 쌍: {len(matches)}개")
                    
                    if matches:
                        # 성능 지표 계산
                        reference_audios = []
                        estimated_audios = []
                        
                        for ref_path, est_path in matches:
                            ref_audio = self.evaluator.load_audio(ref_path)
                            est_audio = self.evaluator.load_audio(est_path)
                            
                            if len(ref_audio) > 0 and len(est_audio) > 0:
                                reference_audios.append(ref_audio)
                                estimated_audios.append(est_audio)
                        
                        if reference_audios and estimated_audios:
                            metrics = self.evaluator.calculate_metrics(reference_audios, estimated_audios)
                            
                            result = {
                                'sample_id': i,
                                'mixture_name': pair['mixture_name'],
                                'num_sources': len(pair['sources']),
                                'num_separated': len(separated_files),
                                'num_matched': len(matches),
                                'SDR': metrics['SDR'],
                                'SIR': metrics['SIR'],
                                'SAR': metrics['SAR']
                            }
                            self.results.append(result)
                            
                            print(f"  성능 지표 - SDR: {metrics['SDR']:.2f}, SIR: {metrics['SIR']:.2f}, SAR: {metrics['SAR']:.2f}")
                        else:
                            print("  유효한 오디오 데이터가 없어 평가 건너뜀")
                    else:
                        print("  소스 매칭 실패로 평가 건너뜀")
                else:
                    print("  분리된 파일이 없어 평가 건너뜀")
            else:
                print("  분리 실패로 평가 건너뜀")
        
        # 4. 결과 정리 및 표 출력
        self.print_results_table()
        self.plot_results()
    
    def print_results_table(self):
        """결과를 표로 출력"""
        if not self.results:
            print("평가 결과가 없습니다.")
            return
        
        print("\n=== 평가 결과 요약 ===")
        
        # DataFrame 생성
        df = pd.DataFrame(self.results)
        
        # 개별 결과 표
        print("\n개별 샘플 결과:")
        print(df.to_string(index=False, float_format='%.2f'))
        
        # 평균 결과
        print("\n평균 성능 지표:")
        avg_results = df[['SDR', 'SIR', 'SAR']].mean()
        print(f"SDR: {avg_results['SDR']:.2f} dB")
        print(f"SIR: {avg_results['SIR']:.2f} dB")
        print(f"SAR: {avg_results['SAR']:.2f} dB")
        
        # 표준편차
        std_results = df[['SDR', 'SIR', 'SAR']].std()
        print(f"\n표준편차:")
        print(f"SDR: {std_results['SDR']:.2f} dB")
        print(f"SIR: {std_results['SIR']:.2f} dB")
        print(f"SAR: {std_results['SAR']:.2f} dB")
        
        # 결과를 CSV 파일로 저장
        output_file = "fuss_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n결과가 {output_file}에 저장되었습니다.")
    
    def plot_results(self):
        """결과 시각화"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # 성능 지표별 박스플롯
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['SDR', 'SIR', 'SAR']
        for i, metric in enumerate(metrics):
            axes[i].boxplot(df[metric])
            axes[i].set_title(f'{metric} Distribution')
            axes[i].set_ylabel('dB')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fuss_evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 샘플별 성능 지표 비교
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['SDR'], width, label='SDR', alpha=0.8)
        ax.bar(x, df['SIR'], width, label='SIR', alpha=0.8)
        ax.bar(x + width, df['SAR'], width, label='SAR', alpha=0.8)
        
        ax.set_xlabel('Sample ID')
        ax.set_ylabel('dB')
        ax.set_title('Performance Metrics by Sample')
        ax.set_xticks(x)
        ax.set_xticklabels(df['sample_id'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fuss_evaluation_by_sample.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="FUSS 데이터셋 음원 분리 성능 평가")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="평가할 샘플 수")
    parser.add_argument("--split", default="eval",
                       choices=['train', 'validation', 'eval'],
                       help="사용할 데이터 스플릿")
    parser.add_argument("--root_dir", default="./fuss_data",
                       help="FUSS 데이터셋 루트 디렉토리")
    parser.add_argument("--separator_script", default="separator.py",
                       help="사용할 분리 스크립트")
    
    args = parser.parse_args()
    
    # 평가 파이프라인 실행
    pipeline = FUSSEvaluationPipeline(args.root_dir)
    pipeline.evaluator.separator_script = args.separator_script
    pipeline.run_evaluation(args.num_samples, args.split)

if __name__ == "__main__":
    main()

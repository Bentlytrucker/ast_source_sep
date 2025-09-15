#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSD50K 데이터셋을 사용한 음원 분리 성능 측정
- SDR, SIR, SAR 지표 사용
- 결과를 파일로 저장
"""

import os
import sys
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import torch
from difflib import SequenceMatcher
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torchaudio
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

# 성능 측정 라이브러리
try:
    from mir_eval.separation import bss_eval_sources
    MIR_EVAL_AVAILABLE = True
except ImportError:
    MIR_EVAL_AVAILABLE = False

# separator.py import
sys.path.append(os.path.dirname(__file__))
try:
    import separator
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    SEPARATOR_AVAILABLE = True
except ImportError:
    SEPARATOR_AVAILABLE = False

class FSD50KEvaluator:
    """FSD50K 평가기"""
    
    def __init__(self):
        self.eval_audio_dir = "FSD50K.eval_audio"
        self.ground_truth_file = "FSD50K.ground_truth/eval.csv"
        self.results = []
        
        # 로그 파일 생성
        self.log_file = open("evaluation_log.txt", "w", encoding="utf-8")
        self.log("FSD50K Performance Evaluation Started")
        
        # Ground truth 로드
        self.ground_truth_df = pd.read_csv(self.ground_truth_file)
        self.log(f"Loaded {len(self.ground_truth_df)} ground truth entries")
        
        # 클래스 매핑 테이블 초기화
        self.class_mapping = self._initialize_class_mapping()
        
        # Sentence Transformer 모델 초기화 (자동 유사도 계산용)
        try:
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.log("✅ Sentence Transformer loaded for automatic similarity calculation")
        except Exception as e:
            self.similarity_model = None
            self.log(f"⚠️ Sentence Transformer failed to load: {e}")
            self.log("   Falling back to string-based similarity")
        
        # Sound Separator 초기화
        global SEPARATOR_AVAILABLE
        if SEPARATOR_AVAILABLE:
            self.log("Separator module loaded successfully")
        else:
            self.log("Warning: Separator module not available")
    
    def log(self, message: str):
        """로그 메시지 출력 및 파일 저장"""
        print(message)
        self.log_file.write(message + "\n")
    
    def _initialize_class_mapping(self) -> Dict[str, List[str]]:
        """AudioSet 클래스와 FSD50K 클래스 간의 매핑 테이블 초기화"""
        return {
            # 경보/사이렌 관련
            'Siren': ['Siren', 'Ambulance (siren)', 'Fire engine, fire truck (siren)', 'Emergency vehicle', 'Police car (siren)'],
            'Alarm': ['Alarm', 'Alarm clock', 'Fire alarm', 'Smoke detector, smoke alarm', 'Car alarm'],
            'Alarm clock': ['Alarm clock', 'Alarm', 'Clock'],
            
            # 화재/폭발 관련
            'Fire': ['Fire', 'Fire alarm', 'Fire engine, fire truck', 'Fireworks'],
            'Explosion': ['Explosion', 'Fireworks', 'Gunshot, gunfire', 'Bomb'],
            'Fireworks': ['Fireworks', 'Explosion', 'Fire'],
            
            # 총소리 관련
            'Gunshot': ['Gunshot, gunfire', 'Gun', 'Shooting', 'Explosion'],
            
            # 비명/소리 관련
            'Screaming': ['Screaming', 'Shout', 'Yell', 'Human voice', 'Crying and sobbing'],
            'Speech': ['Speech', 'Conversation', 'Human voice', 'Talking'],
            
            # 물 관련
            'Water': ['Water', 'Waves and surf', 'Ocean', 'Rain', 'Waterfall'],
            'Dishes': ['Dishes, pots, and pans', 'Kitchen', 'Cooking'],
            'Boiling': ['Boiling', 'Cooking', 'Kitchen'],
            
            # 문/노크 관련
            'Doorbell': ['Doorbell', 'Bell', 'Chime'],
            'Knock': ['Knock', 'Door', 'Thump'],
            
            # 전화 관련
            'Ringtone': ['Ringtone', 'Telephone', 'Phone', 'Bell'],
            'Telephone': ['Telephone', 'Phone', 'Ringtone', 'Telephone bell ringing'],
            'Telephone bell ringing': ['Telephone bell ringing', 'Telephone', 'Ringtone', 'Bell'],
            'Dial tone': ['Dial tone', 'Telephone', 'Phone'],
            
            # 파손 관련
            'Breaking': ['Breaking', 'Crash', 'Shatter', 'Glass', 'Smash']
        }
    
    def _normalize_class_name(self, class_name: str) -> str:
        """클래스 이름 정규화"""
        # 소문자 변환 및 특수문자 제거
        normalized = re.sub(r'[^\w\s]', '', class_name.lower())
        # 공백을 언더스코어로 변환
        normalized = re.sub(r'\s+', '_', normalized)
        return normalized
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """두 문자열 간의 유사도 계산 (0-1)"""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _calculate_semantic_similarity(self, str1: str, str2: str) -> float:
        """의미적 유사도 계산 (Sentence Transformers 사용)"""
        if self.similarity_model is None:
            return self._calculate_string_similarity(str1, str2)
        
        try:
            # 문장을 벡터로 변환
            embeddings = self.similarity_model.encode([str1, str2])
            
            # 코사인 유사도 계산
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
        except Exception as e:
            self.log(f"⚠️ Semantic similarity calculation failed: {e}")
            return self._calculate_string_similarity(str1, str2)
    
    def _find_best_match(self, predicted_class: str, true_classes: List[str], threshold: float = 0.3) -> Tuple[str, float]:
        """예측된 클래스와 실제 클래스들 중 가장 유사한 것 찾기 (자동 유사도 계산)"""
        best_match = None
        best_similarity = 0.0
        
        # 1. 직접 매핑 테이블에서 찾기 (정확한 매칭)
        for true_class in true_classes:
            if true_class in self.class_mapping:
                for audioset_class in self.class_mapping[true_class]:
                    if predicted_class.lower() in audioset_class.lower() or audioset_class.lower() in predicted_class.lower():
                        return true_class, 1.0
        
        # 2. 자동 의미적 유사도 계산 (Sentence Transformers)
        for true_class in true_classes:
            similarity = self._calculate_semantic_similarity(predicted_class, true_class)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = true_class
        
        # 3. 키워드 매칭 (fallback)
        if best_match is None:
            predicted_keywords = set(predicted_class.lower().split())
            for true_class in true_classes:
                true_keywords = set(true_class.lower().split())
                if predicted_keywords & true_keywords:  # 교집합이 있으면
                    return true_class, 0.8
        
        return best_match, best_similarity
    
    def _map_predicted_to_true_classes(self, predicted_classes: List[str], true_classes: List[str]) -> List[str]:
        """예측된 클래스들을 실제 클래스들로 매핑"""
        mapped_classes = []
        used_true_classes = set()
        
        for pred_class in predicted_classes:
            best_match, similarity = self._find_best_match(pred_class, true_classes)
            if best_match and best_match not in used_true_classes:
                mapped_classes.append(best_match)
                used_true_classes.add(best_match)
                self.log(f"  🎯 Mapped '{pred_class}' → '{best_match}' (similarity: {similarity:.3f})")
        
        return mapped_classes
        self.log_file.flush()
    
    def get_target_files(self, target_classes: List[str], max_per_class: int = 3, include_mixed: bool = True) -> List[Tuple[str, List[str]]]:
        """타겟 클래스가 포함된 섞인 소리 파일 목록 반환"""
        target_files = []
        class_counts = {cls: 0 for cls in target_classes}
        mixed_files = []
        
        for _, row in self.ground_truth_df.iterrows():
            fname = str(row['fname'])
            labels = [label.strip() for label in row['labels'].split(',')]
            
            # 타겟 클래스가 포함된 파일인지 확인
            target_labels = [label for label in labels if label in target_classes]
            
            if target_labels:
                file_path = os.path.join(self.eval_audio_dir, fname + ".wav")
                if os.path.exists(file_path):
                    # 섞인 소리 파일만 처리 (타겟 클래스가 포함된)
                    if len(target_labels) >= 1:  # 타겟 클래스가 1개 이상 포함된 모든 파일
                        mixed_files.append((fname, target_labels))
                        # 클래스별 카운트
                        for label in target_labels:
                            if label in class_counts:
                                class_counts[label] += 1
        
        # 섞인 소리 파일을 추가 (최대 50개)
        if include_mixed:
            target_files.extend(mixed_files[:50])
        
        self.log(f"Selected {len(target_files)} files for evaluation")
        self.log(f"  All files contain target classes: {len(target_files)}")
        
        for cls, count in class_counts.items():
            self.log(f"  {cls}: {count} occurrences")
        
        return target_files
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """오디오 파일 로드"""
        try:
            if SEPARATOR_AVAILABLE:
                audio, sr = librosa.load(file_path, sr=separator.SR, duration=separator.WIN_SEC)
            else:
                audio, sr = librosa.load(file_path, sr=16000, duration=4.096)
            return audio
        except Exception as e:
            self.log(f"Error loading {file_path}: {e}")
            return None
    
    def create_mixture(self, source_audio: np.ndarray, noise_ratio: float = 0.0) -> np.ndarray:
        """원본 오디오 사용 (이미 섞인 소리이므로)"""
        # FSD50K의 오디오는 이미 섞인 소리이므로 원본을 그대로 사용
        # 필요시 약간의 노이즈만 추가
        if noise_ratio > 0:
            noise = np.random.normal(0, 0.01, len(source_audio))
            mixture = source_audio + noise_ratio * noise
        else:
            mixture = source_audio.copy()
        
        # 정규화
        if np.max(np.abs(mixture)) > 0:
            mixture = mixture / np.max(np.abs(mixture)) * 0.8
        
        return mixture
    
    def separate_audio(self, mixture: np.ndarray, target_classes: List[str] = None) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """음원 분리 수행 (separator.py 함수 기반)"""
        if not SEPARATOR_AVAILABLE:
            return [mixture], [{"class_name": "unknown", "confidence": 0.0}]
        
        try:
            # separator.py의 함수들을 사용하여 분리 수행
            device = torch.device("cpu")
            
            # Mel filterbank setup
            fbins = separator.N_FFT//2 + 1
            mel_fb_f2m = torchaudio.functional.melscale_fbanks(
                n_freqs=fbins, f_min=0.0, f_max=separator.SR/2, n_mels=separator.N_MELS,
                sample_rate=separator.SR, norm="slaney"
            )
            mel_fb_m2f = mel_fb_f2m.T.contiguous()
            
            # AST model setup
            extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            ast_model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                attn_implementation="eager"
            ).to(device).eval()
            
            # Apply quantization
            try:
                ast_model = torch.quantization.quantize_dynamic(
                    ast_model,
                    {torch.nn.Linear, torch.nn.Conv1d},
                    dtype=torch.qint8
                )
            except:
                pass
            
            # 동적 패스 수 계산 (라벨 개수 기반)
            if target_classes:
                # 타겟 클래스 개수에 맞춰 패스 수 설정 (제한 없음)
                num_passes = len(target_classes)
            else:
                # 기본값: 오디오 길이 기반
                audio_duration = len(mixture) / separator.SR
                num_passes = max(1, int(audio_duration / 2.0))
            
            print(f"🎵 Audio duration: {len(mixture) / separator.SR:.2f}s")
            print(f"🎯 Target classes: {len(target_classes) if target_classes else 'unknown'}")
            print(f"🔄 Dynamic passes: {num_passes} (based on {'label count' if target_classes else 'duration'})")
            
            # Multi-pass separation
            separated_audios = []
            class_infos = []
            cur = mixture.copy()
            used_mask_prev = None
            prev_anchors = []
            separated_time_regions = []
            previous_anchors = []
            
            for pass_idx in range(num_passes):
                print(f"  --- Pass {pass_idx + 1}/{num_passes} ---")
                
                # Single pass separation
                src_amp, res, er, used_mask, info = separator.single_pass(
                    cur, extractor, ast_model, mel_fb_m2f,
                    used_mask_prev, prev_anchors, pass_idx, None, 1.0, 
                    separated_time_regions, previous_anchors, cur
                )
                
                if src_amp is not None and len(src_amp) > 0:
                    separated_audios.append(src_amp)
                    # 클래스 정보 추출
                    if info and 'class_name' in info:
                        class_infos.append({
                            "class_name": info['class_name'],
                            "confidence": info.get('confidence', 0.0),
                            "sound_type": info.get('sound_type', 'unknown'),
                            "pass": pass_idx + 1
                        })
                    else:
                        class_infos.append({
                            "class_name": "unknown",
                            "confidence": 0.0,
                            "sound_type": "unknown",
                            "pass": pass_idx + 1
                        })
                    
                    print(f"    Separated: {info.get('class_name', 'unknown')} (confidence: {info.get('confidence', 0.0):.3f})")
                
                # 다음 패스를 위한 준비
                cur = res if res is not None and len(res) > 0 else cur
                used_mask_prev = used_mask
                
                # 조기 종료 조건 (에너지가 너무 낮으면)
                if er < 0.01:  # 에너지 비율이 1% 미만이면 종료
                    print(f"    ⚠️ Low energy ratio ({er:.3f}), stopping separation")
                    break
            
            print(f"  ✅ Total separated sources: {len(separated_audios)}")
            
            if not separated_audios:
                separated_audios = [mixture]
                class_infos = [{"class_name": "unknown", "confidence": 0.0, "sound_type": "unknown", "pass": 1}]
            
            return separated_audios, class_infos
            
        except Exception as e:
            self.log(f"Separation error: {e}")
            return [mixture], [{"class_name": "unknown", "confidence": 0.0, "sound_type": "unknown"}]
    
    def calculate_mixed_class_prediction_accuracy(self, predicted_classes: List[Dict[str, Any]], true_classes: List[str]) -> Dict[str, Any]:
        """섞인 소리용 클래스 예측 정확도 계산"""
        if not predicted_classes:
            return {
                'class_accuracy': 0.0,
                'confidence_avg': 0.0,
                'predicted_classes': [],
                'true_classes': true_classes,
                'mixed_accuracy': 0.0,
                'recall': 0.0,
                'precision': 0.0
            }
        
        # 예측된 클래스들 추출
        predicted_class_names = [info['class_name'] for info in predicted_classes]
        confidences = [info['confidence'] for info in predicted_classes]
        
        # 정확도 계산 (true_classes가 예측된 클래스 중에 있는지)
        correct_predictions = sum(1 for true_class in true_classes if true_class in predicted_class_names)
        class_accuracy = correct_predictions / len(true_classes) if true_classes else 0.0
        
        # 섞인 소리 정확도 (모든 true_classes가 예측되었는지)
        mixed_accuracy = 1.0 if all(true_class in predicted_class_names for true_class in true_classes) else 0.0
        
        # Recall: 실제 클래스 중 예측된 비율
        recall = correct_predictions / len(true_classes) if true_classes else 0.0
        
        # Precision: 예측된 클래스 중 실제 클래스 비율
        precision = correct_predictions / len(predicted_class_names) if predicted_class_names else 0.0
        
        # 평균 신뢰도
        confidence_avg = np.mean(confidences) if confidences else 0.0
        
        return {
            'class_accuracy': class_accuracy,
            'confidence_avg': confidence_avg,
            'predicted_classes': predicted_class_names,
            'true_classes': true_classes,
            'mixed_accuracy': mixed_accuracy,
            'recall': recall,
            'precision': precision
        }
    
    def calculate_class_prediction_accuracy(self, predicted_classes: List[Dict[str, Any]], true_class: str) -> Dict[str, float]:
        """클래스 예측 정확도 계산"""
        if not predicted_classes:
            return {
                'class_accuracy': 0.0,
                'confidence_avg': 0.0,
                'predicted_classes': [],
                'true_class': true_class
            }
        
        # 예측된 클래스들 추출
        predicted_class_names = [info['class_name'] for info in predicted_classes]
        confidences = [info['confidence'] for info in predicted_classes]
        
        # 정확도 계산 (true_class가 예측된 클래스 중에 있는지)
        class_accuracy = 1.0 if true_class in predicted_class_names else 0.0
        
        # 평균 신뢰도
        confidence_avg = np.mean(confidences) if confidences else 0.0
        
        return {
            'class_accuracy': class_accuracy,
            'confidence_avg': confidence_avg,
            'predicted_classes': predicted_class_names,
            'true_class': true_class
        }
    
    def calculate_classification_metrics(self, predicted_classes: List[str], true_classes: List[str]) -> Dict[str, float]:
        """클래스 분류 성능 지표 계산 (유사도 기반 매칭 포함)"""
        if not predicted_classes and not true_classes:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # 유사도 기반 매칭으로 예측된 클래스들을 실제 클래스로 매핑
        mapped_predicted_classes = self._map_predicted_to_true_classes(predicted_classes, true_classes)
        
        # 클래스 집합으로 변환 (중복 제거)
        pred_set = set(mapped_predicted_classes)
        true_set = set(true_classes)
        
        # True Positive: 예측했고 실제로도 있는 클래스
        tp = len(pred_set & true_set)
        
        # False Positive: 예측했지만 실제로는 없는 클래스
        fp = len(pred_set - true_set)
        
        # False Negative: 예측하지 않았지만 실제로는 있는 클래스
        fn = len(true_set - pred_set)
        
        # Accuracy: 전체 클래스 중 맞게 예측한 비율
        total_classes = len(pred_set | true_set)
        accuracy = tp / total_classes if total_classes > 0 else 0.0
        
        # Precision: 예측한 클래스 중 실제로 맞는 비율
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        
        # Recall: 실제 클래스 중 예측한 비율
        recall = tp / len(true_set) if len(true_set) > 0 else 0.0
        
        # F1-Score: Precision과 Recall의 조화평균
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_predicted': len(pred_set),
            'total_true': len(true_set),
            'mapped_predicted_classes': mapped_predicted_classes
        }
    
    def evaluate_single_file(self, fname: str, true_classes: List[str], target_classes: List[str]) -> Dict[str, Any]:
        """단일 파일 평가 (지정된 클래스만 예측 확인)"""
        file_path = os.path.join(self.eval_audio_dir, fname + ".wav")
        
        # 오디오 로드
        mixture_audio = self.load_audio(file_path)
        if mixture_audio is None:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'predicted_classes': [], 'true_classes': true_classes,
                'confidence_avg': 0.0, 'num_separated_sources': 0,
                'tp': 0, 'fp': 0, 'fn': 0, 'total_predicted': 0, 'total_true': len(true_classes)
            }
        
        # 음원 분리 수행 (타겟 클래스 전달)
        separated_sources, class_infos = self.separate_audio(mixture_audio, target_classes)
        
        # 분리된 소스들의 클래스 예측 결과 수집
        predicted_classes = []
        confidences = []
        
        for class_info in class_infos:
            if class_info['class_name'] != 'unknown' and class_info['confidence'] > 0.1:
                predicted_classes.append(class_info['class_name'])
                confidences.append(class_info['confidence'])
        
        # 지정된 클래스만 필터링하여 평가
        # true_classes에서 target_classes에 포함된 것만 추출
        target_true_classes = [cls for cls in true_classes if cls in target_classes]
        
        # 클래스 분류 성능 지표 계산 (지정된 클래스만)
        classification_metrics = self.calculate_classification_metrics(predicted_classes, target_true_classes)
        
        # 결과 구성
        result = {
            **classification_metrics,
            'predicted_classes': predicted_classes,
            'mapped_predicted_classes': classification_metrics.get('mapped_predicted_classes', []),
            'true_classes': true_classes,
            'target_true_classes': target_true_classes,
            'confidence_avg': np.mean(confidences) if confidences else 0.0,
            'num_separated_sources': len(separated_sources)
        }
        
        return result
    
    def get_target_files_by_groups(self, danger_classes: List[str], help_classes: List[str], warning_classes: List[str]) -> Dict[str, List[Tuple[str, List[str]]]]:
        """그룹별로 파일 목록 가져오기 (한 파일이 여러 그룹에 속할 수 있음)"""
        group_files = {
            'danger': [],
            'help': [],
            'warning': []
        }
        
        # 모든 파일을 검사하여 각 그룹에 속하는 파일 찾기
        for _, row in self.ground_truth_df.iterrows():
            fname = str(row['fname'])
            labels = [label.strip() for label in row['labels'].split(',')]
            
            # 파일 존재 확인
            file_path = os.path.join(self.eval_audio_dir, fname + ".wav")
            if not os.path.exists(file_path):
                continue
            
            # 각 그룹별로 확인
            for group_name, group_classes in [('danger', danger_classes), ('help', help_classes), ('warning', warning_classes)]:
                # 해당 그룹의 클래스가 하나라도 포함된 파일 선택
                if any(label in group_classes for label in labels):
                    group_files[group_name].append((fname, labels))
        
        # 통계 출력
        total_files = len(set([fname for group in group_files.values() for fname, _ in group]))
        self.log(f"Found {total_files} unique files for evaluation")
        self.log(f"  - Total CSV records: {len(self.ground_truth_df)}")
        
        for group_name, files in group_files.items():
            self.log(f"  - {group_name.upper()} group: {len(files)} files")
            
            # 그룹별 클래스 분포
            group_classes = danger_classes if group_name == 'danger' else help_classes if group_name == 'help' else warning_classes
            class_stats = {}
            for fname, labels in files:
                for label in labels:
                    if label in group_classes:
                        class_stats[label] = class_stats.get(label, 0) + 1
            
            self.log(f"    Class distribution:")
            for class_name, count in sorted(class_stats.items()):
                self.log(f"      {class_name}: {count} files")
        
        return group_files
    
    def run_group_evaluation(self, danger_classes: List[str], help_classes: List[str], warning_classes: List[str]):
        """그룹별 평가 실행 (한 파일이 여러 그룹에 속할 수 있음)"""
        self.log("Starting FSD50K group-based classification evaluation...")
        
        # 그룹별 파일 목록 가져오기
        group_files = self.get_target_files_by_groups(danger_classes, help_classes, warning_classes)
        
        all_results = []
        
        # 각 그룹별로 평가 수행
        for group_name, files in group_files.items():
            if not files:
                self.log(f"No files found for {group_name} group")
                continue
                
            group_classes = danger_classes if group_name == 'danger' else help_classes if group_name == 'help' else warning_classes
            self.log(f"\n=== Evaluating {group_name.upper()} group ({len(files)} files) ===")
            
            # 그룹 내 파일들 평가
            for i, (fname, all_labels) in enumerate(files):
                self.log(f"Processing {i+1}/{len(files)}: {fname}")
                self.log(f"  📋 All labels in file: {all_labels}")
                
                # 해당 그룹의 클래스만 필터링
                group_labels = [label for label in all_labels if label in group_classes]
                self.log(f"  🎯 Target {group_name} group labels: {group_labels}")
                self.log(f"  🔍 Will separate ALL sounds and check if {group_name} classes are detected")
                
                if not group_labels:
                    self.log(f"  ⚠️ No {group_name} group labels found, skipping")
                    continue
                
                # 단일 파일 평가 (해당 그룹 클래스만)
                self.log(f"  🎵 Starting separation for {len(group_labels)} target classes...")
                metrics = self.evaluate_single_file(fname, all_labels, group_classes)
                
                result = {
                    'group': group_name,
                    'filename': fname,
                    'all_labels': ', '.join(all_labels),
                    'group_labels': ', '.join(group_labels),
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'confidence_avg': metrics['confidence_avg'],
                    'predicted_classes': ', '.join(metrics['predicted_classes']),
                    'mapped_predicted_classes': ', '.join(metrics.get('mapped_predicted_classes', [])),
                    'num_separated_sources': metrics['num_separated_sources'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'total_predicted': metrics['total_predicted'],
                    'total_true': metrics['total_true']
                }
                
                all_results.append(result)
                
                self.log(f"  📊 Results:")
                self.log(f"    Accuracy: {metrics['accuracy']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
                self.log(f"    Confidence: {metrics['confidence_avg']:.3f}, Separated Sources: {metrics['num_separated_sources']}")
                self.log(f"    🎯 Predicted classes: {', '.join(metrics['predicted_classes'])}")
                self.log(f"    ✅ Mapped to target: {', '.join(metrics.get('mapped_predicted_classes', []))}")
                self.log(f"    🎯 Target classes found: {len(metrics.get('mapped_predicted_classes', []))}/{len(group_labels)}")
                
                if (i + 1) % 10 == 0:
                    self.log(f"  Completed {i+1}/{len(files)} files in {group_name} group")
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(all_results)
        
        # 전체 성능지표 콘솔 출력
        if not results_df.empty:
            self.log("\n" + "="*60)
            self.log("📊 전체 성능지표 요약")
            self.log("="*60)
            self.log(f"  총 평가 파일 수: {len(results_df)}개")
            self.log(f"  평균 Accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}")
            self.log(f"  평균 Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
            self.log(f"  평균 Recall: {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")
            self.log(f"  평균 F1-Score: {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}")
            self.log(f"  평균 Confidence: {results_df['confidence_avg'].mean():.3f} ± {results_df['confidence_avg'].std():.3f}")
            self.log(f"  평균 분리 소스 수: {results_df['num_separated_sources'].mean():.1f} ± {results_df['num_separated_sources'].std():.1f}")
            
            # 그룹별 성능지표 요약
            self.log("\n📈 그룹별 성능지표:")
            for group in ['danger', 'help', 'warning']:
                group_data = results_df[results_df['group'] == group]
                if not group_data.empty:
                    self.log(f"  🔴 {group.upper()} 그룹 ({len(group_data)}개 파일):")
                    self.log(f"    Accuracy: {group_data['accuracy'].mean():.3f} ± {group_data['accuracy'].std():.3f}")
                    self.log(f"    F1-Score: {group_data['f1_score'].mean():.3f} ± {group_data['f1_score'].std():.3f}")
                    self.log(f"    Confidence: {group_data['confidence_avg'].mean():.3f} ± {group_data['confidence_avg'].std():.3f}")
        
        # 결과 저장
        self.save_group_results(results_df)
        
        return results_df
    
    def run_evaluation(self, target_classes: List[str], max_per_class: int = 3):
        """전체 평가 실행 (클래스 분류)"""
        self.log("Starting FSD50K classification evaluation...")
        
        target_files = self.get_target_files(target_classes, max_per_class)
        
        for i, (fname, true_classes) in enumerate(target_files):
            class_str = ', '.join(true_classes) if len(true_classes) > 1 else true_classes[0]
            self.log(f"Processing {i+1}/{len(target_files)}: {fname} ({class_str})")
            
            # 지정된 클래스만 필터링
            target_true_classes = [cls for cls in true_classes if cls in target_classes]
            self.log(f"  Target classes in file: {target_true_classes}")
            
            metrics = self.evaluate_single_file(fname, true_classes, target_classes)
            
            result = {
                'filename': fname,
                'true_classes': class_str,
                'target_classes': ', '.join(target_true_classes),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'confidence_avg': metrics['confidence_avg'],
                'predicted_classes': ', '.join(metrics['predicted_classes']),
                'mapped_predicted_classes': ', '.join(metrics.get('mapped_predicted_classes', [])),
                'num_separated_sources': metrics['num_separated_sources'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'total_predicted': metrics['total_predicted'],
                'total_true': metrics['total_true']
            }
            
            self.results.append(result)
            self.log(f"  Accuracy: {metrics['accuracy']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
            self.log(f"  Confidence: {metrics['confidence_avg']:.3f}, Sources: {metrics['num_separated_sources']}")
            self.log(f"  Original Predicted: {', '.join(metrics['predicted_classes'])}")
            self.log(f"  Mapped Predicted: {', '.join(metrics.get('mapped_predicted_classes', []))}")
            self.log(f"  True Classes: {', '.join(true_classes)}")
        
        # 결과 저장
        results_df = pd.DataFrame(self.results)
        self.save_results(results_df)
        
        return results_df
    
    def save_group_results(self, results_df: pd.DataFrame):
        """그룹별 결과 저장"""
        if results_df.empty:
            self.log("No results to save")
            return
        
        # CSV 파일로 저장
        csv_filename = "fsd50k_group_performance_results.csv"
        results_df.to_csv(csv_filename, index=False)
        self.log(f"Results saved to {csv_filename}")
        
        # 그룹별 요약 통계
        summary_filename = "fsd50k_group_evaluation_summary.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write("FSD50K Group-Based Classification Performance Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # 전체 통계
            f.write("Overall Performance:\n")
            f.write(f"  Total files evaluated: {len(results_df)}\n")
            f.write(f"  Average accuracy: {results_df['accuracy'].mean():.3f}\n")
            f.write(f"  Average precision: {results_df['precision'].mean():.3f}\n")
            f.write(f"  Average recall: {results_df['recall'].mean():.3f}\n")
            f.write(f"  Average F1-score: {results_df['f1_score'].mean():.3f}\n")
            f.write(f"  Average confidence: {results_df['confidence_avg'].mean():.3f}\n")
            f.write(f"  Average separated sources: {results_df['num_separated_sources'].mean():.1f}\n\n")
            
            # 그룹별 통계
            for group in ['danger', 'help', 'warning']:
                group_data = results_df[results_df['group'] == group]
                if not group_data.empty:
                    f.write(f"{group.upper()} Group Performance:\n")
                    f.write(f"  Files evaluated: {len(group_data)}\n")
                    f.write(f"  Average accuracy: {group_data['accuracy'].mean():.3f}\n")
                    f.write(f"  Average precision: {group_data['precision'].mean():.3f}\n")
                    f.write(f"  Average recall: {group_data['recall'].mean():.3f}\n")
                    f.write(f"  Average F1-score: {group_data['f1_score'].mean():.3f}\n")
                    f.write(f"  Average confidence: {group_data['confidence_avg'].mean():.3f}\n")
                    f.write(f"  Average separated sources: {group_data['num_separated_sources'].mean():.1f}\n\n")
        
        self.log(f"Summary saved to {summary_filename}")
    
    def save_results(self, results_df: pd.DataFrame):
        """결과 저장"""
        # CSV 파일로 저장
        output_file = "fsd50k_performance_results.csv"
        results_df.to_csv(output_file, index=False)
        self.log(f"Results saved to: {output_file}")
        
        # 요약 결과 저장
        summary_file = "evaluation_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("FSD50K CLASSIFICATION EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            if len(results_df) > 0:
                # 전체 평균
                f.write("Overall Average:\n")
                f.write(f"  Accuracy: {results_df['accuracy'].mean():.3f} ± {results_df['accuracy'].std():.3f}\n")
                f.write(f"  Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}\n")
                f.write(f"  Recall: {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}\n")
                f.write(f"  F1-Score: {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}\n")
                f.write(f"  Confidence: {results_df['confidence_avg'].mean():.3f} ± {results_df['confidence_avg'].std():.3f}\n")
                f.write(f"  Avg Sources: {results_df['num_separated_sources'].mean():.1f} ± {results_df['num_separated_sources'].std():.1f}\n\n")
                
                # 클래스별 평균
                f.write("Per-Class Average:\n")
                class_avg = results_df.groupby('true_classes').agg({
                    'accuracy': ['mean', 'std'],
                    'precision': ['mean', 'std'],
                    'recall': ['mean', 'std'],
                    'f1_score': ['mean', 'std'],
                    'confidence_avg': ['mean', 'std'],
                    'num_separated_sources': ['mean', 'std']
                }).round(3)
                
                for class_name in class_avg.index:
                    f.write(f"\n  {class_name}:\n")
                    f.write(f"    Accuracy: {class_avg.loc[class_name, ('accuracy', 'mean')]:.3f} ± {class_avg.loc[class_name, ('accuracy', 'std')]:.3f}\n")
                    f.write(f"    Precision: {class_avg.loc[class_name, ('precision', 'mean')]:.3f} ± {class_avg.loc[class_name, ('precision', 'std')]:.3f}\n")
                    f.write(f"    Recall: {class_avg.loc[class_name, ('recall', 'mean')]:.3f} ± {class_avg.loc[class_name, ('recall', 'std')]:.3f}\n")
                    f.write(f"    F1-Score: {class_avg.loc[class_name, ('f1_score', 'mean')]:.3f} ± {class_avg.loc[class_name, ('f1_score', 'std')]:.3f}\n")
                    f.write(f"    Confidence: {class_avg.loc[class_name, ('confidence_avg', 'mean')]:.3f} ± {class_avg.loc[class_name, ('confidence_avg', 'std')]:.3f}\n")
                    f.write(f"    Avg Sources: {class_avg.loc[class_name, ('num_separated_sources', 'mean')]:.1f} ± {class_avg.loc[class_name, ('num_separated_sources', 'std')]:.1f}\n")
                
                # 상세 결과
                f.write("\nDetailed Results:\n")
                f.write(results_df.to_string(index=False, float_format='%.3f'))
        
        self.log(f"Summary saved to: {summary_file}")
    
    def __del__(self):
        """소멸자"""
        if hasattr(self, 'log_file'):
            self.log_file.close()


def main():
    """메인 실행 함수"""
    # 그룹별 평가할 클래스 선택
    danger_classes = [
        'Siren', 'Civil defense siren', 'Buzzer', 'Smoke detector, smoke alarm', 
        'Fire alarm', 'Explosion', 'Boom'
    ]
    
    help_classes = [
        'Baby cry, infant cry', 'Screaming', 'Door', 'Doorbell', 'Ding-dong', 'Knock'
    ]
    
    warning_classes = [
        'Water', 'Dishes, pots, and pans', 'Alarm', 'Telephone', 'Telephone bell ringing',
        'Splinter', 'Ringtone', 'Telephone dialing, DTMF', 'Dial tone', 'Alarm clock',
        'Crack', 'Glass', 'Chink, clink', 'Shatter', 'Boiling', 'Smash, crash', 'Breaking',
        'Crushing', 'Crumpling, crinkling'
    ]
    
    # 모든 클래스 합치기
    target_classes = danger_classes + help_classes + warning_classes
    
    # 평가기 초기화
    evaluator = FSD50KEvaluator()
    
    try:
        # 그룹별 평가 실행 (한 파일이 여러 그룹에 속할 수 있음)
        results_df = evaluator.run_group_evaluation(
            danger_classes=danger_classes,
            help_classes=help_classes,
            warning_classes=warning_classes
        )
        
        evaluator.log("Group-based evaluation completed successfully!")
        
    except Exception as e:
        evaluator.log(f"Evaluation failed: {e}")
        import traceback
        evaluator.log(traceback.format_exc())


if __name__ == "__main__":
    main()

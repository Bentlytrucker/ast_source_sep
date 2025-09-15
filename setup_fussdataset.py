#!/usr/bin/env python3
"""
FUSS 데이터셋 설정 스크립트

이 스크립트는:
1. FUSS 데이터셋을 자동으로 다운로드
2. 필요한 디렉토리 구조 생성
3. 데이터셋 검증 및 구조 분석
4. 평가를 위한 환경 설정

사용법:
python setup_fuss_dataset.py --root_dir ./fuss_data
"""

import os
import sys
import argparse
import subprocess
import requests
import tarfile
from pathlib import Path
from typing import Optional
import shutil

class FUSSDatasetSetup:
    """FUSS 데이터셋 설정 클래스"""
    
    def __init__(self, root_dir: str = "./fuss_data"):
        self.root_dir = Path(root_dir)
        self.download_dir = self.root_dir / "download"
        self.data_dir = self.root_dir / "fuss_dev"
        
        # FUSS 데이터셋 URL
        self.urls = {
            'ssdata': "https://zenodo.org/record/3743844/files/FUSS_ssdata.tar.gz",
            'ssdata_reverb': "https://zenodo.org/record/3743844/files/FUSS_ssdata_reverb.tar.gz",
            'fsd_data': "https://zenodo.org/record/3743844/files/FUSS_fsd_data.tar.gz",
            'rir_data': "https://zenodo.org/record/3743844/files/FUSS_rir_data.tar.gz"
        }
        
        # 파일 크기 정보 (MB)
        self.file_sizes = {
            'ssdata': 1200,  # 약 1.2GB
            'ssdata_reverb': 1200,  # 약 1.2GB
            'fsd_data': 800,  # 약 800MB
            'rir_data': 200   # 약 200MB
        }
    
    def create_directories(self):
        """필요한 디렉토리 생성"""
        print("디렉토리 구조 생성 중...")
        
        directories = [
            self.root_dir,
            self.download_dir,
            self.data_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {directory}")
    
    def download_file(self, url: str, output_path: Path, expected_size_mb: int) -> bool:
        """파일 다운로드"""
        print(f"다운로드 중: {url}")
        print(f"저장 위치: {output_path}")
        print(f"예상 크기: {expected_size_mb}MB")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # 진행률 표시
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r  진행률: {progress:.1f}%", end='', flush=True)
            
            print(f"\n  ✓ 다운로드 완료: {output_path}")
            return True
            
        except Exception as e:
            print(f"\n  ✗ 다운로드 실패: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """압축 파일 해제"""
        print(f"압축 해제 중: {archive_path}")
        
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            
            print(f"  ✓ 압축 해제 완료: {extract_dir}")
            return True
            
        except Exception as e:
            print(f"  ✗ 압축 해제 실패: {e}")
            return False
    
    def download_fuss_data(self, components: list = None, force_download: bool = False):
        """FUSS 데이터셋 다운로드"""
        if components is None:
            components = ['ssdata', 'ssdata_reverb']  # 기본적으로 필요한 컴포넌트만
        
        print(f"FUSS 데이터셋 다운로드 시작 (컴포넌트: {components})")
        
        for component in components:
            if component not in self.urls:
                print(f"  ✗ 알 수 없는 컴포넌트: {component}")
                continue
            
            archive_path = self.download_dir / f"{component}.tar.gz"
            extract_path = self.data_dir
            
            # 이미 존재하는 경우 스킵
            if archive_path.exists() and not force_download:
                print(f"  ⚠ {component} 아카이브가 이미 존재합니다. 스킵합니다.")
                continue
            
            # 다운로드
            if not self.download_file(self.urls[component], archive_path, self.file_sizes[component]):
                continue
            
            # 압축 해제
            if not self.extract_archive(archive_path, extract_path):
                continue
        
        print("FUSS 데이터셋 다운로드 완료!")
    
    def verify_dataset(self) -> bool:
        """데이터셋 검증"""
        print("데이터셋 검증 중...")
        
        required_dirs = [
            self.data_dir / "ssdata",
            self.data_dir / "ssdata_reverb"
        ]
        
        all_valid = True
        
        for required_dir in required_dirs:
            if not required_dir.exists():
                print(f"  ✗ 필수 디렉토리 누락: {required_dir}")
                all_valid = False
                continue
            
            # 각 스플릿 확인
            for split in ['train', 'validation', 'eval']:
                split_dir = required_dir / split
                if not split_dir.exists():
                    print(f"  ✗ 스플릿 디렉토리 누락: {split_dir}")
                    all_valid = False
                    continue
                
                # WAV 파일 개수 확인
                wav_files = list(split_dir.glob("*.wav"))
                print(f"  ✓ {split_dir}: {len(wav_files)}개 WAV 파일")
                
                # sources 디렉토리 확인
                sources_dir = split_dir / "sources"
                if sources_dir.exists():
                    source_files = list(sources_dir.glob("*.wav"))
                    print(f"    - sources: {len(source_files)}개 파일")
                else:
                    print(f"    - sources 디렉토리 없음")
        
        if all_valid:
            print("  ✓ 데이터셋 검증 완료")
        else:
            print("  ✗ 데이터셋 검증 실패")
        
        return all_valid
    
    def analyze_dataset(self):
        """데이터셋 구조 분석"""
        print("\n=== 데이터셋 구조 분석 ===")
        
        for dataset_name in ['ssdata', 'ssdata_reverb']:
            dataset_dir = self.data_dir / dataset_name
            if not dataset_dir.exists():
                continue
            
            print(f"\n{dataset_name}:")
            
            for split in ['train', 'validation', 'eval']:
                split_dir = dataset_dir / split
                if not split_dir.exists():
                    continue
                
                # 혼합 오디오 파일들
                mixture_files = list(split_dir.glob("*.wav"))
                
                # sources 디렉토리
                sources_dir = split_dir / "sources"
                source_files = list(sources_dir.glob("*.wav")) if sources_dir.exists() else []
                
                print(f"  {split}:")
                print(f"    - 혼합 오디오: {len(mixture_files)}개")
                print(f"    - 참조 소스: {len(source_files)}개")
                
                # 샘플 파일명 표시
                if mixture_files:
                    sample_file = mixture_files[0]
                    print(f"    - 샘플: {sample_file.name}")
    
    def install_dependencies(self):
        """필요한 의존성 설치"""
        print("필요한 의존성 설치 중...")
        
        dependencies = [
            'mir_eval',
            'soundfile',
            'pandas',
            'matplotlib',
            'numpy',
            'requests'
        ]
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                print(f"  ✓ {dep} 이미 설치됨")
            except ImportError:
                print(f"  설치 중: {dep}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"  ✓ {dep} 설치 완료")
    
    def setup(self, components: list = None, force_download: bool = False):
        """전체 설정 과정 실행"""
        print("=== FUSS 데이터셋 설정 시작 ===")
        
        # 1. 디렉토리 생성
        self.create_directories()
        
        # 2. 의존성 설치
        self.install_dependencies()
        
        # 3. 데이터셋 다운로드
        self.download_fuss_data(components, force_download)
        
        # 4. 데이터셋 검증
        if self.verify_dataset():
            # 5. 구조 분석
            self.analyze_dataset()
            
            print("\n=== 설정 완료 ===")
            print(f"데이터셋 위치: {self.data_dir}")
            print("이제 fuss_evaluation_improved.py를 실행할 수 있습니다.")
        else:
            print("\n=== 설정 실패 ===")
            print("데이터셋 검증에 실패했습니다. 다시 시도해주세요.")

def main():
    parser = argparse.ArgumentParser(description="FUSS 데이터셋 설정")
    parser.add_argument("--root_dir", default="./fuss_data",
                       help="FUSS 데이터셋 루트 디렉토리")
    parser.add_argument("--components", nargs="+", 
                       choices=['ssdata', 'ssdata_reverb', 'fsd_data', 'rir_data'],
                       default=['ssdata', 'ssdata_reverb'],
                       help="다운로드할 컴포넌트")
    parser.add_argument("--force_download", action="store_true",
                       help="기존 파일이 있어도 다시 다운로드")
    
    args = parser.parse_args()
    
    # 설정 실행
    setup = FUSSDatasetSetup(args.root_dir)
    setup.setup(args.components, args.force_download)

if __name__ == "__main__":
    main()

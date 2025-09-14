#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원하는 클래스들에 해당하는 AudioSet 데이터 다운로드
"""

from audioset_download import Downloader
import os

# claude.py에서 정의된 클래스 ID들 (AudioSet 공식 ID로 수정)
DANGER_IDS = {427, 428, 429, 430, 431, 426, 298}  # Gunshot, Machine gun, Fusillade, Artillery fire, Cap gun, Explosion, Fire
HELP_IDS = {23, 14, 3, 4, 5, 0}  # Baby cry, Screaming, Child speech, Conversation, Narration, Speech  
WARNING_IDS = {388, 396, 308, 309, 310, 331, 319, 341, 342, 326, 327, 316, 307, 321, 322, 323, 324, 325}  # Alarm, Siren, Vehicle horn, Toot, Car alarm, Train horn, Reversing beeps, Bicycle, Skateboard, Motorcycle, Traffic noise, Truck, Car, Bus, Emergency vehicle, Police car, Ambulance, Fire engine

# AudioSet 공식 클래스 ID를 이름으로 변환하는 매핑
CLASS_ID_TO_NAME = {
    # Danger classes
    427: "Gunshot, gunfire",      # 396 -> 427
    428: "Machine gun",           # 397 -> 428  
    429: "Fusillade",             # 398 -> 429
    430: "Artillery fire",        # 399 -> 430
    431: "Cap gun",               # 400 -> 431
    426: "Explosion",             # 426 -> 426 (동일)
    298: "Fire",                  # 436 -> 298
    
    # Help classes  
    23: "Baby cry, infant cry",   # 23 -> 23 (동일)
    14: "Screaming",              # 14 -> 14 (동일)
    3: "Child speech, kid speaking",  # 354 -> 3
    4: "Conversation",            # 355 -> 4
    5: "Narration, monologue",    # 356 -> 5
    0: "Speech",                  # 359 -> 0
    
    # Warning classes
    388: "Alarm",                 # 288 -> 388
    396: "Siren",                 # 364 -> 396
    308: "Vehicle horn, car horn, honking",  # 388 -> 308
    309: "Toot",                  # 389 -> 309
    310: "Car alarm",             # 390 -> 310
    331: "Train horn",            # 439 -> 331
    319: "Reversing beeps",       # 391 -> 319
    341: "Bicycle",               # 392 -> 341
    342: "Skateboard",            # 393 -> 342
    326: "Motorcycle",            # 395 -> 326
    327: "Traffic noise, roadway noise",  # 440 -> 327
    316: "Truck",                 # 441 -> 316
    307: "Car",                   # 443 -> 307
    321: "Bus",                   # 456 -> 321
    322: "Emergency vehicle",     # 469 -> 322
    323: "Police car (siren)",    # 470 -> 323
    324: "Ambulance (siren)",     # 478 -> 324
    325: "Fire engine, fire truck (siren)"  # 479 -> 325
}

def get_target_labels():
    """다운로드할 타겟 라벨들을 반환"""
    # 사용자가 지정한 클래스들
    labels = [
        # Siren & Alarm classes
        "Siren",
        "Civil defense siren", 
        "Buzzer",
        "Smoke detector, smoke alarm",
        "Fire alarm",
        "Alarm",
        "Alarm clock",
        
        # Explosion & Breaking sounds
        "Explosion",
        "Boom",
        "Splinter",
        "Crack",
        "Glass",
        "Chink, clink",
        "Shatter",
        "Smash, crash",
        "Breaking",
        "Crushing",
        "Crumpling, crinkling",
        
        # Human sounds
        "Baby cry, infant cry",
        "Screaming",
        
        # Door & Household sounds
        "Door",
        "Doorbell",
        "Ding-dong",
        "Knock",
        "Water",
        "Dishes, pots, and pans",
        "Boiling",
        
        # Telephone sounds
        "Telephone",
        "Telephone bell ringing",
        "Ringtone",
        "Telephone dialing, DTMF",
        "Dial tone"
    ]
    
    return labels

def download_target_classes():
    """타겟 클래스들 다운로드"""
    labels = get_target_labels()
    
    print(f"다운로드할 클래스 수: {len(labels)}")
    print("클래스 목록:")
    for i, label in enumerate(labels, 1):
        print(f"  {i:2d}. {label}")
    
    # 다운로드 디렉토리 설정
    download_dir = "audioset_data/target_classes"
    os.makedirs(download_dir, exist_ok=True)
    
    # Downloader 초기화
    d = Downloader(
        root_path=download_dir,
        labels=labels,
        n_jobs=2
    )
    
    print(f"\n다운로드 시작... (디렉토리: {download_dir})")
    
    try:
        # Strong labels 다운로드 (train, eval 세트)
        d.download_strong(
            root_path=download_dir,
            format='wav',
            download_sets=['train', 'eval']
        )
        print("다운로드 완료!")
        return True
        
    except Exception as e:
        print(f"다운로드 실패: {e}")
        return False

if __name__ == "__main__":
    download_target_classes()

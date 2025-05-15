# generate_test_eeg_data.py

import numpy as np
import pandas as pd
import mne
from datetime import datetime

def generate_sample_eeg_data(duration=10, sfreq=256, n_channels=4):
    """
    샘플 EEG 데이터를 생성합니다.
    
    Parameters:
    -----------
    duration : float
        데이터 길이 (초)
    sfreq : float
        샘플링 주파수 (Hz)
    n_channels : int
        채널 수
    
    Returns:
    --------
    pd.DataFrame
        생성된 EEG 데이터
    """
    # 시간 벡터 생성
    t = np.arange(0, duration, 1/sfreq)
    n_samples = len(t)
    
    # 채널 이름 설정
    ch_names = ['F3', 'F4', 'C3', 'C4']
    
    # 기본 신호 생성 (알파 파동 + 노이즈)
    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        # 알파 파동 (10 Hz)
        alpha = 5 * np.sin(2 * np.pi * 10 * t)
        # 베타 파동 (20 Hz)
        beta = 2 * np.sin(2 * np.pi * 20 * t)
        # 세타 파동 (5 Hz)
        theta = 3 * np.sin(2 * np.pi * 5 * t)
        # 노이즈
        noise = np.random.normal(0, 1, n_samples)
        
        # 채널별 특성 부여
        if i % 2 == 0:  # 좌측 채널
            data[i] = alpha + beta + theta + noise
        else:  # 우측 채널
            data[i] = alpha + beta + theta + noise + 2
    
    # 데이터프레임 생성
    df = pd.DataFrame(data.T, columns=ch_names)
    
    # 파일 저장
    filename = f'test_eeg_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(filename, index=False)
    print(f'생성된 데이터가 {filename}에 저장되었습니다.')
    
    return df

def generate_sample_events(n_events=5, sfreq=256, duration=10):
    """
    샘플 이벤트 데이터를 생성합니다.
    
    Parameters:
    -----------
    n_events : int
        생성할 이벤트 수
    sfreq : float
        샘플링 주파수 (Hz)
    duration : float
        데이터 길이 (초)
    
    Returns:
    --------
    pd.DataFrame
        생성된 이벤트 데이터
    """
    # 이벤트 시간 생성 (균등 분포)
    event_times = np.linspace(1, duration-1, n_events)
    event_samples = (event_times * sfreq).astype(int)
    
    # 이벤트 데이터 생성 [샘플, 이전 값, 이벤트 ID]
    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = event_samples  # 샘플
    events[:, 1] = 0  # 이전 값
    events[:, 2] = np.random.randint(1, 3, n_events)  # 이벤트 ID (1 또는 2)
    
    # 데이터프레임 생성
    df = pd.DataFrame(events, columns=['sample', 'previous', 'event_id'])
    
    # 파일 저장
    filename = f'test_events_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(filename, index=False)
    print(f'생성된 이벤트가 {filename}에 저장되었습니다.')
    
    return df

if __name__ == '__main__':
    # 샘플 데이터 생성
    eeg_data = generate_sample_eeg_data(duration=10, sfreq=256, n_channels=4)
    events = generate_sample_events(n_events=5, sfreq=256, duration=10)

    # 테스트 데이터 생성 및 저장
    test_data = generate_sample_eeg_data()
    test_data.to_csv('test_eeg_data.csv', index=False)  # 헤더 포함 저장
    print("테스트 데이터가 생성되었습니다.")
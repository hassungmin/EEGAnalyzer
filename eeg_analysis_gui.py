import sys
import numpy as np
import pandas as pd
import mne
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class EEGAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('EEG 데이터 분석')
        self.setGeometry(100, 100, 1200, 800)
        
        # 데이터 저장 변수
        self.raw_data = None
        self.events = None
        self.epochs = None
        
        # 메인 위젯 설정
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 컨트롤 패널
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # 데이터 로드 버튼
        self.load_data_btn = QPushButton('EEG 데이터 로드')
        self.load_data_btn.clicked.connect(self.load_eeg_data)
        control_layout.addWidget(self.load_data_btn)
        
        self.load_events_btn = QPushButton('이벤트 데이터 로드')
        self.load_events_btn.clicked.connect(self.load_events)
        control_layout.addWidget(self.load_events_btn)
        
        # 필터 설정
        filter_group = QWidget()
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.addWidget(QLabel('필터 설정'))
        
        # 필터 타입 선택
        self.filter_type_combo = QComboBox()
        self.filter_types = {
            'Band-pass': {
                'desc': '주요 대역 추출 (ERP/감정)',
                'params': {'l_freq': 0.1, 'h_freq': 30},
                'ref': 'Luck (2014), Widmann et al. (2015)'
            },
            'Notch': {
                'desc': '전원 노이즈 제거',
                'params': {'freqs': [50, 60]},
                'ref': 'Delorme & Makeig (2004)'
            },
            'High-pass': {
                'desc': '드리프트 제거',
                'params': {'l_freq': 0.1, 'h_freq': None},
                'ref': 'Tanner et al. (2015)'
            },
            'FIR': {
                'desc': '선형 위상 유지',
                'params': {'l_freq': 0.1, 'h_freq': 30, 'filter_type': 'fir'},
                'ref': 'Widmann et al. (2015)'
            },
            'Zero-phase': {
                'desc': '위상 왜곡 방지',
                'params': {'l_freq': 0.1, 'h_freq': 30, 'method': 'fir', 'phase': 'zero'},
                'ref': 'Hu et al. (2010)'
            }
        }
        self.filter_type_combo.addItems(list(self.filter_types.keys()))
        filter_layout.addWidget(QLabel('필터 종류:'))
        filter_layout.addWidget(self.filter_type_combo)
        
        # 필터 설명 레이블
        self.filter_desc_label = QLabel()
        self.filter_desc_label.setWordWrap(True)
        filter_layout.addWidget(self.filter_desc_label)
        
        # 필터 파라미터 설정 위젯
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        
        # Band-pass 파라미터
        self.bandpass_widget = QWidget()
        bandpass_layout = QVBoxLayout(self.bandpass_widget)
        
        self.low_freq = QDoubleSpinBox()
        self.low_freq.setRange(0.1, 100)
        self.low_freq.setValue(0.1)
        self.low_freq.setSingleStep(0.1)
        bandpass_layout.addWidget(QLabel('저주파 (Hz):'))
        bandpass_layout.addWidget(self.low_freq)
        
        self.high_freq = QDoubleSpinBox()
        self.high_freq.setRange(0.1, 100)
        self.high_freq.setValue(30)
        self.high_freq.setSingleStep(0.1)
        bandpass_layout.addWidget(QLabel('고주파 (Hz):'))
        bandpass_layout.addWidget(self.high_freq)
        
        self.params_layout.addWidget(self.bandpass_widget)
        
        # Notch 파라미터
        self.notch_widget = QWidget()
        notch_layout = QVBoxLayout(self.notch_widget)
        
        self.notch_freq = QComboBox()
        self.notch_freq.addItems(['50 Hz', '60 Hz', '50/60 Hz'])
        notch_layout.addWidget(QLabel('노치 주파수:'))
        notch_layout.addWidget(self.notch_freq)
        
        self.params_layout.addWidget(self.notch_widget)
        self.notch_widget.hide()
        
        filter_layout.addWidget(self.params_widget)
        
        # 필터 적용 버튼
        self.apply_filter_btn = QPushButton('필터 적용')
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        filter_layout.addWidget(self.apply_filter_btn)
        
        # 필터 타입 변경 시 UI 업데이트
        self.filter_type_combo.currentTextChanged.connect(self.update_filter_ui)
        self.update_filter_ui(self.filter_type_combo.currentText())
        
        control_layout.addWidget(filter_group)
        
        # ICA 설정
        ica_group = QWidget()
        ica_layout = QVBoxLayout(ica_group)
        ica_layout.addWidget(QLabel('ICA 설정'))
        
        self.n_components = QSpinBox()
        self.n_components.setRange(1, 100)
        self.n_components.setValue(4)
        ica_layout.addWidget(QLabel('컴포넌트 수:'))
        ica_layout.addWidget(self.n_components)
        
        self.apply_ica_btn = QPushButton('ICA 적용')
        self.apply_ica_btn.clicked.connect(self.apply_ica)
        ica_layout.addWidget(self.apply_ica_btn)
        
        control_layout.addWidget(ica_group)
        
        # 에포킹 설정
        epoch_group = QWidget()
        epoch_layout = QVBoxLayout(epoch_group)
        epoch_layout.addWidget(QLabel('에포킹 설정'))
        
        self.tmin = QDoubleSpinBox()
        self.tmin.setRange(-10, 0)
        self.tmin.setValue(-0.2)
        self.tmin.setSingleStep(0.1)
        epoch_layout.addWidget(QLabel('시작 시간 (초):'))
        epoch_layout.addWidget(self.tmin)
        
        self.tmax = QDoubleSpinBox()
        self.tmax.setRange(0, 10)
        self.tmax.setValue(0.5)
        self.tmax.setSingleStep(0.1)
        epoch_layout.addWidget(QLabel('종료 시간 (초):'))
        epoch_layout.addWidget(self.tmax)
        
        self.apply_epoch_btn = QPushButton('에포킹 적용')
        self.apply_epoch_btn.clicked.connect(self.apply_epoching)
        epoch_layout.addWidget(self.apply_epoch_btn)
        
        control_layout.addWidget(epoch_group)
        
        # ERP 분석 버튼
        self.analyze_erp_btn = QPushButton('ERP 분석')
        self.analyze_erp_btn.clicked.connect(self.analyze_erp)
        control_layout.addWidget(self.analyze_erp_btn)
        
        control_layout.addStretch()
        layout.addWidget(control_panel)
        
        # 그래프 영역 (2개의 서브플롯)
        graph_panel = QWidget()
        graph_layout = QVBoxLayout(graph_panel)
        
        # EEG 데이터 그래프
        self.eeg_figure = Figure(figsize=(8, 4))
        self.eeg_canvas = FigureCanvas(self.eeg_figure)
        graph_layout.addWidget(self.eeg_canvas)
        
        # 이벤트 데이터 그래프
        self.event_figure = Figure(figsize=(8, 2))
        self.event_canvas = FigureCanvas(self.event_figure)
        graph_layout.addWidget(self.event_canvas)
        
        layout.addWidget(graph_panel)
        
    def load_eeg_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'EEG 데이터 파일 선택', '', 'CSV Files (*.csv)')
        if file_name:
            try:
                data = pd.read_csv(file_name)
                ch_names = data.columns.tolist()
                sfreq = 256  # 샘플링 주파수
                
                # MNE Raw 객체 생성
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg']*len(ch_names))
                self.raw_data = mne.io.RawArray(data.T.values, info)
                
                # 데이터 시각화
                self.plot_raw_data()
                QMessageBox.information(self, '성공', 'EEG 데이터가 성공적으로 로드되었습니다.')
            except Exception as e:
                QMessageBox.critical(self, '오류', f'데이터 로드 중 오류 발생: {str(e)}')
    
    def load_events(self):
        file_name, _ = QFileDialog.getOpenFileName(self, '이벤트 데이터 파일 선택', '', 'CSV Files (*.csv)')
        if file_name:
            try:
                events_data = pd.read_csv(file_name)
                self.events = events_data[['sample', 'previous', 'event_id']].values
                self.plot_events()  # 이벤트 시각화 추가
                QMessageBox.information(self, '성공', '이벤트 데이터가 성공적으로 로드되었습니다.')
            except Exception as e:
                QMessageBox.critical(self, '오류', f'이벤트 데이터 로드 중 오류 발생: {str(e)}')
    
    def update_filter_ui(self, filter_type):
        # 필터 설명 업데이트
        filter_info = self.filter_types[filter_type]
        desc = (f"목적: {filter_info['desc']}\n"
               f"참고문헌: {filter_info['ref']}")
        self.filter_desc_label.setText(desc)
        
        # 파라미터 위젯 업데이트
        self.bandpass_widget.hide()
        self.notch_widget.hide()
        
        if filter_type in ['Band-pass', 'High-pass', 'FIR', 'Zero-phase']:
            self.bandpass_widget.show()
        elif filter_type == 'Notch':
            self.notch_widget.show()
    
    def apply_filter(self):
        if self.raw_data is None:
            QMessageBox.warning(self, '경고', '먼저 EEG 데이터를 로드해주세요.')
            return
        
        try:
            filter_type = self.filter_type_combo.currentText()
            
            if filter_type == 'Band-pass':
                self.raw_data.filter(
                    l_freq=self.low_freq.value(),
                    h_freq=self.high_freq.value(),
                    method='fir'
                )
            
            elif filter_type == 'Notch':
                freqs = {
                    '50 Hz': [50],
                    '60 Hz': [60],
                    '50/60 Hz': [50, 60]
                }[self.notch_freq.currentText()]
                
                for freq in freqs:
                    self.raw_data.notch_filter(
                        freqs=freq,
                        picks='eeg',
                        method='fir'
                    )
            
            elif filter_type == 'High-pass':
                self.raw_data.filter(
                    l_freq=self.low_freq.value(),
                    h_freq=None,
                    method='fir'
                )
            
            elif filter_type == 'FIR':
                self.raw_data.filter(
                    l_freq=self.low_freq.value(),
                    h_freq=self.high_freq.value(),
                    method='fir',
                    phase='zero'
                )
            
            elif filter_type == 'Zero-phase':
                self.raw_data.filter(
                    l_freq=self.low_freq.value(),
                    h_freq=self.high_freq.value(),
                    method='fir',
                    phase='zero',
                    fir_window='hamming'
                )
            
            self.plot_raw_data()
            
            msg = f'{filter_type} 필터가 성공적으로 적용되었습니다.'
            if filter_type in ['Band-pass', 'High-pass', 'FIR', 'Zero-phase']:
                msg += f'\n주파수 대역: {self.low_freq.value()}-{self.high_freq.value() if self.high_freq.value() else "∞"} Hz'
            elif filter_type == 'Notch':
                msg += f'\n제거 주파수: {self.notch_freq.currentText()}'
            
            QMessageBox.information(self, '성공', msg)
            
        except Exception as e:
            QMessageBox.critical(self, '오류', f'필터 적용 중 오류 발생: {str(e)}')
    
    def apply_ica(self):
        if self.raw_data is None:
            QMessageBox.warning(self, '경고', '먼저 EEG 데이터를 로드해주세요.')
            return
        
        try:
            # ICA 적용
            ica = mne.preprocessing.ICA(n_components=self.n_components.value())
            ica.fit(self.raw_data)
            self.raw_data = ica.apply(self.raw_data)
            self.plot_raw_data()
            QMessageBox.information(self, '성공', 'ICA가 성공적으로 적용되었습니다.')
        except Exception as e:
            QMessageBox.critical(self, '오류', f'ICA 적용 중 오류 발생: {str(e)}')
    
    def apply_epoching(self):
        if self.raw_data is None or self.events is None:
            QMessageBox.warning(self, '경고', 'EEG 데이터와 이벤트 데이터를 모두 로드해주세요.')
            return
        
        try:
            # 에포킹 적용
            self.epochs = mne.Epochs(self.raw_data, self.events, 
                                   tmin=self.tmin.value(), tmax=self.tmax.value(),
                                   baseline=(None, 0), preload=True)
            self.plot_epochs()
            QMessageBox.information(self, '성공', '에포킹이 성공적으로 적용되었습니다.')
        except Exception as e:
            QMessageBox.critical(self, '오류', f'에포킹 적용 중 오류 발생: {str(e)}')
    
    def analyze_erp(self):
        if self.epochs is None:
            QMessageBox.warning(self, '경고', '먼저 에포킹을 적용해주세요.')
            return
        
        try:
            # ERP 분석 및 시각화
            self.plot_erp()
            QMessageBox.information(self, '성공', 'ERP 분석이 완료되었습니다.')
        except Exception as e:
            QMessageBox.critical(self, '오류', f'ERP 분석 중 오류 발생: {str(e)}')
    
    def plot_raw_data(self):
        self.eeg_figure.clear()
        ax = self.eeg_figure.add_subplot(111)
        
        # 데이터와 시간 벡터 가져오기
        data = self.raw_data.get_data()
        times = np.arange(data.shape[1]) / self.raw_data.info['sfreq']
        
        # 각 채널 데이터 플로팅
        for i, ch_name in enumerate(self.raw_data.ch_names):
            offset = i * np.std(data) * 3
            ax.plot(times, data[i] + offset, label=ch_name)
        
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('진폭')
        ax.set_title('EEG 데이터')
        ax.legend()
        
        # 이벤트가 있다면 이벤트 시점 표시
        if self.events is not None:
            event_times = self.events[:, 0] / self.raw_data.info['sfreq']
            for t in event_times:
                ax.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        
        self.eeg_canvas.draw()
        
        # 이벤트 데이터가 있다면 이벤트도 다시 그리기
        if self.events is not None:
            self.plot_events()

    def plot_events(self):
        if self.events is None:
            return
            
        self.event_figure.clear()
        ax = self.event_figure.add_subplot(111)
        
        # 이벤트 타임라인 생성
        if self.raw_data is not None:
            times = self.events[:, 0] / self.raw_data.info['sfreq']
        else:
            times = self.events[:, 0] / 256  # 기본 샘플링 레이트 사용
            
        event_ids = self.events[:, 2]
        unique_events = np.unique(event_ids)
        
        # 이벤트 타입별로 다른 색상으로 표시
        for event_id in unique_events:
            mask = event_ids == event_id
            ax.scatter(times[mask], [1]*sum(mask), 
                      label=f'이벤트 {event_id}', 
                      marker='v', s=100)
        
        ax.set_xlabel('시간 (초)')
        ax.set_title('이벤트 타이밍')
        ax.set_yticks([])  # y축 눈금 제거
        ax.legend()
        ax.grid(True)
        
        # x축 범위를 EEG 데이터와 맞추기
        if self.raw_data is not None:
            ax.set_xlim(0, len(self.raw_data) / self.raw_data.info['sfreq'])
        
        self.event_canvas.draw()
    
    def plot_epochs(self):
        self.eeg_figure.clear()
        ax = self.eeg_figure.add_subplot(111)
        
        # 에포크 데이터 가져오기
        data = self.epochs.get_data()
        times = self.epochs.times
        
        # 각 조건별 평균 계산
        unique_events = np.unique(self.epochs.events[:, 2])
        
        for event_id in unique_events:
            # 해당 이벤트의 에포크 선택
            event_mask = self.epochs.events[:, 2] == event_id
            event_data = data[event_mask].mean(axis=0)  # 조건별 평균
            
            # 각 채널 데이터 플로팅
            for i, ch_name in enumerate(self.epochs.ch_names):
                offset = i * np.std(event_data) * 3
                ax.plot(times, event_data[i] + offset, 
                       label=f'{ch_name} (Event {event_id})')
        
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('진폭')
        ax.set_title('에포크 데이터')
        ax.legend()
        ax.grid(True)
        
        self.eeg_canvas.draw()
        
        # 이벤트 타이밍 표시
        self.event_figure.clear()
        ax = self.event_figure.add_subplot(111)
        ax.set_title('에포크 기준점')
        ax.axvline(x=0, color='r', linestyle='--', label='기준점')
        ax.set_xlabel('시간 (초)')
        ax.set_yticks([])
        ax.legend()
        ax.grid(True)
        ax.set_xlim(times[0], times[-1])
        self.event_canvas.draw()
    
    def plot_erp(self):
        self.eeg_figure.clear()
        ax = self.eeg_figure.add_subplot(111)
        
        # ERP 데이터 계산 (모든 에포크의 평균)
        evoked = self.epochs.average()
        data = evoked.data
        times = evoked.times
        
        # 각 채널의 ERP 플로팅
        for i, ch_name in enumerate(evoked.ch_names):
            offset = i * np.std(data) * 3
            ax.plot(times, data[i] + offset, label=ch_name)
        
        ax.set_xlabel('시간 (초)')
        ax.set_ylabel('진폭')
        ax.set_title('사건관련전위 (ERP)')
        ax.legend()
        ax.grid(True)
        
        # 기준선(0초) 표시
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        self.eeg_canvas.draw()
        
        # 이벤트 타이밍 표시
        self.event_figure.clear()
        ax = self.event_figure.add_subplot(111)
        ax.set_title('ERP 기준점')
        ax.axvline(x=0, color='r', linestyle='--', label='자극 제시')
        ax.set_xlabel('시간 (초)')
        ax.set_yticks([])
        ax.legend()
        ax.grid(True)
        ax.set_xlim(times[0], times[-1])
        self.event_canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EEGAnalysisGUI()
    window.show()
    sys.exit(app.exec_()) 
# eeg_analyzer_gui.py

import sys
import os
import numpy as np
import pandas as pd
import mne
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QComboBox, QSpinBox, QTextEdit, QMessageBox,
                            QDoubleSpinBox, QGroupBox, QCheckBox, QTableWidget,
                            QTableWidgetItem, QHeaderView, QStackedWidget, QAbstractScrollArea,
                            QListWidget, QLineEdit, QListWidgetItem, QRadioButton, QButtonGroup,
                            QScrollArea)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from functools import partial  # 꼭 추가하세요
from PyQt5.QtCore import QTimer



class EEGAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.raw = None
        self.ica = None
        self.epochs = None
        self.events = None
        self.event_id = {}
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('EEG 데이터 분석기')
        self.setGeometry(100, 100, 1600, 900)
        
        # 메인 위젯과 레이아웃 설정
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # 1. 좌측 메뉴 패널
        menu_panel = QWidget()
        menu_layout = QVBoxLayout()
        menu_panel.setLayout(menu_layout)
        menu_panel.setFixedWidth(150)
        self.menu_buttons = []
        menu_names = ['데이터 로드', '채널 품질 검사', '필터링', '참조 설정', 'ICA', 'ERP 추출']
        for idx, name in enumerate(menu_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self.switch_panel(i))
            menu_layout.addWidget(btn)
            self.menu_buttons.append(btn)
        menu_layout.addStretch(1)
        self.menu_buttons[0].setChecked(True)

        # 2. 중앙 기능 패널 (QStackedWidget)
        self.stacked_widget = QStackedWidget()
        # 각 기능별 위젯 생성
        self.data_load_widget = self.create_data_load_widget()
        self.quality_widget = self.create_quality_widget()
        self.filter_widget = self.create_filter_widget()
        self.reference_widget = self.create_reference_widget()
        self.ica_widget = self.create_ica_widget()
        self.erp_widget = self.create_erp_widget()
        self.stacked_widget.addWidget(self.data_load_widget)
        self.stacked_widget.addWidget(self.quality_widget)
        self.stacked_widget.addWidget(self.filter_widget)
        self.stacked_widget.addWidget(self.reference_widget)
        self.stacked_widget.addWidget(self.ica_widget)
        self.stacked_widget.addWidget(self.erp_widget)

        # 3. 우측 결과 그래프 + DataFrame 패널
        plot_panel = QWidget()
        plot_panel_layout = QVBoxLayout()
        plot_panel.setLayout(plot_panel_layout)
        # 데이터 타입 선택 버튼 추가
        self.data_type_group = QButtonGroup(plot_panel)
        self.raw_radio = QRadioButton('Raw 데이터')
        self.mne_radio = QRadioButton('MNE 데이터')
        self.raw_radio.setChecked(True)
        self.mne_radio.setEnabled(False)
        self.data_type_group.addButton(self.raw_radio)
        self.data_type_group.addButton(self.mne_radio)
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.raw_radio)
        radio_layout.addWidget(self.mne_radio)
        plot_panel_layout.addLayout(radio_layout)
        self.raw_radio.toggled.connect(self.update_plot_by_radio)
        self.mne_radio.toggled.connect(self.update_plot_by_radio)
        # 스크롤 영역 및 레이아웃
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.plot_layout = QVBoxLayout(self.scroll_content)
        self.scroll.setWidget(self.scroll_content)
        plot_panel_layout.addWidget(self.scroll, stretch=1)
        # DataFrame 테이블 (결과용)
        self.result_df_table = QTableWidget()
        self.result_df_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.result_df_table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.result_df_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_df_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_df_table.setSelectionMode(QTableWidget.SingleSelection)
        self.result_df_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_df_table.setMinimumHeight(100)
        self.result_df_table.setMaximumHeight(self.height() // 3)
        plot_panel_layout.addWidget(self.result_df_table, stretch=0)
        # 플롯 컨트롤
        plot_control_layout = QHBoxLayout()
        self.plot_duration_spin = QDoubleSpinBox()
        self.plot_duration_spin.setRange(1, 60)
        self.plot_duration_spin.setValue(5)
        plot_control_layout.addWidget(QLabel('표시 구간 (초):'))
        plot_control_layout.addWidget(self.plot_duration_spin)
        self.update_plot_btn = QPushButton('플롯 업데이트')
        self.update_plot_btn.clicked.connect(self.plot_data)
        plot_control_layout.addWidget(self.update_plot_btn)
        plot_panel_layout.addLayout(plot_control_layout)

        # 전체 레이아웃 배치
        main_layout.addWidget(menu_panel)
        main_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(plot_panel)

        # 초기 패널 표시
        self.switch_panel(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 윈도우 크기 변경 시 DataFrame 테이블 최대 높이 갱신
        if hasattr(self, 'result_df_table'):
            self.result_df_table.setMaximumHeight(self.height() // 3)

    def switch_panel(self, idx):
        for i, btn in enumerate(self.menu_buttons):
            btn.setChecked(i == idx)
        self.stacked_widget.setCurrentIndex(idx)
        # 채널 품질 검사 패널로 전환 시 bad_channel_list를 채널명+bad 상태로 초기화
        if hasattr(self, 'bad_channel_list') and idx == 1:
            self.bad_channel_list.clear()
            ch_names = []
            bads = []
            if hasattr(self, 'raw') and self.raw is not None:
                ch_names = self.raw.ch_names
                bads = self.raw.info['bads']
            elif hasattr(self, 'df') and self.df is not None:
                ch_names = list(self.df.columns)
            for ch in ch_names:
                item = QListWidgetItem(ch)
                if ch in bads:
                    item.setCheckState(2)
                else:
                    item.setCheckState(0)
                self.bad_channel_list.addItem(item)

    def create_data_load_widget(self):
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        # 좌측: 데이터 로드 컨트롤
        control_layout = QVBoxLayout()
        # 파일 로드 버튼
        self.load_btn = QPushButton('CSV 파일 로드')
        self.load_btn.clicked.connect(self.load_csv)
        control_layout.addWidget(self.load_btn)
        # 채널 설정
        control_layout.addWidget(QLabel('채널 이름:'))
        self.channel_text = QTextEdit()
        self.channel_text.setPlaceholderText('채널 이름을 쉼표로 구분하여 입력\n예: F3,F4,C3,C4')
        control_layout.addWidget(self.channel_text)
        # 샘플링 주파수 설정
        sfreq_layout = QHBoxLayout()
        sfreq_layout.addWidget(QLabel('샘플링 주파수 (Hz):'))
        self.sfreq_spin = QSpinBox()
        self.sfreq_spin.setRange(1, 1000)
        self.sfreq_spin.setValue(256)
        sfreq_layout.addWidget(self.sfreq_spin)
        control_layout.addLayout(sfreq_layout)
        # 데이터 변환 버튼
        self.convert_btn = QPushButton('MNE Raw 객체로 변환')
        self.convert_btn.clicked.connect(self.convert_to_raw)
        self.convert_btn.setEnabled(False)
        control_layout.addWidget(self.convert_btn)
        # 데이터 정보 표시
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        control_layout.addWidget(self.info_text)
        control_layout.addStretch(1)
        layout.addLayout(control_layout)
        return widget

    def create_quality_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        # 자동 탐지 옵션
        auto_layout = QHBoxLayout()
        auto_layout.addWidget(QLabel('자동 탐지 기준:'))
        self.auto_method_combo = QComboBox()
        self.auto_method_combo.addItems(['평균 제곱', '편평성', '이상치'])
        auto_layout.addWidget(self.auto_method_combo)
        self.auto_detect_btn = QPushButton('자동 탐지')
        self.auto_detect_btn.clicked.connect(self.auto_detect_bad_channels)
        auto_layout.addWidget(self.auto_detect_btn)
        layout.addLayout(auto_layout)
        # bad 채널 리스트
        layout.addWidget(QLabel('불량 채널 목록:'))
        self.bad_channel_list = QListWidget()
        self.bad_channel_list.itemChanged.connect(self.on_bad_channel_list_changed)
        layout.addWidget(self.bad_channel_list)
        # bad 채널 수동 추가/삭제
        manual_layout = QHBoxLayout()
        self.manual_bad_input = QLineEdit()
        self.manual_bad_input.setPlaceholderText('채널명 입력')
        manual_layout.addWidget(self.manual_bad_input)
        self.add_bad_btn = QPushButton('추가')
        self.add_bad_btn.clicked.connect(self.add_bad_channel)
        manual_layout.addWidget(self.add_bad_btn)
        self.remove_bad_btn = QPushButton('삭제')
        self.remove_bad_btn.clicked.connect(self.remove_bad_channel)
        manual_layout.addWidget(self.remove_bad_btn)
        layout.addLayout(manual_layout)
        # 보간 방법 선택 콤보박스
        interp_method_layout = QHBoxLayout()
        interp_method_layout.addWidget(QLabel('보간 방법:'))
        self.interpolate_method_combo = QComboBox()
        self.interpolate_method_combo.addItems(['spline', 'nearest', 'linear'])
        self.interpolate_method_combo.setCurrentText('spline')
        interp_method_layout.addWidget(self.interpolate_method_combo)
        layout.addLayout(interp_method_layout)
        self.interpolate_btn = QPushButton('보간 실행')
        self.interpolate_btn.clicked.connect(self.run_interpolate)
        layout.addWidget(self.interpolate_btn)
        layout.addStretch(1)
        return widget

    def create_filter_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        # 대역 통과 필터 설정
        bandpass_layout = QHBoxLayout()
        bandpass_layout.addWidget(QLabel('대역 통과 필터 (Hz):'))
        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(0.1, 100.0)
        self.low_freq_spin.setValue(1.0)
        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(0.1, 100.0)
        self.high_freq_spin.setValue(40.0)
        bandpass_layout.addWidget(self.low_freq_spin)
        bandpass_layout.addWidget(QLabel('-'))
        bandpass_layout.addWidget(self.high_freq_spin)
        layout.addLayout(bandpass_layout)
        # 노치 필터 설정
        notch_layout = QHBoxLayout()
        self.notch_check = QCheckBox('노치 필터 적용')
        self.notch_freq_spin = QDoubleSpinBox()
        self.notch_freq_spin.setRange(50.0, 60.0)
        self.notch_freq_spin.setValue(60.0)
        notch_layout.addWidget(self.notch_check)
        notch_layout.addWidget(self.notch_freq_spin)
        layout.addLayout(notch_layout)
        # 필터 적용 버튼
        self.apply_filter_btn = QPushButton('필터 적용')
        self.apply_filter_btn.clicked.connect(self.apply_filters)
        self.apply_filter_btn.setEnabled(False)
        layout.addWidget(self.apply_filter_btn)
        layout.addStretch(1)
        return widget

    def create_reference_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        # 참조 방식 선택
        self.ref_group = QButtonGroup(widget)
        self.radio_avg = QRadioButton('공통 평균 참조 (average reference)')
        self.radio_avg.setChecked(True)
        self.ref_group.addButton(self.radio_avg)
        layout.addWidget(self.radio_avg)
        self.radio_custom = QRadioButton('특정 채널 참조')
        self.ref_group.addButton(self.radio_custom)
        layout.addWidget(self.radio_custom)
        # 특정 채널 입력
        self.ref_channel_input = QLineEdit()
        self.ref_channel_input.setPlaceholderText('예: TP9,TP10,Cz')
        layout.addWidget(self.ref_channel_input)
        # 참조 적용 버튼
        self.apply_ref_btn = QPushButton('참조 설정 적용')
        self.apply_ref_btn.clicked.connect(self.apply_reference)
        layout.addWidget(self.apply_ref_btn)
        # ERP 파형 비교 버튼
        self.compare_erp_btn = QPushButton('ERP 파형 비교')
        self.compare_erp_btn.clicked.connect(self.compare_erp)
        layout.addWidget(self.compare_erp_btn)
        layout.addStretch(1)
        return widget

    def create_ica_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        # ICA 컴포넌트 수 설정
        ica_comp_layout = QHBoxLayout()
        ica_comp_layout.addWidget(QLabel('ICA 컴포넌트 수:'))
        self.ica_n_components = QSpinBox()
        self.ica_n_components.setRange(5, 50)
        self.ica_n_components.setValue(20)
        ica_comp_layout.addWidget(self.ica_n_components)
        layout.addLayout(ica_comp_layout)
        # ICA 버튼들
        self.fit_ica_btn = QPushButton('ICA 적용')
        self.fit_ica_btn.clicked.connect(self.fit_ica)
        self.fit_ica_btn.setEnabled(False)
        layout.addWidget(self.fit_ica_btn)
        self.plot_components_btn = QPushButton('ICA 컴포넌트 시각화')
        self.plot_components_btn.clicked.connect(self.plot_ica_components)
        self.plot_components_btn.setEnabled(False)
        layout.addWidget(self.plot_components_btn)
        # ICA 컴포넌트 제거 설정
        ica_exclude_layout = QHBoxLayout()
        ica_exclude_layout.addWidget(QLabel('제거할 컴포넌트:'))
        self.ica_exclude_text = QTextEdit()
        self.ica_exclude_text.setPlaceholderText('제거할 컴포넌트 번호를 쉼표로 구분\n예: 0,1,2')
        self.ica_exclude_text.setMaximumHeight(50)
        ica_exclude_layout.addWidget(self.ica_exclude_text)
        layout.addLayout(ica_exclude_layout)
        self.apply_ica_btn = QPushButton('선택된 컴포넌트 제거')
        self.apply_ica_btn.clicked.connect(self.apply_ica)
        self.apply_ica_btn.setEnabled(False)
        layout.addWidget(self.apply_ica_btn)
        layout.addStretch(1)
        return widget

    def create_erp_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        # 이벤트 파일 로드
        event_layout = QHBoxLayout()
        self.load_events_btn = QPushButton('이벤트 파일 로드')
        self.load_events_btn.clicked.connect(self.load_events)
        self.load_events_btn.setEnabled(False)
        event_layout.addWidget(self.load_events_btn)
        layout.addLayout(event_layout)
        # 이벤트 테이블
        self.event_table = QTableWidget()
        self.event_table.setColumnCount(3)
        self.event_table.setHorizontalHeaderLabels(['샘플', '이전 값', '이벤트 ID'])
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.event_table)
        # 에포크 설정
        epoch_settings_layout = QVBoxLayout()
        # 시간 범위 설정
        time_range_layout = QHBoxLayout()
        time_range_layout.addWidget(QLabel('시간 범위 (초):'))
        self.tmin_spin = QDoubleSpinBox()
        self.tmin_spin.setRange(-2.0, 0.0)
        self.tmin_spin.setValue(-0.2)
        self.tmin_spin.setSingleStep(0.1)
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(0.0, 2.0)
        self.tmax_spin.setValue(0.8)
        self.tmax_spin.setSingleStep(0.1)
        time_range_layout.addWidget(self.tmin_spin)
        time_range_layout.addWidget(QLabel('-'))
        time_range_layout.addWidget(self.tmax_spin)
        epoch_settings_layout.addLayout(time_range_layout)
        # Baseline 설정
        baseline_layout = QHBoxLayout()
        baseline_layout.addWidget(QLabel('Baseline (초):'))
        self.baseline_start_spin = QDoubleSpinBox()
        self.baseline_start_spin.setRange(-2.0, 0.0)
        self.baseline_start_spin.setValue(-0.2)
        self.baseline_start_spin.setSingleStep(0.1)
        self.baseline_end_spin = QDoubleSpinBox()
        self.baseline_end_spin.setRange(-2.0, 0.0)
        self.baseline_end_spin.setValue(0.0)
        self.baseline_end_spin.setSingleStep(0.1)
        baseline_layout.addWidget(self.baseline_start_spin)
        baseline_layout.addWidget(QLabel('-'))
        baseline_layout.addWidget(self.baseline_end_spin)
        epoch_settings_layout.addLayout(baseline_layout)
        layout.addLayout(epoch_settings_layout)
        # ERP 추출 버튼
        self.extract_erp_btn = QPushButton('ERP 추출')
        self.extract_erp_btn.clicked.connect(self.extract_erp)
        self.extract_erp_btn.setEnabled(False)
        layout.addWidget(self.extract_erp_btn)
        # ERP 플롯 버튼
        self.plot_erp_btn = QPushButton('ERP 플롯')
        self.plot_erp_btn.clicked.connect(self.plot_erp)
        self.plot_erp_btn.setEnabled(False)
        layout.addWidget(self.plot_erp_btn)
        layout.addStretch(1)
        return widget

    def load_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'CSV 파일 선택', '', 'CSV files (*.csv)')
        if file_name:
            try:
                self.df = pd.read_csv(file_name)
                self.info_text.append(f'파일 로드 완료: {file_name}')
                self.info_text.append(f'데이터 형태: {self.df.shape}')
                # 채널 이름을 자동으로 채널 입력란에 표시
                channel_names = ','.join(self.df.columns)
                self.channel_text.setText(channel_names)
                self.convert_btn.setEnabled(True)
                # 결과창에 DataFrame과 그래프 표시
                self.show_result_df_in_table(self.df)
                self.plot_data_preview()
            except Exception as e:
                QMessageBox.critical(self, '오류', f'파일 로드 중 오류 발생: {str(e)}')

    def convert_to_raw(self):
        try:
            ch_names = list(self.df.columns)
            if not ch_names:
                raise ValueError("CSV 파일에 채널 이름이 없습니다.")
            data = self.df.values.T
            if len(ch_names) != data.shape[0]:
                raise ValueError(f"채널 수({len(ch_names)})가 데이터 채널 수({data.shape[0]})와 일치하지 않습니다.")
            info = mne.create_info(ch_names=ch_names, 
                                 sfreq=self.sfreq_spin.value(), 
                                 ch_types='eeg')
            self.raw = mne.io.RawArray(data, info)
            # 표준 10-20 몽타주 적용 (채널명이 표준과 맞는 경우)
            montage = mne.channels.make_standard_montage('standard_1020')
            self.raw.set_montage(montage, match_case=False)
            self.info_text.append('\nMNE Raw 객체 생성 완료')
            self.info_text.append(f'채널 수: {len(self.raw.ch_names)}')
            self.info_text.append(f'데이터 길이: {self.raw.times[-1]:.2f}초')
            self.apply_filter_btn.setEnabled(True)
            self.fit_ica_btn.setEnabled(True)
            self.load_events_btn.setEnabled(True)
            # MNE 변환 후 버튼 활성화
            self.mne_radio.setEnabled(True)
            # 데이터 플롯
            self.update_plot_by_radio()
            self.show_result_df_in_table(self.df)
        except Exception as e:
            QMessageBox.critical(self, '오류', f'변환 중 오류 발생: {str(e)}')
    
    def apply_filters(self):
        if self.raw is None:
            return
            
        try:
            # 대역 통과 필터 적용
            self.raw.filter(
                l_freq=self.low_freq_spin.value(),
                h_freq=self.high_freq_spin.value(),
                picks='eeg'
            )
            
            # 노치 필터 적용
            if self.notch_check.isChecked():
                self.raw.notch_filter(
                    freqs=self.notch_freq_spin.value(),
                    picks='eeg'
                )
            
            self.info_text.append('\n필터 적용 완료')
            self.info_text.append(f'대역 통과: {self.low_freq_spin.value()}-{self.high_freq_spin.value()} Hz')
            if self.notch_check.isChecked():
                self.info_text.append(f'노치 필터: {self.notch_freq_spin.value()} Hz')
            
            # 데이터 플롯 업데이트
            self.update_plot_by_radio()
            
        except Exception as e:
            QMessageBox.critical(self, '오류', f'필터 적용 중 오류 발생: {str(e)}')
    
    def fit_ica(self):
        if self.raw is None:
            return
            
        try:
            # ICA 객체 생성 및 피팅
            self.ica = mne.preprocessing.ICA(
                n_components=self.ica_n_components.value(),
                random_state=42
            )
            
            self.ica.fit(self.raw)
            
            self.info_text.append('\nICA 피팅 완료')
            self.info_text.append(f'컴포넌트 수: {self.ica_n_components.value()}')
            
            # ICA 관련 버튼 활성화
            self.plot_components_btn.setEnabled(True)
            self.apply_ica_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, '오류', f'ICA 피팅 중 오류 발생: {str(e)}')
    
    def plot_ica_components(self):
        if self.ica is None:
            return
            
        try:
            self.ica.plot_components()
        except Exception as e:
            QMessageBox.critical(self, '오류', f'ICA 컴포넌트 플롯 중 오류 발생: {str(e)}')
    
    def apply_ica(self):
        if self.ica is None or self.raw is None:
            return
            
        try:
            # 제거할 컴포넌트 파싱
            exclude_str = self.ica_exclude_text.toPlainText().strip()
            if not exclude_str:
                QMessageBox.warning(self, '경고', '제거할 컴포넌트를 선택해주세요.')
                return
                
            exclude_components = [int(x.strip()) for x in exclude_str.split(',')]
            
            # ICA 적용
            self.ica.exclude = exclude_components
            self.raw = self.ica.apply(self.raw.copy())
            
            self.info_text.append('\nICA 컴포넌트 제거 완료')
            self.info_text.append(f'제거된 컴포넌트: {exclude_components}')
            
            # 데이터 플롯 업데이트
            self.update_plot_by_radio()
            
        except Exception as e:
            QMessageBox.critical(self, '오류', f'ICA 적용 중 오류 발생: {str(e)}')
    
    def load_events(self):
        if self.raw is None:
            return
            
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, '이벤트 파일 선택', '', 'CSV files (*.csv)')
            
            if file_name:
                # 이벤트 데이터 로드
                events_df = pd.read_csv(file_name)
                if len(events_df.columns) != 3:
                    raise ValueError("이벤트 파일은 3개의 열(샘플, 이전 값, 이벤트 ID)을 가져야 합니다.")
                
                self.events = events_df.values
                
                # 이벤트 테이블 업데이트
                self.event_table.setRowCount(len(self.events))
                for i, event in enumerate(self.events):
                    for j, value in enumerate(event):
                        self.event_table.setItem(i, j, QTableWidgetItem(str(value)))
                
                # 고유한 이벤트 ID 수집
                unique_ids = np.unique(self.events[:, 2])
                self.event_id = {f'Event_{id}': int(id) for id in unique_ids}
                
                self.info_text.append('\n이벤트 로드 완료')
                self.info_text.append(f'이벤트 수: {len(self.events)}')
                self.info_text.append(f'이벤트 ID: {self.event_id}')
                
                # ERP 추출 버튼 활성화
                self.extract_erp_btn.setEnabled(True)
                
        except Exception as e:
            QMessageBox.critical(self, '오류', f'이벤트 로드 중 오류 발생: {str(e)}')
    
    def extract_erp(self):
        if self.raw is None or self.events is None:
            return
            
        try:
            # 에포크 생성
            self.epochs = mne.Epochs(
                self.raw,
                self.events,
                event_id=self.event_id,
                tmin=self.tmin_spin.value(),
                tmax=self.tmax_spin.value(),
                baseline=(self.baseline_start_spin.value(), self.baseline_end_spin.value()),
                preload=True
            )
            
            self.info_text.append('\nERP 추출 완료')
            self.info_text.append(f'에포크 수: {len(self.epochs)}')
            self.info_text.append(f'시간 범위: {self.tmin_spin.value()}-{self.tmax_spin.value()}초')
            self.info_text.append(f'Baseline: {self.baseline_start_spin.value()}-{self.baseline_end_spin.value()}초')
            
            # ERP 플롯 버튼 활성화
            self.plot_erp_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, '오류', f'ERP 추출 중 오류 발생: {str(e)}')
    
    def plot_erp(self):
        if self.epochs is None:
            return
            
        try:
            self.clear_plot_layout()
            erp = self.epochs.average()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            for i, ch_name in enumerate(erp.ch_names):
                row_layout = QHBoxLayout()
                label = QLabel(ch_name)
                label.setFixedWidth(60)
                pw = pg.PlotWidget()
                pw.setBackground(None)
                scale = np.std(erp.data[i]) * 2
                pw.plot(erp.times, erp.data[i] / scale, pen=pg.mkPen(color=colors[i % len(colors)], width=2))
                pw.setXRange(0, erp.times[-1])
                pw.setMouseEnabled(x=True, y=False)
                pw.setLimits(xMin=0, xMax=erp.times[-1])
                pw.plotItem.showAxis('left', True)
                pw.plotItem.showAxis('bottom', True)
                pw.getAxis('left').setStyle(showValues=True)
                pw.getAxis('bottom').setStyle(showValues=True)
                pw.getAxis('left').setPen(None)
                pw.getAxis('bottom').setPen(None)
                row_layout.addWidget(label)
                row_layout.addWidget(pw)
                self.plot_layout.addLayout(row_layout)
        except Exception as e:
            QMessageBox.critical(self, '오류', f'ERP 플롯 중 오류 발생: {str(e)}')
    
    def update_plot_by_radio(self):
        self.clear_plot_layout()
        if self.raw_radio.isChecked():
            self.plot_data_preview()
        elif self.mne_radio.isChecked():
            self.plot_data()

    def plot_data(self):
        if self.raw is None:
            return
        try:
            self.clear_plot_layout()
            self.plot_lines = []  # 그래프 라인 저장 (매번 초기화)
            duration = self.plot_duration_spin.value()
            n_samples = int(duration * self.raw.info['sfreq'])
            times = self.raw.times[:n_samples]
            data = self.raw.get_data()[:, :n_samples]

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # bad_channel_list에서 체크된 채널을 bad로 간주
            bads = []
            if hasattr(self, 'bad_channel_list'):
                for i in range(self.bad_channel_list.count()):
                    item = self.bad_channel_list.item(i)
                    if item.checkState() == 2:
                        bads.append(item.text())

            for i, ch_name in enumerate(self.raw.ch_names):
                row_container = QWidget()
                row_layout = QHBoxLayout(row_container)
                row_layout.setContentsMargins(5, 5, 5, 5)
                row_layout.setSpacing(5)

                label = QLabel(ch_name)
                label.setFixedWidth(100)

                pw = pg.PlotWidget()
                pw.setMinimumWidth(400)
                pw.setBackground(None)  # 배경 투명

                # bad_channel_list에서 체크된 채널을 흐릿하게 표시
                is_bad = ch_name in bads
                alpha = 80 if is_bad else 255
                color = pg.mkColor(colors[i % len(colors)])
                color.setAlpha(alpha)
                pen = pg.mkPen(color=color, width=1)

                # 보간된 bad 채널이면: 보간 전 신호 흐릿하게, 보간 후 신호 진하게 두 번 그림
                if hasattr(self, '_pre_interpolate_data') and hasattr(self, '_pre_interpolate_bads') and ch_name in self._pre_interpolate_bads:
                    # 1. 보간 전 신호(흐릿하게)
                    pre_data = self._pre_interpolate_data[i][:n_samples]
                    faded_color = pg.mkColor(colors[i % len(colors)])
                    faded_color.setAlpha(80)
                    faded_pen = pg.mkPen(color=faded_color, width=1, style=Qt.DashLine)
                    pre_curve = pw.plot(times, pre_data, pen=faded_pen)
                    pre_curve.curve.setClickable(True)
                    pre_curve.curve.mouseClickEvent = partial(self.on_curve_clicked, idx=i)
                    self.plot_lines.append(pre_curve.curve)
                    # 2. 보간 후 신호(진하게)
                    strong_color = pg.mkColor(colors[i % len(colors)])
                    strong_color.setAlpha(255)
                    strong_pen = pg.mkPen(color=strong_color, width=1)
                    post_curve = pw.plot(times, data[i], pen=strong_pen)
                    post_curve.curve.setClickable(True)
                    post_curve.curve.mouseClickEvent = partial(self.on_curve_clicked, idx=i)
                    self.plot_lines.append(post_curve.curve)
                else:
                    # 일반 채널(혹은 bad가 아닌 경우) 기존 방식
                    plot_item = pw.plot(times, data[i], pen=pen)
                    plot_item.curve.setClickable(True)
                    plot_item.curve.mouseClickEvent = partial(self.on_curve_clicked, idx=i)
                    self.plot_lines.append(plot_item.curve)

                pw.setXRange(0, times[-1])
                pw.setMouseEnabled(x=True, y=False)
                pw.setLimits(xMin=0, xMax=times[-1])
                pw.plotItem.showAxis('left', True)
                pw.plotItem.showAxis('bottom', True)
                pw.getAxis('left').setStyle(showValues=True)
                pw.getAxis('bottom').setStyle(showValues=True)
                pw.getAxis('left').setPen(None)
                pw.getAxis('bottom').setPen(None)

                row_layout.addWidget(label, stretch=0)
                row_layout.addWidget(pw, stretch=1)
                self.plot_layout.addWidget(row_container)
        except Exception as e:
            print(f"[DEBUG] Exception in plot_data: {e}")
            QMessageBox.critical(self, '오류', f'데이터 플롯 중 오류 발생: {str(e)}')

    def _make_mouse_click_handler(self, idx):
        def handler(ev):
            self.toggle_bad_channel_by_curve(idx)
        return handler

    def on_curve_clicked(self, event, idx):
        """그래프 선 클릭 시 호출되는 함수"""
        self.toggle_bad_channel_by_curve(idx)

    
    def toggle_bad_channel_by_curve(self, idx):
        ch_name = self.raw.ch_names[idx]
        if ch_name in self.raw.info['bads']:
            self.raw.info['bads'].remove(ch_name)
            print(f"[DEBUG] {ch_name} removed from bads (via graph)")
        else:
            self.raw.info['bads'].append(ch_name)
            print(f"[DEBUG] {ch_name} added to bads (via graph)")

        # ✅ bad_channel_list 동기화
        if hasattr(self, 'bad_channel_list'):
            self.bad_channel_list.clear()
            for ch in self.raw.ch_names:
                item = QListWidgetItem(ch)
                if ch in self.raw.info['bads']:
                    item.setCheckState(2)
                else:
                    item.setCheckState(0)
                self.bad_channel_list.addItem(item)

        # ✅ 이벤트 루프가 끝난 후에 plot_data 실행
        QTimer.singleShot(0, self.plot_data)
        
    def show_result_df_in_table(self, df):
        df_t = df.T  # transpose: 채널이 행, 샘플이 열
        self.result_df_table.clear()
        n_rows = len(df_t)
        n_cols = min(100, len(df_t.columns))  # 최대 100개 샘플만 표시
        self.result_df_table.setRowCount(n_rows)
        self.result_df_table.setColumnCount(n_cols)
        self.result_df_table.setHorizontalHeaderLabels([str(df_t.columns[i]) for i in range(n_cols)])
        self.result_df_table.setVerticalHeaderLabels([str(idx) for idx in df_t.index])
        for i in range(n_rows):
            for j in range(n_cols):
                self.result_df_table.setItem(i, j, QTableWidgetItem(str(df_t.iloc[i, j])))
        self.result_df_table.resizeColumnsToContents()

    def plot_data_preview(self):
        if hasattr(self, 'df') and self.df is not None:
            self.clear_plot_layout()
            n_plot = min(1000, len(self.df))
            t = np.arange(n_plot)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            for i, col in enumerate(self.df.columns):
                row_layout = QHBoxLayout()
                label = QLabel(col)
                label.setFixedWidth(60)
                pw = pg.PlotWidget()
                pw.setBackground(None)
                pw.plot(t, self.df[col].values[:n_plot], pen=pg.mkPen(color=colors[i % len(colors)], width=1))
                pw.setXRange(0, t[-1])
                pw.setMouseEnabled(x=True, y=False)
                pw.setLimits(xMin=0, xMax=t[-1])
                pw.plotItem.showAxis('left', True)
                pw.plotItem.showAxis('bottom', True)
                pw.getAxis('left').setStyle(showValues=True)
                pw.getAxis('bottom').setStyle(showValues=True)
                pw.getAxis('left').setPen(None)
                pw.getAxis('bottom').setPen(None)
                row_layout.addWidget(label)
                row_layout.addWidget(pw)
                self.plot_layout.addLayout(row_layout)

    def auto_detect_bad_channels(self):
        if self.raw is None:
            QMessageBox.warning(self, '경고', '먼저 EEG 데이터를 로드하세요.')
            return
        method = self.auto_method_combo.currentText()
        bads = []
        data = self.raw.get_data()
        ch_names = self.raw.ch_names
        if method == '평균 제곱':
            # 평균 제곱이 전체 평균의 2배 이상인 채널
            ms = (data ** 2).mean(axis=1)
            threshold = ms.mean() + 2 * ms.std()
            bads = [ch for ch, v in zip(ch_names, ms) if v > threshold]
        elif method == '편평성':
            # 표준편차가 매우 작은 채널(편평)
            stds = data.std(axis=1)
            threshold = stds.mean() * 0.1
            bads = [ch for ch, v in zip(ch_names, stds) if v < threshold]
        elif method == '이상치':
            # 3시그마 이상 이상치가 많은 채널
            z = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
            outlier_ratio = (np.abs(z) > 3).mean(axis=1)
            bads = [ch for ch, v in zip(ch_names, outlier_ratio) if v > 0.1]
        # bad_channel_list를 모든 채널로 채우고, bads만 체크
        self.bad_channel_list.clear()
        for ch in ch_names:
            item = QListWidgetItem(ch)
            if ch in bads:
                item.setCheckState(2)  # 체크
            else:
                item.setCheckState(0)  # 체크 해제
            self.bad_channel_list.addItem(item)

    def add_bad_channel(self):
        ch = self.manual_bad_input.text().strip()
        if ch and (ch not in [self.bad_channel_list.item(i).text() for i in range(self.bad_channel_list.count())]):
            self.bad_channel_list.addItem(ch)
        self.manual_bad_input.clear()

    def remove_bad_channel(self):
        selected = self.bad_channel_list.currentRow()
        if selected >= 0:
            self.bad_channel_list.takeItem(selected)

    def run_interpolate(self):
        if self.raw is None:
            QMessageBox.warning(self, '경고', '먼저 EEG 데이터를 로드하세요.')
            return
        bads = []
        for i in range(self.bad_channel_list.count()):
            item = self.bad_channel_list.item(i)
            if item.checkState() == 2:  # 체크된 항목만 bad로
                bads.append(item.text())
        if not bads:
            QMessageBox.information(self, '안내', '불량 채널을 먼저 선택하세요.')
            return
        # 이미 보간된 bad 채널에 대해 중복 보간 시도 시 경고
        if hasattr(self, '_pre_interpolate_bads'):
            overlap = set(bads) & set(self._pre_interpolate_bads)
            if overlap:
                overlap_str = ', '.join(overlap)
                QMessageBox.warning(self, '경고', f'이미 보간된 채널({overlap_str})에 대해 중복 보간을 실행할 수 없습니다.')
                return
        # 보간 전 데이터 저장
        self._pre_interpolate_data = self.raw.get_data().copy()
        self._pre_interpolate_bads = list(bads)
        self.raw.info['bads'] = bads
        interp_mode = self.interpolate_method_combo.currentText()
        self.raw.interpolate_bads(reset_bads=True, mode=interp_mode)
        self.raw.info['bads'] = []
        QMessageBox.information(self, '완료', '보간이 완료되었습니다.')

    def apply_reference(self):
        if self.raw is None:
            QMessageBox.warning(self, '경고', '먼저 EEG 데이터를 로드하세요.')
            return
        if self.radio_avg.isChecked():
            self.raw.set_eeg_reference('average')
            QMessageBox.information(self, '완료', '공통 평균 참조가 적용되었습니다.')
        elif self.radio_custom.isChecked():
            ch_text = self.ref_channel_input.text().strip()
            if not ch_text:
                QMessageBox.warning(self, '경고', '참조 채널명을 입력하세요.')
                return
            ch_list = [ch.strip() for ch in ch_text.split(',') if ch.strip()]
            # 채널 존재 여부 확인
            not_found = [ch for ch in ch_list if ch not in self.raw.ch_names]
            if not_found:
                QMessageBox.warning(self, '경고', f'존재하지 않는 채널: {", ".join(not_found)}')
                return
            self.raw.set_eeg_reference(ch_list)
            QMessageBox.information(self, '완료', f'채널 {", ".join(ch_list)} 참조가 적용되었습니다.')

    def compare_erp(self):
        # ERP가 없는 경우 경고
        if not hasattr(self, 'epochs') or self.epochs is None:
            QMessageBox.warning(self, '경고', 'ERP(epochs)가 먼저 추출되어야 합니다.')
            return
        # 참조 적용 전/후 ERP 파형 비교
        # 1. 참조 전 ERP 복사
        erp_before = self.epochs.average().copy()
        # 2. 참조 적용 (현재 raw 기준)
        # epochs를 새로 생성하여 참조 후 ERP 계산
        epochs_after = mne.Epochs(self.raw, self.events, event_id=self.event_id,
                                 tmin=self.tmin_spin.value(), tmax=self.tmax_spin.value(),
                                 baseline=(self.baseline_start_spin.value(), self.baseline_end_spin.value()),
                                 preload=True)
        erp_after = epochs_after.average()
        # 3. 두 ERP 파형을 나란히 플롯
        fig = erp_before.plot(show=False, spatial_colors=True, titles='참조 전 ERP')
        fig2 = erp_after.plot(show=False, spatial_colors=True, titles='참조 후 ERP')
        fig.show()
        fig2.show()

    def clear_plot_layout(self):
        # plot_layout의 모든 아이템(위젯, 레이아웃 모두) 완전 삭제
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item is not None:
                # 레이아웃(행)인 경우 내부 위젯도 삭제
                layout = item.layout()
                if layout is not None:
                    while layout.count():
                        sub_item = layout.takeAt(0)
                        widget = sub_item.widget()
                        if widget is not None:
                            widget.setParent(None)
                    # 레이아웃 자체도 삭제
                    del layout
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                del item
        # plot_lines도 초기화
        self.plot_lines = []

    def on_bad_channel_list_changed(self, item):
        # bad_channel_list에서 체크박스가 변경될 때마다 그래프를 즉시 업데이트
        self.plot_data()

def main():
    app = QApplication(sys.argv)
    ex = EEGAnalyzerGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
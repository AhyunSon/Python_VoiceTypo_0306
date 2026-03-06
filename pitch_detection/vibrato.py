"""비브라토 분석 모듈.

HTML 원본의 analyzeVibrato()를 포팅.
피치 히스토리에서 진동 패턴을 추출하여
rate(진동 빈도)와 extent(진폭)를 계산.

사용법:
    analyzer = VibratoAnalyzer()
    analyzer.push(freq)  # 매 프레임 호출
    rate, extent = analyzer.get()
"""

import numpy as np
from collections import deque

PITCH_HISTORY_SIZE = 30
MIN_VIBRATO_RATE = 3.0    # Hz 이하: 비브라토로 안 봄
MAX_VIBRATO_RATE = 10.0   # Hz 이상: 비브라토로 안 봄
MIN_VIBRATO_EXTENT = 0.5  # 반음 이하: 비브라토로 안 봄


class VibratoAnalyzer:
    def __init__(self, frames_per_sec: float = 21.5):
        """frames_per_sec: 초당 프레임 수 (44100/2048 ≈ 21.5)"""
        self.fps = frames_per_sec
        self._history = deque(maxlen=PITCH_HISTORY_SIZE)
        self._rate = 0.0
        self._extent = 0.0

    def push(self, freq: float):
        """유효한 피치가 감지될 때마다 호출."""
        if freq <= 0:
            return
        self._history.append(freq)
        if len(self._history) >= 8:
            self._analyze()

    def get(self) -> tuple:
        """현재 비브라토 상태.
        Returns: (rate, extent)
            rate: 진동 빈도 (Hz). 0이면 비브라토 아님.
            extent: 진폭 (반음 단위).
        """
        return self._rate, self._extent

    def reset(self):
        self._history.clear()
        self._rate = 0.0
        self._extent = 0.0

    def _analyze(self):
        pitches = np.array(self._history, dtype=np.float64)
        n = len(pitches)

        # 평균 피치
        mean_pitch = np.mean(pitches)
        if mean_pitch < 1.0:
            self._rate = 0.0
            self._extent = 0.0
            return

        # 평균 대비 편차 (반음 단위)
        deviations = 12.0 * np.log2(pitches / mean_pitch)

        # 진폭: peak-to-peak / 2
        extent = (np.max(deviations) - np.min(deviations)) / 2.0

        # 영점 교차 횟수 → 진동 주파수
        zero_crossings = 0
        for i in range(1, n):
            if deviations[i - 1] * deviations[i] < 0:
                zero_crossings += 1

        # rate = (교차 횟수 / 2) / 구간 시간
        duration = (n - 1) / self.fps
        if duration > 0:
            rate = (zero_crossings / 2.0) / duration
        else:
            rate = 0.0

        # 유효성 검증
        if (MIN_VIBRATO_RATE <= rate <= MAX_VIBRATO_RATE
                and extent >= MIN_VIBRATO_EXTENT):
            self._rate = rate
            self._extent = extent
        else:
            self._rate = 0.0
            self._extent = 0.0

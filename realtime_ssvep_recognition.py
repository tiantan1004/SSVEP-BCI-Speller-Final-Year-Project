"""
Real-time SSVEP Recognition Module - Upgraded Version
实时SSVEP识别模块 - 升级版

升级内容:
1. ★ FBCCA (Filter Bank CCA) - 显著提升准确率
2. ★ 改进的预处理 (自适应Alpha抑制)
3. ★ 更好的置信度评估
4. 保留原有CCA/TRCA接口

原理说明:
- FBCCA 将信号分成多个子带 (sub-bands)，每个子带做CCA
- 高频子带包含更多谐波信息，且不受Alpha节律干扰
- 子带权重: a(n) = n^(-1.25) + 0.25，强调低阶子带但不忽视高阶
"""

import numpy as np
from scipy import signal
from collections import deque
import threading
import time


class RealTimePreprocessor:
    """
    实时信号预处理器 - 升级版
    
    改进:
    - 更合理的带通范围
    - 可选的Alpha抑制模式
    - CAR (Common Average Reference) 空间滤波
    """
    
    def __init__(self, fs, lowcut=6, highcut=90, notch=50, 
                 use_car=True, suppress_alpha=False):
        """
        参数:
            fs: 采样率
            lowcut: 带通滤波低频截止
            highcut: 带通滤波高频截止
            notch: 陷波频率 (None则跳过)
            use_car: 是否使用共同平均参考 (Common Average Reference)
            suppress_alpha: 是否额外抑制Alpha频段 (8-13Hz)
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = min(highcut, fs / 2 - 1)  # 不超过Nyquist
        self.notch_freq = notch
        self.use_car = use_car
        self.suppress_alpha = suppress_alpha
        
        self._design_filters()
    
    def _design_filters(self):
        """设计滤波器"""
        nyq = 0.5 * self.fs
        
        # 带通滤波器 — 使用更高阶以获得更陡峭的截止
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.bp_b, self.bp_a = signal.butter(5, [low, high], btype='band')
        
        # 陷波滤波器 (50Hz 工频干扰)
        if self.notch_freq and self.notch_freq < nyq:
            freq = self.notch_freq / nyq
            self.notch_b, self.notch_a = signal.iirnotch(freq, 35)
        else:
            self.notch_freq = None
    
    def process(self, data):
        """
        处理数据
        
        参数:
            data: shape (n_channels, n_samples)
        
        返回:
            filtered_data: 滤波后的数据
        """
        # 1) 共同平均参考 (CAR) — 减去所有通道均值
        if self.use_car and data.shape[0] > 1:
            data = data - np.mean(data, axis=0, keepdims=True)
        
        # 2) 带通滤波
        filtered = signal.filtfilt(self.bp_b, self.bp_a, data, axis=1)
        
        # 3) 陷波滤波
        if self.notch_freq:
            filtered = signal.filtfilt(self.notch_b, self.notch_a, filtered, axis=1)
        
        # 4) 去均值
        filtered = filtered - np.mean(filtered, axis=1, keepdims=True)
        
        return filtered


# ============================================
# ★ FBCCA — 核心升级算法
# ============================================

class FilterBankCCA:
    """
    Filter Bank CCA (FBCCA) 识别器
    
    原理:
      将信号通过 M 个带通滤波器 (子带)，
      对每个子带 m 做 CCA 得到 ρ_m(f)，
      最终得分: Σ w_m * ρ_m(f)^2
      
    子带划分:
      sub-band m: [m × 8Hz, 90Hz]  (m = 0, 1, 2, ...)
      这样第0个子带包含基频+所有谐波，
      第1个子带去掉了基频的Alpha干扰区域，
      高阶子带只保留高次谐波。
    
    权重:
      w_m = (m+1)^(-1.25) + 0.25
      强调低阶子带但不完全忽视高阶。
    
    参考文献:
      Chen et al. (2015) "Filter bank canonical correlation analysis
      for implementing a high-speed SSVEP-based brain–computer interface"
    """
    
    def __init__(self, frequencies, fs, n_harmonics=5, n_subbands=5):
        """
        参数:
            frequencies: 刺激频率列表
            fs: 采样率
            n_harmonics: CCA参考信号的谐波数量
            n_subbands: 子带数量 (通常3-7)
        """
        self.frequencies = np.array(frequencies)
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.n_subbands = n_subbands
        
        # 子带权重: w(n) = (n+1)^(-1.25) + 0.25
        self.subband_weights = np.array([
            (n + 1) ** (-1.25) + 0.25 for n in range(n_subbands)
        ])
        
        # 设计子带滤波器
        self.subband_filters = self._design_subband_filters()
        
        # 参考信号缓存
        self.reference_cache = {}
    
    def _design_subband_filters(self):
        """
        设计子带滤波器
        
        子带 m 的通带: [base_freq * (m+1) - 2, min_high_freq] Hz
        
        例如: 如果最低刺激频率是 8Hz:
          子带0: [6, 90] Hz — 包含基频和所有谐波
          子带1: [14, 90] Hz — 去掉基频区域（含Alpha）
          子带2: [22, 90] Hz — 只保留3次及以上谐波
          ...
        """
        nyq = self.fs / 2
        max_high = min(90, nyq - 1)
        
        base_freq = float(np.min(self.frequencies))
        
        filters = []
        for m in range(self.n_subbands):
            # 子带下限: 从基频开始逐步升高
            low = base_freq * (m + 1) - 2.0
            low = max(low, 2.0)  # 最低不低于2Hz
            
            # 子带上限: 固定为高频
            high = max_high
            
            if low >= high - 2:
                # 如果下限已经接近上限，跳过
                filters.append(None)
                continue
            
            try:
                b, a = signal.cheby1(4, 0.5, [low / nyq, high / nyq], btype='band')
                filters.append((b, a))
            except Exception:
                filters.append(None)
        
        return filters
    
    def _get_reference(self, freq, n_samples):
        """获取参考信号（带缓存）"""
        key = (freq, n_samples)
        
        if key not in self.reference_cache:
            t = np.arange(n_samples) / self.fs
            ref = []
            for h in range(1, self.n_harmonics + 1):
                ref.append(np.sin(2 * np.pi * freq * h * t))
                ref.append(np.cos(2 * np.pi * freq * h * t))
            self.reference_cache[key] = np.array(ref)
        
        return self.reference_cache[key]
    
    def _cca_corr(self, X, Y):
        """
        计算CCA最大典型相关系数
        
        参数:
            X: shape (n_channels, n_samples) — EEG数据
            Y: shape (2*n_harmonics, n_samples) — 参考信号
        
        返回:
            max_corr: 最大相关系数 [0, 1]
        """
        n = X.shape[1]
        
        # 去均值
        X = X - np.mean(X, axis=1, keepdims=True)
        Y = Y - np.mean(Y, axis=1, keepdims=True)
        
        # 协方差矩阵
        Cxx = (X @ X.T) / n
        Cyy = (Y @ Y.T) / n
        Cxy = (X @ Y.T) / n
        Cyx = (Y @ X.T) / n
        
        # 正则化
        reg = 1e-6
        Cxx += reg * np.eye(Cxx.shape[0])
        Cyy += reg * np.eye(Cyy.shape[0])
        
        try:
            Cxx_inv = np.linalg.inv(Cxx)
            Cyy_inv = np.linalg.inv(Cyy)
            M = Cxx_inv @ Cxy @ Cyy_inv @ Cyx
            eigenvalues = np.real(np.linalg.eigvals(M))
            max_corr = np.sqrt(np.clip(np.max(eigenvalues), 0, 1))
            return max_corr
        except np.linalg.LinAlgError:
            return 0.0
    
    def recognize(self, data):
        """
        FBCCA 识别
        
        参数:
            data: EEG数据, shape (n_channels, n_samples)
        
        返回:
            freq_idx: 识别的频率索引
            freq: 识别的频率值
            confidence: 置信度
            all_scores: 与所有频率的FBCCA得分
        """
        n_samples = data.shape[1]
        n_freqs = len(self.frequencies)
        
        # 对每个子带做CCA
        # scores[f] = Σ_m  w_m * ρ_m(f)^2
        scores = np.zeros(n_freqs)
        
        for m, filt in enumerate(self.subband_filters):
            if filt is None:
                continue
            
            b, a = filt
            
            # 子带滤波
            try:
                data_filtered = signal.filtfilt(b, a, data, axis=1)
            except Exception:
                continue
            
            # 对每个候选频率计算CCA
            for f_idx, freq in enumerate(self.frequencies):
                ref = self._get_reference(freq, n_samples)
                rho = self._cca_corr(data_filtered, ref)
                scores[f_idx] += self.subband_weights[m] * (rho ** 2)
        
        # 找最大得分
        freq_idx = np.argmax(scores)
        freq = self.frequencies[freq_idx]
        
        # 置信度: 最大得分与次大得分之比
        sorted_scores = np.sort(scores)[::-1]
        if len(sorted_scores) > 1 and sorted_scores[1] > 0:
            confidence = sorted_scores[0] / sorted_scores[1] - 1.0
        else:
            confidence = sorted_scores[0]
        
        return freq_idx, freq, confidence, scores


# ============================================
# 原有CCA（保留，作为对比基线）
# ============================================

class RealTimeCCA:
    """实时CCA识别器（原版，保留兼容）"""
    
    def __init__(self, frequencies, fs, n_harmonics=5):
        self.frequencies = np.array(frequencies)
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.reference_cache = {}
    
    def _get_reference(self, freq, n_samples):
        key = (freq, n_samples)
        if key not in self.reference_cache:
            t = np.arange(n_samples) / self.fs
            ref = []
            for h in range(1, self.n_harmonics + 1):
                ref.append(np.sin(2 * np.pi * freq * h * t))
                ref.append(np.cos(2 * np.pi * freq * h * t))
            self.reference_cache[key] = np.array(ref)
        return self.reference_cache[key]
    
    def _cca_correlation(self, X, Y):
        n_samples = X.shape[1]
        X = X - np.mean(X, axis=1, keepdims=True)
        Y = Y - np.mean(Y, axis=1, keepdims=True)
        Cxx = X @ X.T / n_samples
        Cyy = Y @ Y.T / n_samples
        Cxy = X @ Y.T / n_samples
        Cyx = Y @ X.T / n_samples
        reg = 1e-6
        Cxx += reg * np.eye(Cxx.shape[0])
        Cyy += reg * np.eye(Cyy.shape[0])
        try:
            Cxx_inv = np.linalg.inv(Cxx)
            Cyy_inv = np.linalg.inv(Cyy)
            M = Cxx_inv @ Cxy @ Cyy_inv @ Cyx
            eigenvalues, _ = np.linalg.eig(M)
            max_corr = np.sqrt(np.max(np.real(eigenvalues)))
            return np.clip(max_corr, 0, 1)
        except:
            return 0.0
    
    def recognize(self, data):
        n_samples = data.shape[1]
        correlations = []
        for freq in self.frequencies:
            ref = self._get_reference(freq, n_samples)
            corr = self._cca_correlation(data, ref)
            correlations.append(corr)
        correlations = np.array(correlations)
        freq_idx = np.argmax(correlations)
        freq = self.frequencies[freq_idx]
        sorted_corrs = np.sort(correlations)[::-1]
        if len(sorted_corrs) > 1:
            confidence = sorted_corrs[0] - sorted_corrs[1]
        else:
            confidence = sorted_corrs[0]
        return freq_idx, freq, confidence, correlations


class RealTimeTRCA:
    """实时TRCA识别器（保留兼容）"""
    
    def __init__(self, frequencies, fs):
        self.frequencies = np.array(frequencies)
        self.fs = fs
        self.is_trained = False
        self.spatial_filters = {}
        self.templates = {}
        self.training_data = {i: [] for i in range(len(frequencies))}
    
    def add_training_data(self, freq_idx, data):
        self.training_data[freq_idx].append(data.copy())
    
    def train(self):
        for freq_idx, trials in self.training_data.items():
            if len(trials) < 2:
                continue
            trial_data = np.array(trials)
            W = self._compute_trca_filter(trial_data)
            self.spatial_filters[freq_idx] = W
            self.templates[freq_idx] = np.mean(trial_data, axis=0)
        self.is_trained = len(self.spatial_filters) > 0
        print(f"TRCA模型训练完成，共 {len(self.spatial_filters)} 个类别")
    
    def _compute_trca_filter(self, data):
        n_trials, n_channels, n_samples = data.shape
        S = np.zeros((n_channels, n_channels))
        for i in range(n_trials):
            for j in range(n_trials):
                if i != j:
                    xi = data[i] - np.mean(data[i], axis=1, keepdims=True)
                    xj = data[j] - np.mean(data[j], axis=1, keepdims=True)
                    S += xi @ xj.T
        Q = np.zeros((n_channels, n_channels))
        for i in range(n_trials):
            xi = data[i] - np.mean(data[i], axis=1, keepdims=True)
            Q += xi @ xi.T
        Q += 1e-6 * np.eye(n_channels)
        try:
            from scipy.linalg import eig
            eigenvalues, eigenvectors = eig(S, Q)
            max_idx = np.argmax(np.real(eigenvalues))
            W = np.real(eigenvectors[:, max_idx])
            W = W / np.linalg.norm(W)
            return W
        except:
            return np.ones(n_channels) / np.sqrt(n_channels)
    
    def recognize(self, data):
        if not self.is_trained:
            raise RuntimeError("TRCA模型尚未训练")
        correlations = []
        for freq_idx in sorted(self.spatial_filters.keys()):
            W = self.spatial_filters[freq_idx]
            template = self.templates[freq_idx]
            data_filtered = W @ data
            template_filtered = W @ template
            corr = np.corrcoef(data_filtered, template_filtered)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        correlations = np.array(correlations)
        freq_idx = np.argmax(correlations)
        freq = self.frequencies[freq_idx]
        sorted_corrs = np.sort(correlations)[::-1]
        if len(sorted_corrs) > 1:
            confidence = sorted_corrs[0] - sorted_corrs[1]
        else:
            confidence = sorted_corrs[0]
        return freq_idx, freq, confidence, correlations


# ============================================
# 整合识别器（升级版）
# ============================================

class RealTimeSSVEPRecognizer:
    """
    实时SSVEP识别器 - 升级版
    
    支持方法:
      - 'cca':   原始CCA（基线）
      - 'fbcca': ★ Filter Bank CCA（推荐）
      - 'trca':  TRCA（需要训练数据）
    """
    
    def __init__(self, frequencies, fs, window_seconds=4.0, 
                 method='fbcca', confidence_threshold=0.1):
        """
        参数:
            frequencies: 刺激频率列表
            fs: 采样率
            window_seconds: 分析窗口长度(秒)
            method: 'cca', 'fbcca', 或 'trca'
            confidence_threshold: 置信度阈值
        """
        self.frequencies = frequencies
        self.fs = fs
        self.window_samples = int(window_seconds * fs)
        self.method = method
        self.confidence_threshold = confidence_threshold
        
        # 组件
        self.preprocessor = RealTimePreprocessor(fs, use_car=True)
        self.cca = RealTimeCCA(frequencies, fs, n_harmonics=5)
        self.fbcca = FilterBankCCA(
            frequencies, fs, 
            n_harmonics=5,
            n_subbands=min(5, max(2, int((fs / 2 - 10) / np.min(frequencies))))
        )
        self.trca = RealTimeTRCA(frequencies, fs)
        
        # 结果历史（用于平滑）
        self.result_history = deque(maxlen=5)
        
        # 状态
        self.last_result = None
        self.recognition_count = 0
        
        print(f"[SSVEP识别器] 方法={method.upper()}, "
              f"频率数={len(frequencies)}, fs={fs}Hz, "
              f"窗口={window_seconds}s ({self.window_samples}点)")
        
        if len(frequencies) > 20:
            print(f"  ⚠️ 警告: {len(frequencies)}个频率可能太多，"
                  f"建议减少到6-12个以获得更好的准确率")
    
    def set_method(self, method):
        """设置识别方法"""
        if method in ['cca', 'fbcca', 'trca']:
            self.method = method
            print(f"识别方法设置为: {method.upper()}")
    
    def add_training_data(self, freq_idx, data):
        """添加TRCA训练数据"""
        processed = self.preprocessor.process(data)
        self.trca.add_training_data(freq_idx, processed)
    
    def train_trca(self):
        """训练TRCA模型"""
        self.trca.train()
    
    def recognize(self, raw_data):
        """
        实时识别
        
        参数:
            raw_data: 原始EEG数据, shape (n_channels, n_samples)
        
        返回:
            result: 识别结果字典
        """
        if raw_data.shape[1] < self.window_samples:
            return None
        
        # 取最近的窗口
        data = raw_data[:, -self.window_samples:]
        
        # 预处理 (CAR + 带通 + 陷波 + 去均值)
        processed = self.preprocessor.process(data)
        
        # 识别
        if self.method == 'fbcca':
            freq_idx, freq, confidence, correlations = self.fbcca.recognize(processed)
        elif self.method == 'trca':
            if not self.trca.is_trained:
                # 回退到FBCCA
                freq_idx, freq, confidence, correlations = self.fbcca.recognize(processed)
            else:
                freq_idx, freq, confidence, correlations = self.trca.recognize(processed)
        else:
            # 原始CCA
            freq_idx, freq, confidence, correlations = self.cca.recognize(processed)
        
        is_valid = confidence >= self.confidence_threshold
        
        result = {
            'freq_idx': freq_idx,
            'frequency': freq,
            'confidence': confidence,
            'correlations': correlations,
            'is_valid': is_valid,
            'method': self.method,
            'timestamp': time.time()
        }
        
        if is_valid:
            self.result_history.append(freq_idx)
            self.recognition_count += 1
        
        self.last_result = result
        return result
    
    def get_smoothed_result(self):
        """获取平滑后的结果（投票机制）"""
        if len(self.result_history) == 0:
            return None, None
        from collections import Counter
        counts = Counter(self.result_history)
        freq_idx = counts.most_common(1)[0][0]
        freq = self.frequencies[freq_idx]
        return freq_idx, freq
    
    def reset(self):
        """重置状态"""
        self.result_history.clear()
        self.last_result = None
        self.recognition_count = 0


class SSVEPRecognitionThread(threading.Thread):
    """SSVEP识别线程 — 在后台持续识别"""
    
    def __init__(self, eeg_device, recognizer, recognition_interval=0.5):
        super().__init__(daemon=True)
        self.device = eeg_device
        self.recognizer = recognizer
        self.interval = recognition_interval
        self._stop_event = threading.Event()
        self._result_lock = threading.Lock()
        self._latest_result = None
        self.on_result_callback = None
    
    def run(self):
        while not self._stop_event.is_set():
            try:
                data = self.device.get_data(self.recognizer.window_samples)
                if data is not None:
                    result = self.recognizer.recognize(data)
                    if result:
                        with self._result_lock:
                            self._latest_result = result
                        if self.on_result_callback and result['is_valid']:
                            self.on_result_callback(result)
                time.sleep(self.interval)
            except Exception as e:
                print(f"识别错误: {e}")
                time.sleep(0.1)
    
    def stop(self):
        self._stop_event.set()
    
    def get_latest_result(self):
        with self._result_lock:
            return self._latest_result


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("实时SSVEP识别测试 - 升级版 (FBCCA)")
    print("=" * 60)
    
    # ── 测试1: 少量频率 (理想场景) ──
    frequencies_small = [8.0, 9.25, 10.5, 11.75, 13.0, 14.25]
    fs = 256
    
    print(f"\n【测试1】 6个频率, 步进~1.25Hz")
    print(f"  频率: {frequencies_small}")
    
    recognizer = RealTimeSSVEPRecognizer(
        frequencies=frequencies_small,
        fs=fs,
        window_seconds=4.0,
        method='fbcca'
    )
    
    for target_freq in frequencies_small:
        n_samples = 4 * fs
        t = np.arange(n_samples) / fs
        data = np.zeros((9, n_samples))
        for ch in range(9):
            for h in range(1, 4):
                data[ch] += (10.0 / h) * np.sin(2 * np.pi * target_freq * h * t)
            # Alpha节律 + 噪声
            data[ch] += 8 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2 * np.pi))
            data[ch] += 5 * np.random.randn(n_samples)
        
        result = recognizer.recognize(data)
        ok = "✓" if result['frequency'] == target_freq else "✗"
        print(f"  {ok} 目标: {target_freq}Hz → 识别: {result['frequency']}Hz "
              f"(conf: {result['confidence']:.3f})")
    
    # ── 测试2: 大量频率 (困难场景) ──
    print(f"\n{'='*60}")
    print(f"【测试2】 40个频率, 步进0.2Hz (困难)")
    frequencies_large = [round(8.0 + i * 0.2, 1) for i in range(40)]
    
    recognizer2 = RealTimeSSVEPRecognizer(
        frequencies=frequencies_large,
        fs=fs,
        window_seconds=4.0,
        method='fbcca'
    )
    
    # 只测试几个代表性频率
    test_freqs = [8.0, 10.0, 12.0, 14.0]
    correct = 0
    for target_freq in test_freqs:
        n_samples = 4 * fs
        t = np.arange(n_samples) / fs
        data = np.zeros((9, n_samples))
        for ch in range(9):
            for h in range(1, 4):
                data[ch] += (10.0 / h) * np.sin(2 * np.pi * target_freq * h * t)
            data[ch] += 8 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2 * np.pi))
            data[ch] += 5 * np.random.randn(n_samples)
        
        result = recognizer2.recognize(data)
        # 允许 ±0.4Hz 误差 (对于0.2Hz步进)
        err = abs(result['frequency'] - target_freq)
        ok = "✓" if err <= 0.4 else "✗"
        if err <= 0.4:
            correct += 1
        print(f"  {ok} 目标: {target_freq}Hz → 识别: {result['frequency']}Hz "
              f"(偏差: {err:.1f}Hz, conf: {result['confidence']:.3f})")
    
    print(f"\n  40频率准确率: {correct}/{len(test_freqs)} "
          f"({'建议减少频率数量!' if correct < len(test_freqs) else ''})")
    
    # ── 对比 CCA vs FBCCA ──
    print(f"\n{'='*60}")
    print(f"【测试3】 CCA vs FBCCA 对比 (6频率, 强Alpha噪声)")
    
    for method in ['cca', 'fbcca']:
        rec = RealTimeSSVEPRecognizer(
            frequencies=frequencies_small,
            fs=fs,
            window_seconds=4.0,
            method=method
        )
        correct = 0
        total = len(frequencies_small) * 3  # 每个频率测3次
        for target_freq in frequencies_small:
            for trial in range(3):
                n_samples = 4 * fs
                t = np.arange(n_samples) / fs
                data = np.zeros((9, n_samples))
                for ch in range(9):
                    for h in range(1, 4):
                        data[ch] += (8.0 / h) * np.sin(2 * np.pi * target_freq * h * t)
                    # 强Alpha: 幅值为SSVEP的1.5倍
                    data[ch] += 12 * np.sin(2 * np.pi * (9.5 + np.random.uniform(-0.5, 0.5)) * t)
                    data[ch] += 6 * np.random.randn(n_samples)
                
                result = rec.recognize(data)
                if result['frequency'] == target_freq:
                    correct += 1
        
        print(f"  {method.upper():>5s}: {correct}/{total} = {correct/total*100:.1f}%")
    
    print(f"\n测试完成！")

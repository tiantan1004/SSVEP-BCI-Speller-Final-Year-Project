"""
Data Saving Module
数据保存模块 - 保存实验数据和结果

功能:
- 保存原始EEG数据 (per trial)
- ★ 连续录制完整实验过程的9通道脑电信号 (从头到尾)
- ★ 事件标记系统 (Event Markers) - 记录每个trial的时间戳
- 保存实验结果（目标、识别结果、准确率、ITR）
- 保存实验设置
- 支持多种格式：CSV, NPY, MAT, JSON
"""

import numpy as np
import json
import os
import time
import threading
from datetime import datetime
from dataclasses import asdict


class DataSaver:
    """实验数据保存器"""
    
    def __init__(self, save_dir="experiment_data"):
        """
        参数:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        self.session_dir = save_dir
        self.session_id = None
        
        # 实验数据缓存 (per trial)
        self.trial_data = []      # 每个trial的EEG数据（识别窗口）
        self.trial_results = []   # 每个trial的结果
        self.block_results = []   # 每个block的结果
        self.settings = {}        # 实验设置
        
        # ==========================================
        # ★ 连续录制：完整实验过程脑电信号
        # ==========================================
        self._continuous_chunks = []       # 每次回调的数据块列表，元素shape: (n_samples, n_channels)
        self._is_recording_continuous = False
        self._recording_start_time = None
        self._continuous_fs = None
        self._continuous_n_channels = None
        self._eeg_device_ref = None
        self._original_callback = None    # 保存设备原有回调，以便恢复
        self._recording_lock = threading.Lock()
        
        # ★ 事件标记列表
        # 每条记录: {'time': float秒(相对录制开始), 'event': str, 'block': int, 'trial': int, ...}
        self.event_markers = []
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
    
    # ==========================================
    # ★ 连续录制接口
    # ==========================================

    def start_continuous_recording(self, eeg_device):
        """
        开始连续录制完整实验过程的脑电信号。
        通过挂接设备的 on_data_callback，每次设备推送新数据时都会自动追加。
        
        参数:
            eeg_device: EEGDeviceBase 实例（需已开始 streaming）
        
        调用时机:
            在 start_recognition() / 实验正式开始前调用。
        """
        if self._is_recording_continuous:
            print("警告: 连续录制已在进行中，忽略重复调用")
            return
        
        with self._recording_lock:
            self._continuous_chunks = []
            self._is_recording_continuous = True
            self._recording_start_time = time.time()
            self._continuous_fs = getattr(eeg_device, 'fs', None)
            self._continuous_n_channels = getattr(eeg_device, 'n_channels', None)
            self._eeg_device_ref = eeg_device
            self.event_markers = []
        
        # 保存原有回调并包装
        self._original_callback = eeg_device.on_data_callback
        
        def _data_hook(data):
            """
            data shape 取决于设备:
              - EEGDeviceBase 子类: (n_samples, n_channels)  [来自 _stream_loop]
              - GDSDevice callback: (n_scans, n_channels)
            """
            if self._is_recording_continuous:
                arr = np.array(data, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                with self._recording_lock:
                    self._continuous_chunks.append(arr)
            
            # 调用原有回调（不破坏识别线程）
            if self._original_callback is not None:
                self._original_callback(data)
        
        eeg_device.on_data_callback = _data_hook
        
        n_ch = self._continuous_n_channels or '?'
        fs   = self._continuous_fs or '?'
        print(f"[DataSaver] 开始连续录制: {n_ch}通道, fs={fs}Hz")
    
    def stop_continuous_recording(self):
        """
        停止连续录制。
        
        调用时机:
            实验完全结束 / stop_recognition() 之后调用。
        """
        if not self._is_recording_continuous:
            return
        
        # 先关闭录制标志（回调中不再追加数据）
        self._is_recording_continuous = False
        
        # 短暂等待，让可能正在执行的回调完成写入
        time.sleep(0.05)
        
        # 恢复设备原有回调
        if self._eeg_device_ref is not None:
            self._eeg_device_ref.on_data_callback = self._original_callback
        
        # 统计已录制样本数
        with self._recording_lock:
            total_samples = sum(c.shape[0] for c in self._continuous_chunks)
        duration = time.time() - (self._recording_start_time or time.time())
        print(f"[DataSaver] 停止连续录制: 共 {total_samples} 采样点, "
              f"约 {duration:.1f}s, {len(self.event_markers)} 个事件标记")
    
    # ★ 视觉延迟补偿常量 (秒)
    # 显示器从接收到信号到实际发光的延迟，典型值约 140ms
    VISUAL_LATENCY_SEC = 0.14

    def add_event_marker(self, event: str, block_idx: int = -1,
                         trial_idx: int = -1, extra: dict = None):
        """
        在连续录制的时间轴上打标记（不依赖连续录制是否启动，均可调用）。

        对 flickering_start / flickering_end 事件自动加上视觉延迟补偿
        (VISUAL_LATENCY_SEC = 0.14s)，使 EEG 时间戳与实际视觉刺激对齐。

        参数:
            event:      事件名称，如 'experiment_start', 'cue_start',
                        'flickering_start', 'flickering_end', 'block_complete'
            block_idx:  当前 block 索引（0-based）
            trial_idx:  当前 trial 索引（0-based）
            extra:      附加信息字典，如 {'target_char': 'a', 'freq': 10.0}

        时间（time）以相对实验录制开始的秒数表示；
        若录制尚未启动则使用绝对时间戳。
        """
        if self._recording_start_time is not None:
            t = time.time() - self._recording_start_time
        else:
            t = time.time()

        # ★ 对 flickering 事件加上视觉延迟补偿
        # 视觉刺激实际到达大脑的时间 = 标记时间 + 显示器延迟
        if event in ('flickering_start', 'flickering_end'):
            t = t + self.VISUAL_LATENCY_SEC

        # 计算对应的采样点位置
        sample_idx = None
        if self._recording_start_time is not None and self._continuous_fs:
            sample_idx = int(t * self._continuous_fs)

        marker = {
            'time_sec': round(t, 4),
            'sample_idx': sample_idx,
            'event': event,
            'block_idx': block_idx,
            'trial_idx': trial_idx,
            'wall_time': datetime.now().isoformat(),
            'visual_latency_applied': self.VISUAL_LATENCY_SEC if event in ('flickering_start', 'flickering_end') else 0.0
        }
        if extra:
            marker.update(extra)

        self.event_markers.append(marker)
    
    def get_continuous_eeg(self):
        """
        获取当前已录制的完整脑电数据（实时可调用）。
        
        返回:
            data:  np.ndarray, shape (n_channels, total_samples)  或 None
        """
        with self._recording_lock:
            if not self._continuous_chunks:
                return None
            all_data = np.concatenate(self._continuous_chunks, axis=0)  # (total_samples, n_channels)
        return all_data.T  # (n_channels, total_samples)
    
    # ==========================================
    # 原有 Session / Trial / Block 接口（保持不变）
    # ==========================================
    
    def start_session(self, settings=None, subject_id="unknown"):
        """
        开始新的实验会话
        
        参数:
            settings: 实验设置对象
            subject_id: 被试ID
        """
        # 生成会话ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{subject_id}_{timestamp}"
        
        # 创建会话目录
        self.session_dir = os.path.join(self.save_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 清空缓存
        self.trial_data = []
        self.trial_results = []
        self.block_results = []
        
        # 保存设置
        if settings:
            try:
                if hasattr(settings, '__dataclass_fields__'):
                    self.settings = asdict(settings)
                else:
                    self.settings = vars(settings).copy()
            except:
                self.settings = {'raw': str(settings)}
        
        self.settings['subject_id'] = subject_id
        self.settings['session_id'] = self.session_id
        self.settings['start_time'] = datetime.now().isoformat()
        
        print(f"开始新会话: {self.session_id}")
        print(f"数据将保存到: {self.session_dir}")
        
        return self.session_id
    
    def add_trial_data(self, eeg_data, target_char, recognized_char,
                       frequency, confidence, block_idx, trial_idx,
                       is_correct=None, correlations=None):
        """
        添加单个trial的识别窗口EEG数据（保持原逻辑不变）
        
        参数:
            eeg_data: EEG数据 (n_channels, n_samples) 或 None
            target_char: 目标字符
            recognized_char: 识别的字符
            frequency: 识别的频率
            confidence: 置信度
            block_idx: Block索引
            trial_idx: Trial索引
            is_correct: 是否正确
            correlations: 与各频率的相关系数
        """
        if is_correct is None:
            is_correct = (target_char == recognized_char)
        
        trial_result = {
            'block_idx': block_idx,
            'trial_idx': trial_idx,
            'target_char': target_char,
            'recognized_char': recognized_char,
            'frequency': frequency,
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        if correlations is not None:
            trial_result['correlations'] = (
                correlations.tolist() if isinstance(correlations, np.ndarray) else correlations
            )
        
        self.trial_results.append(trial_result)
        
        # 保存识别窗口EEG数据
        if eeg_data is not None:
            self.trial_data.append({
                'block_idx': block_idx,
                'trial_idx': trial_idx,
                'target_char': target_char,
                'data': eeg_data
            })
    
    def add_block_result(self, block_idx, target_text, result_text,
                         accuracy, itr, duration=None):
        """添加Block结果"""
        block_result = {
            'block_idx': block_idx,
            'target_text': target_text,
            'result_text': result_text,
            'accuracy': accuracy,
            'itr': itr,
            'n_chars': len(target_text),
            'n_correct': sum(1 for t, r in zip(target_text, result_text) if t == r),
            'timestamp': datetime.now().isoformat()
        }
        if duration:
            block_result['duration'] = duration
        self.block_results.append(block_result)
    
    # ==========================================
    # 保存接口
    # ==========================================
    
    def save_all(self, format='all'):
        """
        保存所有数据，包括完整连续脑电信号。
        
        参数:
            format: 'all', 'json', 'csv', 'npy', 'mat'
        
        生成文件:
            results.json              - 实验结果与统计
            trial_results.csv         - 每个trial的结果明细
            trial_eeg.npz             - 每个trial的识别窗口EEG
            raw_eeg_continuous.npz    - ★ 完整实验过程的连续脑电信号 (9ch × T)
            event_markers.json        - ★ 事件标记时间轴
            flash_eeg_segments.npz    - ★ 自动切出的闪烁段EEG (n_trials × 9ch × samples)
            experiment_data.mat       - MATLAB兼容格式（含上述所有数据）
        """
        if not self.session_dir:
            print("错误: 请先调用 start_session()")
            return None
        
        saved_files = []
        
        # 更新结束时间
        self.settings['end_time'] = datetime.now().isoformat()
        
        # 计算总体统计
        summary = self._calculate_summary()
        
        if format in ['all', 'json']:
            json_file = self._save_json(summary)
            saved_files.append(json_file)
        
        if format in ['all', 'csv']:
            csv_file = self._save_csv()
            saved_files.append(csv_file)
        
        if format in ['all', 'npy']:
            # 识别窗口EEG (per trial)
            if self.trial_data:
                npy_file = self._save_trial_eeg_npy()
                saved_files.append(npy_file)
            
            # ★ 完整连续脑电信号
            cont_file = self._save_continuous_eeg_npy()
            if cont_file:
                saved_files.append(cont_file)
            
            # ★ 事件标记
            marker_file = self._save_event_markers()
            if marker_file:
                saved_files.append(marker_file)
            
            # ★ 自动切出闪烁段
            flash_file = self._save_flash_segments()
            if flash_file:
                saved_files.append(flash_file)
        
        if format in ['all', 'mat']:
            mat_file = self._save_mat(summary)
            if mat_file:
                saved_files.append(mat_file)
        
        print(f"\n数据已保存到: {self.session_dir}")
        for f in saved_files:
            print(f"  - {os.path.basename(f)}")
        
        return saved_files
    
    # ==========================================
    # 私有：保存各格式
    # ==========================================
    
    def _calculate_summary(self):
        """计算总体统计"""
        summary = {
            'settings': self.settings,
            'blocks': self.block_results,
            'trials': self.trial_results,
            'statistics': {}
        }
        
        if self.block_results:
            accuracies = [b['accuracy'] for b in self.block_results]
            itrs = [b['itr'] for b in self.block_results]
            
            summary['statistics'] = {
                'total_blocks': len(self.block_results),
                'total_trials': len(self.trial_results),
                'total_correct': sum(1 for t in self.trial_results if t['is_correct']),
                'overall_accuracy': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies)),
                'overall_itr': float(np.mean(itrs)),
                'itr_std': float(np.std(itrs)),
                'max_accuracy': float(max(accuracies)),
                'min_accuracy': float(min(accuracies)),
            }
        
        # ★ 连续录制信息
        if self._continuous_chunks:
            total_samples = sum(c.shape[0] for c in self._continuous_chunks)
            summary['continuous_eeg_info'] = {
                'total_samples': total_samples,
                'n_channels': self._continuous_n_channels,
                'fs': self._continuous_fs,
                'duration_sec': round(
                    total_samples / self._continuous_fs, 3
                ) if self._continuous_fs else None,
                'n_event_markers': len(self.event_markers)
            }
        
        return summary
    
    def _save_json(self, summary):
        """保存为JSON格式"""
        filepath = os.path.join(self.session_dir, "results.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return filepath
    
    def _save_csv(self):
        """保存trial结果为CSV格式"""
        filepath = os.path.join(self.session_dir, "trial_results.csv")
        
        if not self.trial_results:
            return filepath
        
        fields = [f for f in self.trial_results[0].keys() if f != 'correlations']
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(','.join(fields) + '\n')
            for trial in self.trial_results:
                row = [str(trial.get(field, '')) for field in fields]
                f.write(','.join(row) + '\n')
        
        return filepath
    
    def _save_trial_eeg_npy(self):
        """保存每个trial的识别窗口EEG（原 _save_npy 重命名）"""
        filepath = os.path.join(self.session_dir, "trial_eeg.npz")
        
        if not self.trial_data:
            return filepath
        
        data_dict = {
            'eeg_data':       np.array([t['data'] for t in self.trial_data]),   # (n_trials, n_ch, n_samples)
            'block_indices':  np.array([t['block_idx'] for t in self.trial_data]),
            'trial_indices':  np.array([t['trial_idx'] for t in self.trial_data]),
            'target_chars':   np.array([t['target_char'] for t in self.trial_data])
        }
        np.savez(filepath, **data_dict)
        return filepath
    
    def _save_continuous_eeg_npy(self):
        """
        ★ 保存完整连续脑电信号。
        
        输出文件: raw_eeg_continuous.npz
        
        内容:
            eeg          - shape (n_channels, total_samples), float32
                           行 = 通道, 列 = 时间点（从实验开始到结束）
            fs           - 采样率 (scalar)
            n_channels   - 通道数 (scalar)
            duration_sec - 总时长秒数 (scalar)
            start_wall_time - 录制开始的 ISO 时间字符串
        """
        if not self._continuous_chunks:
            print("[DataSaver] 警告: 无连续脑电数据，跳过保存")
            return None
        
        filepath = os.path.join(self.session_dir, "raw_eeg_continuous.npz")
        
        with self._recording_lock:
            # 拼接所有块: (total_samples, n_channels)
            all_data = np.concatenate(self._continuous_chunks, axis=0).astype(np.float32)
        
        eeg = all_data.T  # (n_channels, total_samples)
        
        fs = float(self._continuous_fs) if self._continuous_fs else 0.0
        duration = float(eeg.shape[1] / fs) if fs > 0 else 0.0
        
        start_time_str = (
            datetime.fromtimestamp(self._recording_start_time).isoformat()
            if self._recording_start_time else ''
        )
        
        np.savez(
            filepath,
            eeg=eeg,                                     # (n_channels, total_samples)
            fs=np.array(fs),
            n_channels=np.array(eeg.shape[0]),
            total_samples=np.array(eeg.shape[1]),
            duration_sec=np.array(duration),
            start_wall_time=np.array(start_time_str)
        )
        
        print(f"[DataSaver] 连续脑电已保存: shape={eeg.shape}, "
              f"时长={duration:.1f}s, fs={fs:.0f}Hz")
        return filepath
    
    def _save_event_markers(self):
        """
        ★ 保存事件标记为JSON文件。
        
        输出文件: event_markers.json
        
        每条记录包含:
            time_sec   - 相对录制开始的秒数（可与连续脑电对齐）
            sample_idx - 对应连续脑电的采样点位置
            event      - 事件名称
            block_idx  - block索引
            trial_idx  - trial索引
            wall_time  - 绝对时间字符串
        """
        if not self.event_markers:
            return None
        
        filepath = os.path.join(self.session_dir, "event_markers.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.event_markers, f, indent=2, ensure_ascii=False, default=str)
        return filepath
    
    def _save_mat(self, summary):
        """保存为MAT格式（MATLAB兼容），含连续脑电"""
        try:
            from scipy.io import savemat
        except ImportError:
            print("提示: 安装scipy可以保存MAT格式 (pip install scipy)")
            return None
        
        filepath = os.path.join(self.session_dir, "experiment_data.mat")
        
        mat_dict = {
            'settings':      self.settings,
            'block_results': self.block_results,
            'trial_results': self.trial_results,
        }
        
        if 'statistics' in summary:
            mat_dict['statistics'] = summary['statistics']
        
        # 识别窗口EEG (per trial)
        if self.trial_data:
            mat_dict['trial_eeg']           = np.array([t['data'] for t in self.trial_data])
            mat_dict['trial_block_indices'] = np.array([t['block_idx'] for t in self.trial_data])
            mat_dict['trial_indices']       = np.array([t['trial_idx'] for t in self.trial_data])
            mat_dict['trial_target_chars']  = [t['target_char'] for t in self.trial_data]
        
        # ★ 连续脑电 (n_channels × total_samples)
        if self._continuous_chunks:
            with self._recording_lock:
                all_data = np.concatenate(self._continuous_chunks, axis=0).astype(np.float32)
            mat_dict['raw_eeg_continuous'] = all_data.T  # (n_channels, total_samples)
            mat_dict['raw_eeg_fs']         = float(self._continuous_fs or 0)
            mat_dict['raw_eeg_n_channels'] = int(self._continuous_n_channels or 0)
        
        # ★ 事件标记
        if self.event_markers:
            # 拆分为几个向量（MATLAB友好）
            mat_dict['marker_time_sec']   = np.array([m['time_sec'] for m in self.event_markers])
            mat_dict['marker_sample_idx'] = np.array(
                [m['sample_idx'] if m['sample_idx'] is not None else -1
                 for m in self.event_markers]
            )
            mat_dict['marker_event']      = [m['event'] for m in self.event_markers]
            mat_dict['marker_block_idx']  = np.array([m['block_idx'] for m in self.event_markers])
            mat_dict['marker_trial_idx']  = np.array([m['trial_idx'] for m in self.event_markers])
        
        # ★ 闪烁段 (自动切割)
        flash_segments = self._extract_flash_segments()
        if flash_segments:
            fs = float(self._continuous_fs or 1)
            n_ch = flash_segments[0]['data'].shape[0]
            max_len = max(seg['data'].shape[1] for seg in flash_segments)
            n_t = len(flash_segments)
            flash_arr = np.zeros((n_t, n_ch, max_len), dtype=np.float32)
            flash_len = np.zeros(n_t, dtype=np.int32)
            for i, seg in enumerate(flash_segments):
                L = seg['data'].shape[1]
                flash_arr[i, :, :L] = seg['data']
                flash_len[i] = L
            mat_dict['flash_eeg']     = flash_arr
            mat_dict['flash_lengths'] = flash_len
            mat_dict['flash_targets'] = [seg['target_char'] for seg in flash_segments]
            mat_dict['flash_blocks']  = np.array([seg['block_idx'] for seg in flash_segments])
            mat_dict['flash_trials']  = np.array([seg['trial_idx'] for seg in flash_segments])
        
        savemat(filepath, mat_dict)
        return filepath
    
    # ==========================================
    # ★ Flash段自动切割
    # ==========================================

    def _extract_flash_segments(self):
        """
        根据 event_markers 中的 flickering_start / flickering_end，
        从连续脑电信号中自动切出每个 trial 的闪烁段。

        无论 cue_duration、flickering_duration、pause_duration、rest_duration
        如何设置，都能正确切割——因为完全依赖事件标记的采样点位置，
        不依赖任何固定时长参数。

        返回:
            list[dict] —— 每个元素:
                'data':        np.ndarray, shape (n_channels, n_samples)
                'block_idx':   int
                'trial_idx':   int
                'target_char': str
                'start_sample': int
                'end_sample':   int
                'duration_sec': float
            若数据不足则返回空列表。
        """
        if not self._continuous_chunks or not self.event_markers:
            return []

        with self._recording_lock:
            eeg = np.concatenate(self._continuous_chunks, axis=0).astype(np.float32).T
            # eeg shape: (n_channels, total_samples)

        fs = self._continuous_fs or 1.0

        # 收集成对的 start/end 标记
        starts = [m for m in self.event_markers if m['event'] == 'flickering_start']
        ends   = [m for m in self.event_markers if m['event'] == 'flickering_end']

        # 按 (block_idx, trial_idx) 配对
        end_map = {}
        for m in ends:
            key = (m['block_idx'], m['trial_idx'])
            end_map[key] = m

        # 按 trial_results 顺序获取 target_char（兜底用 extra 中的信息）
        segments = []
        trial_counter = 0
        for s in starts:
            key = (s['block_idx'], s['trial_idx'])
            e = end_map.get(key)
            if e is None:
                trial_counter += 1
                continue

            s_start = s.get('sample_idx')
            s_end   = e.get('sample_idx')
            if s_start is None or s_end is None:
                trial_counter += 1
                continue

            s_start = max(0, int(s_start))
            s_end   = min(int(s_end), eeg.shape[1])
            if s_end <= s_start:
                trial_counter += 1
                continue

            # 确定 target_char: 优先从 trial_results 取，其次从 marker extra 取
            tgt_char = ''
            if trial_counter < len(self.trial_results):
                tgt_char = self.trial_results[trial_counter].get('target_char', '')
            if not tgt_char:
                tgt_char = s.get('target_char', '')

            segments.append({
                'data':         eeg[:, s_start:s_end],
                'block_idx':    int(s['block_idx']),
                'trial_idx':    int(s['trial_idx']),
                'target_char':  tgt_char,
                'start_sample': s_start,
                'end_sample':   s_end,
                'duration_sec': round((s_end - s_start) / fs, 4),
            })
            trial_counter += 1

        return segments

    def _save_flash_segments(self):
        """
        ★ 自动切出 flickering 段并保存。

        输出文件: flash_eeg_segments.npz

        内容:
            flash_eeg      - (n_trials, n_channels, max_samples), float32, 短段末尾零填充
            flash_lengths  - (n_trials,), 每段的实际采样点数
            flash_targets  - (n_trials,), 每段的目标字符
            flash_blocks   - (n_trials,), block索引
            flash_trials   - (n_trials,), trial索引
            fs             - 采样率
            n_channels     - 通道数
        """
        segments = self._extract_flash_segments()
        if not segments:
            print("[DataSaver] 警告: 无法切出闪烁段（缺少连续信号或事件标记）")
            return None

        fs = float(self._continuous_fs or 0)
        n_ch = segments[0]['data'].shape[0]
        max_len = max(seg['data'].shape[1] for seg in segments)
        n_trials = len(segments)

        flash_eeg     = np.zeros((n_trials, n_ch, max_len), dtype=np.float32)
        flash_lengths = np.zeros(n_trials, dtype=np.int32)
        flash_targets = []
        flash_blocks  = np.zeros(n_trials, dtype=np.int32)
        flash_trials  = np.zeros(n_trials, dtype=np.int32)

        for i, seg in enumerate(segments):
            L = seg['data'].shape[1]
            flash_eeg[i, :, :L] = seg['data']
            flash_lengths[i]    = L
            flash_targets.append(seg['target_char'])
            flash_blocks[i]     = seg['block_idx']
            flash_trials[i]     = seg['trial_idx']

        filepath = os.path.join(self.session_dir, "flash_eeg_segments.npz")
        np.savez(filepath,
            flash_eeg=flash_eeg,
            flash_lengths=flash_lengths,
            flash_targets=np.array(flash_targets),
            flash_blocks=flash_blocks,
            flash_trials=flash_trials,
            fs=np.array(fs),
            n_channels=np.array(n_ch),
        )

        print(f"[DataSaver] 闪烁段已保存: {n_trials}段, "
              f"shape={flash_eeg.shape}, 时长={[round(l/fs,3) for l in flash_lengths]}s")
        return filepath

    # ==========================================
    # 辅助
    # ==========================================
    
    def get_summary_text(self):
        """获取结果摘要文本"""
        if not self.block_results:
            return "暂无数据"
        
        summary = self._calculate_summary()
        stats = summary['statistics']
        
        text = f"""
╔══════════════════════════════════════════╗
║          实验结果摘要                      ║
╠══════════════════════════════════════════╣
║  会话ID: {self.session_id}
║  被试ID: {self.settings.get('subject_id', 'N/A')}
╠══════════════════════════════════════════╣
║  总Block数: {stats['total_blocks']}
║  总Trial数: {stats['total_trials']}
║  正确数:    {stats['total_correct']}
╠══════════════════════════════════════════╣
║  平均准确率: {stats['overall_accuracy']:.1f}% ± {stats['accuracy_std']:.1f}%
║  最高准确率: {stats['max_accuracy']:.1f}%
║  最低准确率: {stats['min_accuracy']:.1f}%
╠══════════════════════════════════════════╣
║  平均ITR: {stats['overall_itr']:.1f} ± {stats['itr_std']:.1f} bits/min
╠══════════════════════════════════════════╣
║  Block详情:
"""
        for block in self.block_results:
            text += f"║    Block {block['block_idx']+1}: {block['target_text']} → {block['result_text']} ({block['accuracy']:.0f}%)\n"
        
        # ★ 连续录制信息
        if 'continuous_eeg_info' in summary:
            ci = summary['continuous_eeg_info']
            text += (f"╠══════════════════════════════════════════╣\n"
                     f"║  连续脑电: {ci['n_channels']}ch × {ci['total_samples']} pts "
                     f"({ci['duration_sec']:.1f}s @ {ci['fs']}Hz)\n"
                     f"║  事件标记: {ci['n_event_markers']} 条\n")
        
        text += "╚══════════════════════════════════════════╝"
        return text


# ==========================================
# 快速保存器（简化版，保持不变）
# ==========================================

class QuickDataSaver:
    """快速数据保存器 - 简化版，用于快速集成"""
    
    def __init__(self, save_dir="experiment_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_experiment(self, target_texts, result_texts, accuracies, itrs,
                        settings=None, subject_id="unknown"):
        """快速保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{subject_id}_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        data = {
            'subject_id': subject_id,
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'settings': settings or {},
            'results': {'blocks': []},
            'summary': {
                'total_blocks': len(target_texts),
                'average_accuracy': float(np.mean(accuracies)) if accuracies else 0,
                'average_itr': float(np.mean(itrs)) if itrs else 0,
            }
        }
        
        for i, (target, result, acc, itr) in enumerate(zip(
                target_texts, result_texts, accuracies, itrs)):
            data['results']['blocks'].append({
                'block_idx': i,
                'target': target,
                'result': result,
                'accuracy': acc,
                'itr': itr
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"实验数据已保存: {filepath}")
        return filepath


# ==========================================
# 测试
# ==========================================

if __name__ == "__main__":
    print("=" * 55)
    print("数据保存模块测试（含连续录制功能）")
    print("=" * 55)

    # --------------------------------------------------
    # 模拟一个极简EEG设备（只需要 on_data_callback + fs + n_channels）
    # --------------------------------------------------
    class _MockDevice:
        def __init__(self, fs=256, n_channels=9):
            self.fs = fs
            self.n_channels = n_channels
            self.on_data_callback = None

        def push_data(self, n_samples=8):
            """模拟产生新数据并触发回调"""
            chunk = np.random.randn(n_samples, self.n_channels).astype(np.float32)
            if self.on_data_callback:
                self.on_data_callback(chunk)
            return chunk

    device = _MockDevice(fs=256, n_channels=9)

    # 创建保存器
    saver = DataSaver(save_dir="test_data")
    saver.start_session(subject_id="test_subject")

    # ★ 开始连续录制
    saver.start_continuous_recording(device)
    saver.add_event_marker('experiment_start', block_idx=0, trial_idx=0)

    # 模拟3个block，每个block若干trial
    frequencies = [8.0, 9.0, 10.0, 11.0, 12.0]
    for block_idx in range(2):
        target_text = "hi"
        result_text = ""

        for trial_idx, target_char in enumerate(target_text):
            # 模拟cue阶段：设备推送数据
            saver.add_event_marker('cue_start', block_idx=block_idx, trial_idx=trial_idx,
                                   extra={'target_char': target_char})
            for _ in range(10):
                device.push_data(n_samples=8)       # 模拟 ~80/256 ≈ 0.3s

            # 模拟flickering阶段
            saver.add_event_marker('flickering_start', block_idx=block_idx, trial_idx=trial_idx,
                                   extra={'target_char': target_char})
            eeg_window = None
            for _ in range(30):
                chunk = device.push_data(n_samples=8)
                if eeg_window is None:
                    eeg_window = chunk
                else:
                    eeg_window = np.vstack([eeg_window, chunk])

            # eeg_window: (n_samples, n_channels) → 转置为 (n_channels, n_samples)
            eeg_window_ch_first = eeg_window.T

            recognized_char = target_char  # 模拟正确识别
            result_text += recognized_char

            saver.add_event_marker('flickering_end', block_idx=block_idx, trial_idx=trial_idx,
                                   extra={'recognized_char': recognized_char,
                                          'correct': recognized_char == target_char})

            saver.add_trial_data(
                eeg_data=eeg_window_ch_first,
                target_char=target_char,
                recognized_char=recognized_char,
                frequency=frequencies[trial_idx % len(frequencies)],
                confidence=0.9,
                block_idx=block_idx,
                trial_idx=trial_idx
            )

        saver.add_event_marker('block_complete', block_idx=block_idx, trial_idx=-1)
        correct = sum(t == r for t, r in zip(target_text, result_text))
        saver.add_block_result(
            block_idx=block_idx,
            target_text=target_text,
            result_text=result_text,
            accuracy=correct / len(target_text) * 100,
            itr=35.0
        )

    saver.add_event_marker('experiment_end', block_idx=-1, trial_idx=-1)

    # ★ 停止连续录制
    saver.stop_continuous_recording()

    # 查看连续数据
    cont = saver.get_continuous_eeg()
    if cont is not None:
        print(f"\n连续脑电数据形状: {cont.shape}  (n_channels × total_samples)")

    # 保存所有
    saved_files = saver.save_all()

    # 摘要
    print(saver.get_summary_text())
    print("\n测试完成！")

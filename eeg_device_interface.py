"""
EEG Device Interface Module
EEG设备接口模块 - 支持多种EEG采集设备

支持设备:
- OpenBCI (Cyton, Ganglion)
- Neuracle
- 模拟数据 (用于测试)
- LSL (Lab Streaming Layer) 通用接口
"""

import numpy as np
import time
import threading
from collections import deque
from abc import ABC, abstractmethod


class EEGDeviceBase(ABC):
    """EEG设备基类"""
    
    def __init__(self, fs=600, n_channels=9, buffer_seconds=15):
        """
        参数:
            fs: 采样率
            n_channels: 通道数
            buffer_seconds: 缓冲区长度(秒)
        """
        self.fs = fs
        self.n_channels = n_channels
        self.buffer_size = int(fs * buffer_seconds)
        
        # 数据缓冲区
        self.buffer = deque(maxlen=self.buffer_size)
        
        # 状态
        self.is_connected = False
        self.is_streaming = False
        self._stream_thread = None
        self._stop_event = threading.Event()
        
        # 回调函数
        self.on_data_callback = None
    
    @abstractmethod
    def connect(self):
        """连接设备"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开设备"""
        pass
    
    @abstractmethod
    def _read_data(self):
        """读取数据（由子类实现）"""
        pass
    
    def start_streaming(self):
        """开始数据流"""
        if not self.is_connected:
            raise RuntimeError("设备未连接")
        
        self._stop_event.clear()
        self.is_streaming = True
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        print(f"开始数据采集 (fs={self.fs}Hz, channels={self.n_channels})")
    
    def stop_streaming(self):
        """停止数据流"""
        self._stop_event.set()
        self.is_streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2)
        print("停止数据采集")
    
    def _stream_loop(self):
        """数据流循环"""
        while not self._stop_event.is_set():
            try:
                data = self._read_data()
                if data is not None:
                    # 添加到缓冲区
                    for sample in data:
                        self.buffer.append(sample)
                    
                    # 回调
                    if self.on_data_callback:
                        self.on_data_callback(data)
            except Exception as e:
                print(f"数据读取错误: {e}")
                time.sleep(0.01)
    
    def get_data(self, n_samples):
        """
        获取最近n_samples个采样点的数据
        
        返回:
            data: shape (n_channels, n_samples)
        """
        if len(self.buffer) < n_samples:
            return None
        
        data = list(self.buffer)[-n_samples:]
        return np.array(data).T  # shape: (n_channels, n_samples)
    
    def get_data_seconds(self, seconds):
        """获取最近n秒的数据"""
        n_samples = int(seconds * self.fs)
        return self.get_data(n_samples)
    
    def clear_buffer(self):
        """清空缓冲区"""
        self.buffer.clear()


class SimulatedEEG(EEGDeviceBase):
    """
    模拟EEG设备 - 用于测试和演示
    生成带有SSVEP特征的模拟脑电信号
    """
    
    def __init__(self, fs=250, n_channels=8, frequencies=None, **kwargs):
        super().__init__(fs, n_channels, **kwargs)
        
        self.frequencies = frequencies or [8.0, 9.0, 10.0, 11.0, 12.0]
        self.current_target_freq = self.frequencies[0]
        self.snr_db = 0  # 信噪比
        
        self._sample_counter = 0
        self._read_interval = 0.04  # 40ms读取一次
        self._samples_per_read = int(self.fs * self._read_interval)
    
    def connect(self):
        """模拟连接"""
        self.is_connected = True
        print("模拟EEG设备已连接")
        return True
    
    def disconnect(self):
        """模拟断开"""
        self.stop_streaming()
        self.is_connected = False
        print("模拟EEG设备已断开")
    
    def set_target_frequency(self, freq):
        """设置当前目标频率（模拟用户注视）"""
        self.current_target_freq = freq
    
    def _read_data(self):
        """生成模拟SSVEP数据"""
        time.sleep(self._read_interval)
        
        n_samples = self._samples_per_read
        t = (self._sample_counter + np.arange(n_samples)) / self.fs
        self._sample_counter += n_samples
        
        data = np.zeros((n_samples, self.n_channels))
        
        freq = self.current_target_freq
        
        for ch in range(self.n_channels):
            # SSVEP信号 (基频 + 谐波)
            signal = np.zeros(n_samples)
            for h in range(1, 4):
                amplitude = 10.0 / h  # 微伏级别
                phase = np.random.uniform(0, 0.5) * ch
                signal += amplitude * np.sin(2 * np.pi * freq * h * t + phase)
            
            # 添加噪声
            noise_power = np.var(signal) / (10 ** (self.snr_db / 10))
            noise = np.sqrt(noise_power) * np.random.randn(n_samples)
            
            # 添加alpha波背景 (8-12Hz)
            alpha = 5 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
            
            data[:, ch] = signal + noise + alpha
        
        return data


class LSLReceiver(EEGDeviceBase):
    """
    Lab Streaming Layer (LSL) 接收器
    通用接口，支持任何LSL兼容的EEG设备
    """
    
    def __init__(self, stream_name=None, stream_type='EEG', **kwargs):
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.inlet = None
        super().__init__(**kwargs)
    
    def connect(self):
        """连接LSL流"""
        try:
            from pylsl import StreamInlet, resolve_stream
            
            print(f"搜索LSL流 (type={self.stream_type})...")
            
            if self.stream_name:
                streams = resolve_stream('name', self.stream_name, timeout=5)
            else:
                streams = resolve_stream('type', self.stream_type, timeout=5)
            
            if not streams:
                print("未找到LSL流")
                return False
            
            self.inlet = StreamInlet(streams[0])
            info = self.inlet.info()
            
            self.fs = info.nominal_srate()
            self.n_channels = info.channel_count()
            self.buffer_size = int(self.fs * 10)
            self.buffer = deque(maxlen=self.buffer_size)
            
            self.is_connected = True
            print(f"已连接LSL流: {info.name()}")
            print(f"  采样率: {self.fs} Hz")
            print(f"  通道数: {self.n_channels}")
            
            return True
            
        except ImportError:
            print("需要安装pylsl: pip install pylsl")
            return False
        except Exception as e:
            print(f"LSL连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开LSL流"""
        self.stop_streaming()
        if self.inlet:
            self.inlet.close_stream()
        self.inlet = None
        self.is_connected = False
        print("LSL流已断开")
    
    def _read_data(self):
        """从LSL读取数据"""
        if self.inlet is None:
            return None
        
        samples, timestamps = self.inlet.pull_chunk(timeout=0.1)
        
        if samples:
            return np.array(samples)
        return None


class GDSDevice(EEGDeviceBase):
    """
    g.tec gDS 设备接口 (pygds)
    支持 g.USBamp, g.HIamp, g.Nautilus 等g.tec设备
    """

    def __init__(self, fs=256, n_channels=9, n_scans=8, **kwargs):
        """
        参数:
            fs: 采样率 (默认256Hz)
            n_channels: 采集通道数 (默认9)
            n_scans: 每次回调返回的采样点数 (默认8)
        """
        super().__init__(fs=fs, n_channels=n_channels, **kwargs)
        self.n_scans = n_scans
        self.device = None
        self._stop_event = threading.Event()
        self._acq_thread = None

        # 滤波器设置 (7~64Hz 带通, 50Hz 陷波)
        self.bp_low = 7.0
        self.bp_high = 64.0
        self.notch_freq = 50.0

    def connect(self):
        """连接g.tec设备"""
        try:
            import pygds as g

            self.device = g.GDS()
            self.device.SamplingRate = self.fs
            self.device.NumberOfScans = self.n_scans

            # 自动选择最接近目标的滤波器
            bp_filters = self.device.GetBandpassFilters()[0]
            notch_filters = self.device.GetNotchFilters()[0]

            bp_avail = [f for f in bp_filters if f['SamplingRate'] == self.fs]
            n_avail  = [f for f in notch_filters if f['SamplingRate'] == self.fs]

            target_bp = min(bp_avail,
                key=lambda x: abs(x['LowerCutoffFrequency'] - self.bp_low)
                             + abs(x['UpperCutoffFrequency'] - self.bp_high)
            ) if bp_avail else None

            target_n = min(n_avail,
                key=lambda x: abs((x['LowerCutoffFrequency'] + x['UpperCutoffFrequency']) / 2
                                  - self.notch_freq)
            ) if n_avail else None

            for i, ch in enumerate(self.device.Channels):
                ch.Acquire = (i < self.n_channels)
                if target_bp:
                    ch.BandpassFilterIndex = target_bp['BandpassFilterIndex']
                if target_n:
                    ch.NotchFilterIndex = target_n['NotchFilterIndex']

            self.device.SetConfiguration()
            self.is_connected = True

            bp_info = f"{target_bp['LowerCutoffFrequency']}~{target_bp['UpperCutoffFrequency']}Hz" if target_bp else "无"
            n_info  = f"{self.notch_freq}Hz" if target_n else "无"
            print(f"g.tec 设备已连接: {self.fs}Hz, {self.n_channels}通道")
            print(f"  带通滤波: {bp_info}  陷波: {n_info}")
            return True

        except ImportError:
            print("错误: 未安装 pygds，请确认 g.tec 驱动已安装")
            return False
        except Exception as e:
            print(f"g.tec 连接失败: {e}")
            return False

    def disconnect(self):
        """断开设备"""
        self.stop_streaming()
        if self.device:
            try:
                self.device.Close()
            except Exception:
                pass
        self.device = None
        self.is_connected = False
        print("g.tec 设备已断开")

    def start_streaming(self):
        """开始采集 — 在独立线程里运行 GetData 回调"""
        if not self.is_connected:
            raise RuntimeError("设备未连接")
        self._stop_event.clear()
        self.is_streaming = True
        self._acq_thread = threading.Thread(target=self._acq_loop, daemon=True)
        self._acq_thread.start()
        print(f"g.tec 开始采集 (fs={self.fs}Hz, channels={self.n_channels})")

    def stop_streaming(self):
        """停止采集"""
        self._stop_event.set()
        self.is_streaming = False
        if self._acq_thread:
            self._acq_thread.join(timeout=3)
        print("g.tec 停止采集")

    def _acq_loop(self):
        """采集线程主循环，使用 pygds 的 more 回调模式"""
        def more(new_data):
            # new_data shape: (n_scans, n_channels)
            arr = np.array(new_data)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # 只取前 n_channels 列（排除加速度等附加通道）
            arr = arr[:, :self.n_channels]

            for sample in arr:
                self.buffer.append(sample)

            if self.on_data_callback:
                self.on_data_callback(arr)

            # 返回 False 停止采集
            return not self._stop_event.is_set()

        try:
            self.device.GetData(self.n_scans, more=more)
        except Exception as e:
            if not self._stop_event.is_set():
                print(f"g.tec 采集异常: {e}")

    def _read_data(self):
        """GDSDevice 使用回调模式，此方法不被调用"""
        return None


class OpenBCIDevice(EEGDeviceBase):
    """
    OpenBCI设备接口
    支持Cyton (8通道) 和 Ganglion (4通道)
    """
    
    def __init__(self, port=None, board_type='cyton', **kwargs):
        """
        参数:
            port: 串口号 (如 '/dev/tty.usbserial-xxx' 或 'COM3')
            board_type: 'cyton' 或 'ganglion'
        """
        self.port = port
        self.board_type = board_type
        self.board = None
        
        if board_type == 'cyton':
            n_channels = 8
            fs = 250
        else:  # ganglion
            n_channels = 4
            fs = 200
        
        super().__init__(fs=fs, n_channels=n_channels, **kwargs)
    
    def connect(self):
        """连接OpenBCI"""
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
            
            params = BrainFlowInputParams()
            
            if self.port:
                params.serial_port = self.port
            
            if self.board_type == 'cyton':
                board_id = BoardIds.CYTON_BOARD
            else:
                board_id = BoardIds.GANGLION_BOARD
            
            self.board = BoardShim(board_id, params)
            self.board.prepare_session()
            
            self.is_connected = True
            print(f"OpenBCI {self.board_type} 已连接")
            return True
            
        except ImportError:
            print("需要安装brainflow: pip install brainflow")
            return False
        except Exception as e:
            print(f"OpenBCI连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开OpenBCI"""
        self.stop_streaming()
        if self.board:
            self.board.release_session()
        self.board = None
        self.is_connected = False
        print("OpenBCI已断开")
    
    def start_streaming(self):
        """开始采集"""
        if self.board:
            self.board.start_stream()
        super().start_streaming()
    
    def stop_streaming(self):
        """停止采集"""
        super().stop_streaming()
        if self.board:
            self.board.stop_stream()
    
    def _read_data(self):
        """读取OpenBCI数据"""
        if self.board is None:
            return None
        
        time.sleep(0.04)
        
        from brainflow.board_shim import BoardShim
        
        data = self.board.get_board_data()
        
        if data.size > 0:
            # 获取EEG通道
            eeg_channels = BoardShim.get_eeg_channels(self.board.board_id)
            eeg_data = data[eeg_channels, :].T
            return eeg_data
        
        return None


# ============================================
# 设备管理器
# ============================================

class EEGDeviceManager:
    """EEG设备管理器 - 统一接口"""
    
    DEVICE_TYPES = {
        'simulated': SimulatedEEG,
        'lsl': LSLReceiver,
        'openbci': OpenBCIDevice,
        'gds': GDSDevice,
    }
    
    @classmethod
    def create_device(cls, device_type='simulated', **kwargs):
        """
        创建EEG设备实例
        
        参数:
            device_type: 'simulated', 'lsl', 'openbci'
            **kwargs: 设备特定参数
        """
        if device_type not in cls.DEVICE_TYPES:
            raise ValueError(f"不支持的设备类型: {device_type}")
        
        device_class = cls.DEVICE_TYPES[device_type]
        return device_class(**kwargs)
    
    @classmethod
    def list_devices(cls):
        """列出可用设备类型"""
        return list(cls.DEVICE_TYPES.keys())


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("EEG设备接口测试")
    print("=" * 50)
    
    # 创建模拟设备
    device = EEGDeviceManager.create_device(
        'simulated',
        fs=250,
        n_channels=8,
        frequencies=[8.0, 10.0, 12.0]
    )
    
    # 连接
    device.connect()
    
    # 设置目标频率
    device.set_target_frequency(10.0)
    
    # 开始采集
    device.start_streaming()
    
    # 采集3秒
    print("\n采集数据中...")
    time.sleep(3)
    
    # 获取数据
    data = device.get_data_seconds(2)
    if data is not None:
        print(f"\n获取到数据: shape = {data.shape}")
        print(f"数据范围: {data.min():.2f} ~ {data.max():.2f} μV")
    
    # 停止
    device.stop_streaming()
    device.disconnect()
    
    print("\n测试完成！")

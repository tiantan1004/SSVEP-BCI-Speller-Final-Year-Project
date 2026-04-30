"""
Sound Feedback Module
声音反馈模块 - 提供实验中的声音提示

功能:
- 识别成功/失败提示音
- Block完成提示音
- 实验开始/结束提示音
- 支持开关和音量调节
"""

import numpy as np
import os

# 尝试导入pygame mixer
try:
    import pygame.mixer as mixer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("警告: pygame.mixer 不可用，声音功能将被禁用")


class SoundGenerator:
    """生成简单的音效"""
    
    @staticmethod
    def generate_sine_wave(frequency, duration, sample_rate=44100, volume=0.5):
        """
        生成正弦波音效
        
        参数:
            frequency: 频率 (Hz)
            duration: 持续时间 (秒)
            sample_rate: 采样率
            volume: 音量 (0-1)
        """
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # 应用淡入淡出
        fade_samples = int(sample_rate * 0.02)  # 20ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out
        
        # 调整音量并转换为16位整数
        wave = (wave * volume * 32767).astype(np.int16)
        
        # 转换为立体声
        stereo_wave = np.column_stack((wave, wave))
        
        return stereo_wave
    
    @staticmethod
    def generate_success_sound(sample_rate=44100):
        """生成成功提示音 - 上升音调"""
        duration = 0.15
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # 频率从400Hz上升到800Hz
        freq = np.linspace(400, 800, len(t))
        wave = np.sin(2 * np.pi * freq * t / sample_rate * np.arange(len(t)))
        
        # 使用简单的正弦波组合
        wave = np.sin(2 * np.pi * 600 * t) * 0.5 + np.sin(2 * np.pi * 900 * t) * 0.3
        
        # 淡出
        fade_out = np.linspace(1, 0, len(t))
        wave *= fade_out
        
        wave = (wave * 0.4 * 32767).astype(np.int16)
        return np.column_stack((wave, wave))
    
    @staticmethod
    def generate_error_sound(sample_rate=44100):
        """生成错误提示音 - 低沉短促"""
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # 低频音
        wave = np.sin(2 * np.pi * 200 * t) * 0.6 + np.sin(2 * np.pi * 250 * t) * 0.4
        
        # 快速淡出
        fade_out = np.exp(-t * 15)
        wave *= fade_out
        
        wave = (wave * 0.4 * 32767).astype(np.int16)
        return np.column_stack((wave, wave))
    
    @staticmethod
    def generate_complete_sound(sample_rate=44100):
        """生成完成提示音 - 和弦"""
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # C大调和弦 (C-E-G)
        wave = (np.sin(2 * np.pi * 523 * t) * 0.4 +  # C5
                np.sin(2 * np.pi * 659 * t) * 0.3 +  # E5
                np.sin(2 * np.pi * 784 * t) * 0.3)   # G5
        
        # 淡入淡出
        envelope = np.ones(len(t))
        fade_in = int(sample_rate * 0.05)
        fade_out = int(sample_rate * 0.2)
        envelope[:fade_in] = np.linspace(0, 1, fade_in)
        envelope[-fade_out:] = np.linspace(1, 0, fade_out)
        wave *= envelope
        
        wave = (wave * 0.35 * 32767).astype(np.int16)
        return np.column_stack((wave, wave))
    
    @staticmethod
    def generate_start_sound(sample_rate=44100):
        """生成开始提示音 - 上升音阶"""
        duration = 0.4
        samples_per_note = int(sample_rate * duration / 3)
        
        waves = []
        frequencies = [523, 659, 784]  # C5, E5, G5
        
        for freq in frequencies:
            t = np.linspace(0, duration/3, samples_per_note, False)
            wave = np.sin(2 * np.pi * freq * t)
            
            # 每个音符的包络
            envelope = np.ones(len(t))
            fade = int(len(t) * 0.1)
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            wave *= envelope
            
            waves.append(wave)
        
        wave = np.concatenate(waves)
        wave = (wave * 0.35 * 32767).astype(np.int16)
        return np.column_stack((wave, wave))
    
    @staticmethod
    def generate_beep(frequency=440, duration=0.1, sample_rate=44100):
        """生成简单蜂鸣音"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)
        
        # 淡入淡出
        fade_samples = min(int(sample_rate * 0.01), len(t) // 4)
        if fade_samples > 0:
            wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        wave = (wave * 0.3 * 32767).astype(np.int16)
        return np.column_stack((wave, wave))


class SoundFeedback:
    """声音反馈管理器"""
    
    def __init__(self, enabled=True, volume=0.7):
        """
        参数:
            enabled: 是否启用声音
            volume: 音量 (0-1)
        """
        self.enabled = enabled
        self.volume = volume
        self.initialized = False
        self.sounds = {}
        
        self._init_mixer()
    
    def _init_mixer(self):
        """初始化音频混合器"""
        if not PYGAME_AVAILABLE:
            self.initialized = False
            return
        
        try:
            # 检查是否已经初始化
            if not mixer.get_init():
                mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            
            self.initialized = True
            self._generate_sounds()
            print("声音系统初始化成功")
            
        except Exception as e:
            print(f"声音系统初始化失败: {e}")
            self.initialized = False
    
    def _generate_sounds(self):
        """生成所有音效"""
        if not self.initialized:
            return
        
        try:
            generator = SoundGenerator()
            
            # 生成各种音效
            sound_data = {
                'success': generator.generate_success_sound(),
                'error': generator.generate_error_sound(),
                'complete': generator.generate_complete_sound(),
                'start': generator.generate_start_sound(),
                'beep': generator.generate_beep(440, 0.1),
                'cue': generator.generate_beep(880, 0.05),
            }
            
            # 转换为pygame Sound对象
            for name, data in sound_data.items():
                sound = mixer.Sound(buffer=data)
                sound.set_volume(self.volume)
                self.sounds[name] = sound
            
        except Exception as e:
            print(f"生成音效失败: {e}")
    
    def play(self, sound_name):
        """
        播放指定音效
        
        参数:
            sound_name: 'success', 'error', 'complete', 'start', 'beep', 'cue'
        """
        if not self.enabled or not self.initialized:
            return
        
        if sound_name in self.sounds:
            try:
                self.sounds[sound_name].play()
            except Exception as e:
                print(f"播放音效失败: {e}")
    
    def play_success(self):
        """播放成功提示音"""
        self.play('success')
    
    def play_error(self):
        """播放错误提示音"""
        self.play('error')
    
    def play_complete(self):
        """播放完成提示音"""
        self.play('complete')
    
    def play_start(self):
        """播放开始提示音"""
        self.play('start')
    
    def play_beep(self):
        """播放蜂鸣音"""
        self.play('beep')
    
    def play_cue(self):
        """播放提示音（Cue阶段）"""
        self.play('cue')
    
    def set_enabled(self, enabled):
        """设置是否启用"""
        self.enabled = enabled
    
    def set_volume(self, volume):
        """设置音量 (0-1)"""
        self.volume = max(0, min(1, volume))
        
        if self.initialized:
            for sound in self.sounds.values():
                sound.set_volume(self.volume)
    
    def toggle(self):
        """切换开关状态"""
        self.enabled = not self.enabled
        return self.enabled


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print("声音反馈模块测试")
    print("=" * 50)
    
    # 需要先初始化pygame
    import pygame
    pygame.init()
    
    # 创建声音反馈
    sound = SoundFeedback(enabled=True, volume=0.7)
    
    if sound.initialized:
        print("\n测试各种音效...")
        
        print("  播放开始音...")
        sound.play_start()
        time.sleep(0.6)
        
        print("  播放提示音...")
        sound.play_cue()
        time.sleep(0.3)
        
        print("  播放成功音...")
        sound.play_success()
        time.sleep(0.3)
        
        print("  播放错误音...")
        sound.play_error()
        time.sleep(0.4)
        
        print("  播放完成音...")
        sound.play_complete()
        time.sleep(0.7)
        
        print("\n测试完成！")
    else:
        print("声音系统未能初始化")
    
    pygame.quit()

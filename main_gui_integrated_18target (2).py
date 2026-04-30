
"""
Real-time SSVEP-BCI System - Fixed Version
Fixes: 1) Frequency consistency 2) SSVEP param editor 3) Dropdown z-order 4) Settings layout
Updates: GDS device support, EEG signal monitor page
"""

import pygame
import numpy as np
import math
import time
import threading
import os
from dataclasses import dataclass

from eeg_device_interface import EEGDeviceManager, SimulatedEEG
from realtime_ssvep_recognition import RealTimeSSVEPRecognizer, SSVEPRecognitionThread
from data_saver import DataSaver
from sound_feedback import SoundFeedback

# Default 18-target subset used for the offline CCA/TRCA evaluation.
# The GUI still contains the full 40-target keyboard, but these targets are
# repeated across calibration/test blocks for the project-specific experiment.
EVAL_TARGET_SEQUENCE = "13579qetuoadgjlzcb"  # 18 targets spread across the keyboard/frequency range


# ============================================
# EEG 信号监测器 (pygame 内绘制)
# ============================================

class EEGMonitor:
    """
    实时EEG侧边栏监测器 (pygame绘制)
    ─────────────────────────────────
    • 侧边栏宽度固定 SIDEBAR_W px，紧贴屏幕右边
    • 每通道一行波形，自动增益，彩色区分
    • 每行右侧显示峰峰值
    • 支持 toggle 展开/折叠
    • draw_sidebar() 同时绘制折叠按钮（tab），外部直接调用
    """

    SIDEBAR_W   = 220          # 展开时侧边栏宽度
    TAB_W       = 22           # 折叠时仅显示的 tab 宽度
    CH_COLORS   = [
        (66, 135, 245), (46, 204, 113), (231, 76, 60),
        (255, 165,   0), (155,  89, 182), (0, 188, 212),
        (240, 200,  70), (255, 130, 180), (130, 230, 180),
    ]

    def __init__(self, display_seconds=5):
        self.display_seconds = display_seconds
        self.visible   = False     # 侧边栏展开状态
        self.paused    = False

        self._smooth_ranges = None
        self._snapshot      = None
        self._lock          = threading.Lock()

        # 延迟初始化字体
        self._f_title = None
        self._f_ch    = None
        self._f_val   = None

        # 按钮 rect（由 draw_sidebar 每帧更新，供点击检测）
        self.toggle_btn_rect = None

    # ──────────────────────────────────────
    # 数据更新
    # ──────────────────────────────────────
    def update(self, eeg_device):
        """每帧从设备缓冲区取最新快照"""
        if self.paused or eeg_device is None:
            return
        buf = eeg_device.buffer
        if len(buf) < 2:
            return
        need = int(self.display_seconds * eeg_device.fs)
        data = list(buf)
        arr  = np.array(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_ch = eeg_device.n_channels
        if self._smooth_ranges is None or len(self._smooth_ranges) != n_ch:
            self._smooth_ranges = np.ones(n_ch)
        if arr.shape[0] < need:
            pad = np.zeros((need - arr.shape[0], arr.shape[1]), dtype=float)
            arr = np.vstack([pad, arr])
        else:
            arr = arr[-need:]
        with self._lock:
            self._snapshot = arr.copy()

    # ──────────────────────────────────────
    # 绘制侧边栏（含折叠 tab）
    # ──────────────────────────────────────
    def draw_sidebar(self, surface, screen_w, screen_h, n_channels, fs):
        """
        绘制整个侧边栏区域（含 toggle tab）。
        必须在每帧 keyboard 之后调用，覆盖在最上层。

        返回: keyboard 可用的右边界 x 坐标
        """
        # ── 初始化字体 ──
        if self._f_title is None:
            self._f_title = pygame.font.Font(None, 18)
            self._f_ch    = pygame.font.Font(None, 17)
            self._f_val   = pygame.font.Font(None, 16)

        sb_w = self.SIDEBAR_W if self.visible else 0
        tab_x = screen_w - sb_w - self.TAB_W

        # ── Toggle tab ──
        tab_rect = pygame.Rect(tab_x, screen_h // 2 - 48, self.TAB_W, 96)
        self.toggle_btn_rect = tab_rect
        pygame.draw.rect(surface, (40, 45, 62), tab_rect, border_radius=6)
        pygame.draw.rect(surface, (70, 75, 95), tab_rect, 1, border_radius=6)

        # 旋转箭头字符
        arrow = "◀" if self.visible else "▶"
        # 垂直排列 "EEG" 文字
        for idx, ch in enumerate(["E", "E", "G"]):
            cs = self._f_ch.render(ch, True, (140, 150, 170))
            surface.blit(cs, cs.get_rect(center=(tab_rect.centerx, tab_rect.y + 16 + idx * 14)))
        ars = self._f_ch.render(arrow, True, (100, 188, 212))
        surface.blit(ars, ars.get_rect(center=(tab_rect.centerx, tab_rect.bottom - 14)))

        if not self.visible:
            return screen_w - self.TAB_W   # keyboard 可用右边界

        # ── 侧边栏背景 ──
        sb_rect = pygame.Rect(screen_w - sb_w, 0, sb_w, screen_h)
        pygame.draw.rect(surface, (16, 19, 28), sb_rect)
        pygame.draw.line(surface, (50, 55, 72),
                         (sb_rect.x, 0), (sb_rect.x, screen_h), 1)

        # ── 标题栏 ──
        title_bar = pygame.Rect(sb_rect.x, 0, sb_w, 38)
        pygame.draw.rect(surface, (22, 26, 38), title_bar)
        pygame.draw.line(surface, (50, 55, 72),
                         (sb_rect.x, 38), (sb_rect.right, 38), 1)
        ts = self._f_title.render("EEG  MONITOR", True, (0, 188, 212))
        surface.blit(ts, ts.get_rect(center=(sb_rect.centerx, 19)))

        # 连接状态小圆点
        dot_color = (46, 204, 113) if self._snapshot is not None else (100, 100, 110)
        pygame.draw.circle(surface, dot_color, (sb_rect.x + 10, 19), 4)

        # ── 波形区 ──
        wave_top  = 42
        wave_bot  = screen_h - 20
        wave_h    = wave_bot - wave_top
        inner_x   = sb_rect.x + 4
        inner_w   = sb_w - 8

        with self._lock:
            snap = self._snapshot

        if snap is None or snap.shape[0] < 2:
            wt = self._f_ch.render("等待信号...", True, (80, 90, 108))
            surface.blit(wt, wt.get_rect(center=(sb_rect.centerx, screen_h // 2)))
            return screen_w - sb_w - self.TAB_W

        n_ch    = min(n_channels, snap.shape[1])
        row_h   = wave_h / n_ch
        lbl_w   = 32
        val_w   = 38
        plot_x  = inner_x + lbl_w
        plot_w  = inner_w - lbl_w - val_w - 2
        need    = int(self.display_seconds * fs)
        n_pts   = min(need, snap.shape[0])
        t_axis  = np.linspace(0, plot_w - 1, n_pts, dtype=int)

        for i in range(n_ch):
            row_top = wave_top + i * row_h
            cy      = row_top + row_h / 2
            color   = self.CH_COLORS[i % len(self.CH_COLORS)]

            # 行背景（交替微深色）
            if i % 2 == 0:
                row_bg = pygame.Rect(inner_x, int(row_top), inner_w, int(row_h))
                pygame.draw.rect(surface, (19, 22, 32), row_bg)

            # 分隔线
            pygame.draw.line(surface, (35, 39, 52),
                             (inner_x, int(row_top + row_h - 1)),
                             (inner_x + inner_w, int(row_top + row_h - 1)), 1)

            # 零线
            pygame.draw.line(surface, (38, 42, 56),
                             (plot_x, int(cy)), (plot_x + plot_w, int(cy)), 1)

            # 通道标签
            lbl_s = self._f_ch.render(f"C{i+1}", True, color)
            surface.blit(lbl_s, (inner_x + 2, int(cy) - 7))

            # 数据处理
            ch_raw  = snap[-n_pts:, i]
            ch_data = ch_raw - np.mean(ch_raw)

            peak = float(np.max(np.abs(ch_data))) if np.any(ch_data != 0) else 1.0
            if peak < 1e-9:
                peak = 1.0
            sr = self._smooth_ranges[i]
            if peak > sr:
                self._smooth_ranges[i] = peak * 1.1
            else:
                self._smooth_ranges[i] = sr * 0.99 + peak * 0.01 * 1.1
            amp = self._smooth_ranges[i]

            # 波形插值 & 绘制
            ch_r = np.interp(
                np.linspace(0, len(ch_data) - 1, len(t_axis)),
                np.arange(len(ch_data)), ch_data
            )
            half_h = row_h * 0.42
            ys = np.clip((cy - ch_r / amp * half_h).astype(int),
                         int(row_top + 1), int(row_top + row_h - 2))

            if len(t_axis) > 1:
                pts = list(zip((t_axis + plot_x).tolist(), ys.tolist()))
                pygame.draw.lines(surface, color, False, pts, 1)

            # 峰峰值
            pp = float(np.max(snap[-n_pts:, i]) - np.min(snap[-n_pts:, i]))
            if pp > 1000:
                v_str = f"{pp/1000:.1f}m"
            elif pp < 0.01:
                v_str = "~0"
            else:
                v_str = f"{pp:.0f}u"
            vs = self._f_val.render(v_str, True, (90, 98, 115))
            surface.blit(vs, (plot_x + plot_w + 2, int(cy) - 6))

        # ── 底部时间标尺 ──
        ruler_y = wave_bot - 12
        pygame.draw.line(surface, (45, 50, 65),
                         (plot_x, ruler_y), (plot_x + plot_w, ruler_y), 1)
        for s in [0, self.display_seconds // 2, self.display_seconds]:
            tx = plot_x + int(s / self.display_seconds * plot_w)
            pygame.draw.line(surface, (55, 60, 78), (tx, ruler_y - 3), (tx, ruler_y + 3), 1)
            tl = self._f_val.render(f"{s}s", True, (70, 78, 95))
            surface.blit(tl, (tx - 7, ruler_y + 4))

        return screen_w - sb_w - self.TAB_W   # keyboard 可用右边界



class Theme:
    BG_PRIMARY = (15, 17, 23)
    BG_SECONDARY = (25, 28, 38)
    BG_CARD = (35, 40, 55)
    ACCENT_BLUE = (66, 135, 245)
    ACCENT_GREEN = (46, 204, 113)
    ACCENT_RED = (231, 76, 60)
    ACCENT_ORANGE = (255, 165, 0)
    ACCENT_PURPLE = (155, 89, 182)
    ACCENT_CYAN = (0, 188, 212)
    TEXT_PRIMARY = (236, 240, 245)
    TEXT_SECONDARY = (160, 170, 185)
    TEXT_MUTED = (100, 110, 125)
    BORDER = (60, 65, 80)
    BORDER_LIGHT = (80, 88, 105)


@dataclass
class BCISettings:
    cue_duration: float = 2.0
    flickering_duration: float = 4.5
    pause_duration: float = 1.0
    rest_duration: float = 30.0
    # ----- 18-target offline-evaluation protocol -----
    # The GUI shows the full 40-target keyboard, but only the
    # following 18 characters are presented as cued targets in
    # each block. Calibration: blocks 0..calibration_blocks-1;
    # held-out test: block calibration_blocks (typically index 5).
    total_blocks: int = 6
    calibration_blocks: int = 5
    test_blocks: int = 1
    eval_target_sequence: str = "13579qetuoadgjlzcb"
    itr_n_targets: int = 18  # used only for in-GUI ITR display;
                             # final paper ITR is computed by offline_eval.py
    # -------------------------------------------------
    block_texts: list = None
    subject_id: str = "S001"
    save_data: bool = True
    save_dir: str = "experiment_data"
    device_type: str = 'simulated'
    fs: int = 256
    n_channels: int = 9
    recognition_method: str = 'cca'
    window_seconds: float = 3.0
    confidence_threshold: float = 0.1
    itr_time_mode: int = 1
    sound_enabled: bool = True
    sound_volume: float = 0.7

    def __post_init__(self):
        if self.block_texts is None:
            # 5 calibration + 1 held-out test, all using the same 18-target sequence
            self.block_texts = [self.eval_target_sequence
                                for _ in range(self.calibration_blocks + self.test_blocks)]
        self.total_blocks = len(self.block_texts)
        # ITR target count = number of unique characters in the eval subset
        self.itr_n_targets = len(set(self.eval_target_sequence))

    def get_itr_time(self):
        if self.itr_time_mode == 0: return self.flickering_duration
        elif self.itr_time_mode == 1: return self.cue_duration + self.flickering_duration
        else: return self.cue_duration + self.flickering_duration + self.pause_duration


class RealTimeBCISystem:

    def __init__(self):
        pygame.init()
        self.settings = BCISettings()
        self.keyboard_layout = [
            ['1','2','3','4','5','6','7','8','9','0','<'],
            ['q','w','e','r','t','y','u','i','o','p'],
            ['a','s','d','f','g','h','j','k','l'],
            ['z','x','c','v','b','n','m','.'],
            ['_', ',']
        ]
        self.frequencies = []
        self.char_to_freq = {}
        self.freq_to_char = {}
        self._assign_frequencies()
        self.beta_params = self._generate_beta_params()
        self._init_display()

        self.eeg_device = None
        self.recognizer = None
        self.recognition_thread = None
        self.experiment_state = None
        self.experiment_paused = False
        self.block_history = {'results': [], 'accuracies': [], 'itrs': []}
        self.current_recognition = None
        self.recognition_lock = threading.Lock()
        self.data_saver = DataSaver(save_dir=self.settings.save_dir)
        self.saved_files = []
        self.subject_id_input_active = False
        self.subject_id_text = self.settings.subject_id
        self.sound = SoundFeedback(enabled=self.settings.sound_enabled, volume=self.settings.sound_volume)
        self.active_input_field = None
        self.block_texts_input = ', '.join(self.settings.block_texts)

        # EEG 信号监测器
        self.eeg_monitor = EEGMonitor(display_seconds=4)

        # SSVEP参数编辑器
        self.ssvep_selected_key = None
        self.ssvep_editing_field = None  # 'frequency' or 'phase'
        self.ssvep_edit_text = ""
        self.ssvep_key_rects = {}

        # 闪烁计时
        self.flicker_start_time = None

    def _assign_frequencies(self):
        base_freq = 8.0
        freq_step = 0.2
        all_chars = [c for row in self.keyboard_layout for c in row]
        for i, char in enumerate(all_chars):
            freq = round(base_freq + i * freq_step, 1)
            self.frequencies.append(freq)
            self.char_to_freq[char] = freq
            self.freq_to_char[freq] = char
        self.frequencies = sorted(set(self.frequencies))

    def _generate_beta_params(self):
        params = {}
        all_chars = [c for row in self.keyboard_layout for c in row]
        for i, char in enumerate(all_chars):
            freq = self.char_to_freq[char]
            phase = round((i % 8) * (2 * math.pi / 8), 4)
            params[char] = {'frequency': freq, 'phase': phase}
        return params

    def _init_display(self):
        info = pygame.display.Info()
        self.screen_width = min(1400, info.current_w - 100)
        self.screen_height = min(900, info.current_h - 100)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Real-time SSVEP-BCI System")
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.font_title = pygame.font.Font(None, 56)
        self.font_header = pygame.font.Font(None, 42)
        self.font_body = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 26)
        self.font_tiny = pygame.font.Font(None, 20)
        self.font_key = pygame.font.Font(None, 36)
        self.font_key_sm = pygame.font.Font(None, 18)
        self.current_page = "main"
        self.device_dropdown_open = False
        self.method_dropdown_open = False

    # === Device ===
    def connect_device(self):
        try:
            kwargs = dict(fs=self.settings.fs, n_channels=self.settings.n_channels)
            if self.settings.device_type == 'simulated':
                kwargs['frequencies'] = self.frequencies
            elif self.settings.device_type == 'gds':
                # GDS 设备使用回调模式，n_scans 固定为8
                kwargs['n_scans'] = 8
            self.eeg_device = EEGDeviceManager.create_device(self.settings.device_type, **kwargs)
            if self.eeg_device.connect():
                self.recognizer = RealTimeSSVEPRecognizer(
                    frequencies=self.frequencies,
                    fs=self.settings.fs,
                    window_seconds=self.settings.window_seconds,
                    method=self.settings.recognition_method,
                    confidence_threshold=self.settings.confidence_threshold
                )
                return True
            return False
        except Exception as e:
            print(f"连接失败: {e}"); return False

    def disconnect_device(self):
        if self.recognition_thread: self.recognition_thread.stop(); self.recognition_thread = None
        if self.eeg_device: self.eeg_device.disconnect(); self.eeg_device = None
        self.recognizer = None

    def start_recognition(self):
        if not self.eeg_device or not self.recognizer: return
        self.eeg_device.start_streaming()
        self.recognition_thread = SSVEPRecognitionThread(self.eeg_device, self.recognizer, recognition_interval=0.5)
        self.recognition_thread.on_result_callback = self._on_recognition_result
        self.recognition_thread.start()

    def stop_recognition(self):
        if self.recognition_thread: self.recognition_thread.stop(); self.recognition_thread = None
        if self.eeg_device: self.eeg_device.stop_streaming()

    def _on_recognition_result(self, result):
        with self.recognition_lock: self.current_recognition = result

    def get_recognized_char(self):
        with self.recognition_lock:
            if self.current_recognition and self.current_recognition['is_valid']:
                freq = self.current_recognition['frequency']
                best = min(self.char_to_freq.items(), key=lambda x: abs(x[1] - freq))
                return best[0]
        return None

    def calculate_itr(self, accuracy, time_per_char=None):
        if time_per_char is None: time_per_char = self.settings.get_itr_time()
        n = self.settings.itr_n_targets if getattr(self.settings, "itr_n_targets", 0) else len(self.char_to_freq)
        if accuracy <= 0 or accuracy > 1 or time_per_char <= 0: return 0.0
        if accuracy == 1.0: bits = np.log2(n)
        else:
            try: bits = np.log2(n) + accuracy * np.log2(accuracy) + (1-accuracy) * np.log2((1-accuracy)/(n-1))
            except: bits = 0
        return max(0, bits * (60.0 / time_per_char))

    # === Drawing utils ===
    def draw_rounded_rect(self, color, rect, radius, border=0, border_color=None):
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)
        if border > 0 and border_color:
            pygame.draw.rect(self.screen, border_color, rect, border, border_radius=radius)

    def draw_button(self, text, rect, color=None, hover=False, font=None):
        c = color or Theme.ACCENT_BLUE
        if hover: c = tuple(min(255, int(v*1.2)) for v in c)
        self.draw_rounded_rect(c, rect, 10)
        f = font or self.font_body
        s = f.render(text, True, Theme.TEXT_PRIMARY)
        self.screen.blit(s, s.get_rect(center=rect.center))

    def draw_card(self, rect, title=None):
        self.draw_rounded_rect(Theme.BG_CARD, rect, 15, 1, Theme.BORDER)
        if title:
            self.screen.blit(self.font_header.render(title, True, Theme.TEXT_PRIMARY), (rect.x+20, rect.y+15))
            pygame.draw.line(self.screen, Theme.BORDER, (rect.x+15, rect.y+55), (rect.x+rect.width-15, rect.y+55), 1)
            return rect.y + 70
        return rect.y + 15

    def draw_progress_bar(self, rect, progress, color=None):
        c = color or Theme.ACCENT_BLUE
        self.draw_rounded_rect(Theme.BG_SECONDARY, rect, 5)
        if progress > 0:
            self.draw_rounded_rect(c, pygame.Rect(rect.x, rect.y, int(rect.width*min(1,progress)), rect.height), 5)

    def _draw_dropdown(self, rect, value, is_open, options):
        mp = pygame.mouse.get_pos()
        bg = (45,50,70) if rect.collidepoint(mp) or is_open else Theme.BG_SECONDARY
        bc = Theme.ACCENT_BLUE if is_open else Theme.BORDER_LIGHT
        self.draw_rounded_rect(bg, rect, 8, 2, bc)
        self.screen.blit(self.font_body.render(value.upper(), True, Theme.TEXT_PRIMARY), (rect.x+15, rect.y+8))
        self.screen.blit(self.font_small.render("^" if is_open else "v", True, Theme.TEXT_SECONDARY), (rect.right-30, rect.y+10))
        if is_open:
            mh = len(options)*42+10
            mr = pygame.Rect(rect.x, rect.bottom+2, rect.width, mh)
            self.draw_rounded_rect((5,5,10), pygame.Rect(mr.x+3, mr.y+3, mr.width, mr.height), 8)
            self.draw_rounded_rect(Theme.BG_CARD, mr, 8, 2, Theme.ACCENT_BLUE)
            for i, opt in enumerate(options):
                orect = pygame.Rect(rect.x+5, rect.bottom+7+i*42, rect.width-10, 36)
                if orect.collidepoint(mp): self.draw_rounded_rect((55,60,80), orect, 6)
                self.screen.blit(self.font_body.render(opt.upper(), True, Theme.TEXT_PRIMARY), (orect.x+10, orect.y+6))

    def _draw_setting_slider(self, x, y, width, label, attr, value, min_val, max_val, step):
        mp = pygame.mouse.get_pos()
        # Layout: [label 180px] [- 30px] [slider flex] [+ 30px] [value 70px]
        label_w = 180
        btn_size = 28
        value_w = 70
        gap = 8

        self.screen.blit(self.font_small.render(f"{label}:", True, Theme.TEXT_SECONDARY), (x, y+3))

        # Right side positions (from right edge)
        value_x = x + width - value_w
        plus_x = value_x - btn_size - gap
        slider_end = plus_x - gap
        minus_x = x + label_w
        slider_x = minus_x + btn_size + gap
        slider_w = max(40, slider_end - slider_x)

        # - button
        mr = pygame.Rect(minus_x, y, btn_size, btn_size)
        self.draw_rounded_rect(Theme.BG_CARD, mr, 6)
        ms = self.font_body.render("-", True, Theme.TEXT_PRIMARY)
        self.screen.blit(ms, ms.get_rect(center=mr.center))

        # slider track
        sr = pygame.Rect(slider_x, y+8, slider_w, 12)
        self.draw_rounded_rect(Theme.BG_SECONDARY, sr, 6)
        p = max(0, min(1, (value-min_val)/(max_val-min_val) if max_val>min_val else 0))
        fw = int(slider_w * p)
        if fw > 0: self.draw_rounded_rect(Theme.ACCENT_BLUE, pygame.Rect(slider_x, y+8, fw, 12), 6)
        hr = pygame.Rect(slider_x+fw-8, y+4, 16, 20)
        self.draw_rounded_rect(Theme.ACCENT_CYAN if hr.collidepoint(mp) else Theme.TEXT_PRIMARY, hr, 4)

        # + button
        pr = pygame.Rect(plus_x, y, btn_size, btn_size)
        self.draw_rounded_rect(Theme.BG_CARD, pr, 6)
        ps = self.font_body.render("+", True, Theme.TEXT_PRIMARY)
        self.screen.blit(ps, ps.get_rect(center=pr.center))

        # value display
        vt = f"{value:.2f}" if isinstance(value,float) and step<1 else (f"{value:.1f}" if isinstance(value,float) else str(int(value)))
        vs = self.font_body.render(vt, True, Theme.ACCENT_CYAN)
        self.screen.blit(vs, (value_x, y+1))

        if not hasattr(self, 'slider_rects'): self.slider_rects = {}
        self.slider_rects[attr] = {'rect': sr, 'min': min_val, 'max': max_val, 'step': step}
        if not hasattr(self, 'setting_buttons'): self.setting_buttons = {}
        self.setting_buttons[attr] = {'minus': mr, 'plus': pr, 'min': min_val, 'max': max_val, 'step': step}

    # ==========================================
    # MAIN PAGE
    # ==========================================
    def draw_main_page(self):
        self.screen.fill(Theme.BG_PRIMARY)
        mp = pygame.mouse.get_pos()
        self.screen.blit(self.font_title.render("Real-time SSVEP-BCI System", True, Theme.TEXT_PRIMARY),
                        self.font_title.render("Real-time SSVEP-BCI System", True, Theme.TEXT_PRIMARY).get_rect(center=(self.screen_width//2, 40)))
        conn = self.eeg_device and self.eeg_device.is_connected
        self.screen.blit(self.font_body.render(f"Device: {'Connected' if conn else 'Disconnected'}", True,
                        Theme.ACCENT_GREEN if conn else Theme.ACCENT_RED), (self.screen_width-280, 15))

        top, col_w = 80, (self.screen_width-120)//2
        lx, rx = 40, 40+col_w+40
        card_h = min(350, self.screen_height-80-70-30)

        # Left card
        lc = pygame.Rect(lx, top, col_w, card_h)
        cy = self.draw_card(lc, "Device Settings")
        self.screen.blit(self.font_small.render("Device Type:", True, Theme.TEXT_SECONDARY), (lc.x+25, cy+5))
        self.device_dropdown_rect = pygame.Rect(lc.x+25, cy+28, col_w-50, 40)
        my = cy+85
        self.screen.blit(self.font_small.render("Recognition Method:", True, Theme.TEXT_SECONDARY), (lc.x+25, my))
        self.method_dropdown_rect = pygame.Rect(lc.x+25, my+23, col_w-50, 40)
        bty = my+95
        bw = min(140, (col_w-80)//2)
        self.connect_btn = pygame.Rect(lc.x+25, bty, bw, 42)
        self.disconnect_btn = pygame.Rect(lc.x+25+bw+15, bty, bw, 42)
        self.draw_button("Connect", self.connect_btn, Theme.ACCENT_GREEN, self.connect_btn.collidepoint(mp))
        self.draw_button("Disconnect", self.disconnect_btn, Theme.ACCENT_RED, self.disconnect_btn.collidepoint(mp))
        # Dropdowns LAST (on top)
        self._draw_dropdown(self.device_dropdown_rect, self.settings.device_type, self.device_dropdown_open, ['simulated','lsl','openbci','gds'])
        self._draw_dropdown(self.method_dropdown_rect, self.settings.recognition_method, self.method_dropdown_open, ['cca','trca'])

        # Right card
        rc = pygame.Rect(rx, top, col_w, card_h)
        cy = self.draw_card(rc, "Experiment Settings")
        self.screen.blit(self.font_small.render("Subject ID:", True, Theme.TEXT_SECONDARY), (rc.x+25, cy+2))
        self.subject_input_rect = pygame.Rect(rc.x+25, cy+22, 180, 36)
        bc = Theme.ACCENT_BLUE if self.subject_id_input_active else Theme.BORDER_LIGHT
        self.draw_rounded_rect(Theme.BG_SECONDARY, self.subject_input_rect, 8, 2, bc)
        self.screen.blit(self.font_body.render(self.subject_id_text+("|" if self.subject_id_input_active else ""), True, Theme.TEXT_PRIMARY),
                        (self.subject_input_rect.x+10, self.subject_input_rect.y+6))
        self.screen.blit(self.font_small.render("Save:", True, Theme.TEXT_SECONDARY), (rc.x+230, cy+2))
        self.save_toggle_rect = pygame.Rect(rc.x+230, cy+22, 70, 36)
        tc = Theme.ACCENT_GREEN if self.settings.save_data else Theme.BG_SECONDARY
        self.draw_rounded_rect(tc, self.save_toggle_rect, 8, 2, Theme.BORDER_LIGHT)
        tt = self.font_body.render("ON" if self.settings.save_data else "OFF", True, Theme.TEXT_PRIMARY)
        self.screen.blit(tt, tt.get_rect(center=self.save_toggle_rect.center))
        iy = cy+68
        for ln in [f"Fs: {self.settings.fs}Hz | Ch: {self.settings.n_channels} | {self.settings.recognition_method.upper()}",
                    f"Targets: {len(self.frequencies)} GUI | Eval N={self.settings.itr_n_targets} ({min(self.frequencies):.1f}-{max(self.frequencies):.1f} Hz)",
                    f"Cue: {self.settings.cue_duration}s | Flash: {self.settings.flickering_duration}s | Pause: {self.settings.pause_duration}s"]:
            self.screen.blit(self.font_small.render(ln, True, Theme.TEXT_PRIMARY), (rc.x+25, iy)); iy += 26

        # Bottom buttons
        by = self.screen_height-65
        bw2, bh, bs = 120, 48, 12
        tw = bw2*6+bs*5
        sx = (self.screen_width-tw)//2
        self.settings_btn     = pygame.Rect(sx, by, bw2, bh)
        self.ssvep_params_btn = pygame.Rect(sx+bw2+bs, by, bw2, bh)
        self.monitor_btn      = pygame.Rect(sx+(bw2+bs)*2, by, bw2, bh)
        self.demo_btn         = pygame.Rect(sx+(bw2+bs)*3, by, bw2, bh)
        self.train_btn        = pygame.Rect(sx+(bw2+bs)*4, by, bw2, bh)
        self.start_btn        = pygame.Rect(sx+(bw2+bs)*5, by, bw2, bh)
        self.draw_button("Settings",     self.settings_btn,     Theme.ACCENT_PURPLE, self.settings_btn.collidepoint(mp))
        self.draw_button("SSVEP Params", self.ssvep_params_btn, Theme.ACCENT_CYAN,   self.ssvep_params_btn.collidepoint(mp), self.font_small)
        self.draw_button("Monitor",      self.monitor_btn,      Theme.ACCENT_GREEN if self.eeg_monitor.visible else Theme.ACCENT_BLUE, self.monitor_btn.collidepoint(mp))
        self.draw_button("Demo",         self.demo_btn,         Theme.BG_CARD,       self.demo_btn.collidepoint(mp))
        self.draw_button("Training",     self.train_btn,        Theme.ACCENT_ORANGE, self.train_btn.collidepoint(mp))
        self.draw_button("Start",        self.start_btn,        Theme.ACCENT_GREEN,  self.start_btn.collidepoint(mp))

    # ==========================================
    # SETTINGS PAGE (two-column, full screen)
    # ==========================================
    def draw_settings_page(self):
        self.screen.fill(Theme.BG_PRIMARY)
        mp = pygame.mouse.get_pos()
        self.screen.blit(self.font_title.render("Settings", True, Theme.TEXT_PRIMARY),
                        self.font_title.render("Settings", True, Theme.TEXT_PRIMARY).get_rect(center=(self.screen_width//2, 35)))
        self.back_btn = pygame.Rect(30, 20, 90, 38)
        self.draw_button("<< Back", self.back_btn, Theme.BG_CARD, self.back_btn.collidepoint(mp), self.font_small)

        m, gap = 30, 25
        cw = (self.screen_width-m*2-gap)//2
        lx, rx = m, m+cw+gap
        ty = 75

        # Left: Time
        tc = pygame.Rect(lx, ty, cw, 230)
        cy = self.draw_card(tc, "Time Settings")
        for i, (lb, at, vl, mn, mx, st) in enumerate([
            ("Cue Duration (s)", "cue_duration", self.settings.cue_duration, 0.5, 5.0, 0.5),
            ("Flash Duration (s)", "flickering_duration", self.settings.flickering_duration, 1.0, 10.0, 0.5),
            ("Pause Duration (s)", "pause_duration", self.settings.pause_duration, 0.1, 2.0, 0.1),
            ("Rest Duration (s)", "rest_duration", self.settings.rest_duration, 5.0, 60.0, 5.0)]):
            self._draw_setting_slider(lx+20, cy+5+i*38, cw-40, lb, at, vl, mn, mx, st)

        # Left: Experiment - Block Manager
        ey = ty+250
        avail_h = self.screen_height - ey - 80  # space until bottom buttons
        ec_h = max(200, avail_h)
        ec = pygame.Rect(lx, ey, cw, ec_h)
        cy = self.draw_card(ec, "Block Manager")

        # Add Block button
        self.add_block_btn = pygame.Rect(ec.right - 130, ec.y+12, 110, 35)
        self.draw_button("+ Add Block", self.add_block_btn, Theme.ACCENT_GREEN, self.add_block_btn.collidepoint(mp), self.font_small)

        # Block list
        block_x = lx + 20
        block_w = cw - 40
        block_item_h = 42
        block_gap = 6

        # Initialize block editing state if needed
        if not hasattr(self, 'editing_block_idx'):
            self.editing_block_idx = None
            self.editing_block_text = ""
        if not hasattr(self, 'block_item_rects'):
            self.block_item_rects = {}
            self.block_delete_btns = {}
            self.block_edit_rects = {}

        self.block_item_rects.clear()
        self.block_delete_btns.clear()
        self.block_edit_rects.clear()

        blocks = self.settings.block_texts
        total_blocks_h = len(blocks) * (block_item_h + block_gap)

        # Scrollable area
        clip_top = cy + 5
        clip_bottom = ec.bottom - 15
        clip_h = clip_bottom - clip_top

        # Clamp scroll
        if not hasattr(self, 'block_scroll'):
            self.block_scroll = 0
        max_scroll = max(0, total_blocks_h - clip_h)
        self.block_scroll = max(0, min(self.block_scroll, max_scroll))

        # Draw blocks
        for i, text in enumerate(blocks):
            item_y = clip_top + i * (block_item_h + block_gap) - self.block_scroll

            # Skip if outside visible area
            if item_y + block_item_h < clip_top or item_y > clip_bottom:
                continue

            item_rect = pygame.Rect(block_x, item_y, block_w, block_item_h)
            self.block_item_rects[i] = item_rect

            is_editing = (self.editing_block_idx == i)
            is_hover = item_rect.collidepoint(mp)

            # Background
            bg = (45, 50, 70) if is_hover or is_editing else Theme.BG_SECONDARY
            bc = Theme.ACCENT_BLUE if is_editing else (Theme.BORDER_LIGHT if is_hover else Theme.BORDER)
            self.draw_rounded_rect(bg, item_rect, 8, 1, bc)

            # Block number badge
            badge_r = pygame.Rect(item_rect.x + 8, item_rect.y + 6, 32, 30)
            self.draw_rounded_rect(Theme.ACCENT_BLUE, badge_r, 6)
            bn = self.font_body.render(str(i+1), True, Theme.TEXT_PRIMARY)
            self.screen.blit(bn, bn.get_rect(center=badge_r.center))

            # Text content
            text_x = item_rect.x + 50
            text_w = item_rect.width - 95

            if is_editing:
                # Editable text field
                edit_rect = pygame.Rect(text_x, item_rect.y + 5, text_w, 32)
                self.block_edit_rects[i] = edit_rect
                self.draw_rounded_rect(Theme.BG_PRIMARY, edit_rect, 6, 1, Theme.ACCENT_BLUE)
                et = self.editing_block_text + "|"
                self.screen.blit(self.font_body.render(et[:40], True, Theme.TEXT_PRIMARY), (edit_rect.x+8, edit_rect.y+4))
            else:
                display = text if text else "(empty)"
                tc = Theme.TEXT_PRIMARY if text else Theme.TEXT_MUTED
                self.screen.blit(self.font_body.render(display[:35], True, tc), (text_x + 5, item_rect.y + 10))

            # Delete button (X)
            del_r = pygame.Rect(item_rect.right - 38, item_rect.y + 7, 28, 28)
            self.block_delete_btns[i] = del_r
            del_hover = del_r.collidepoint(mp)
            del_bg = Theme.ACCENT_RED if del_hover else (60, 40, 40)
            self.draw_rounded_rect(del_bg, del_r, 6)
            xs = self.font_small.render("X", True, Theme.TEXT_PRIMARY)
            self.screen.blit(xs, xs.get_rect(center=del_r.center))

        # Show count summary
        count_text = f"{len(blocks)} block{'s' if len(blocks) != 1 else ''}"
        self.screen.blit(self.font_small.render(count_text, True, Theme.TEXT_MUTED),
                        (block_x, ec.bottom - 20))

        # Right: Sound
        sc = pygame.Rect(rx, ty, cw, 165)
        cy = self.draw_card(sc, "Sound Settings")
        self.screen.blit(self.font_small.render("Sound Feedback:", True, Theme.TEXT_SECONDARY), (rx+20, cy+8))
        self.sound_toggle_rect = pygame.Rect(rx+200, cy+3, 80, 35)
        self.draw_rounded_rect(Theme.ACCENT_GREEN if self.settings.sound_enabled else Theme.BG_SECONDARY, self.sound_toggle_rect, 8, 2, Theme.BORDER_LIGHT)
        sts = self.font_body.render("ON" if self.settings.sound_enabled else "OFF", True, Theme.TEXT_PRIMARY)
        self.screen.blit(sts, sts.get_rect(center=self.sound_toggle_rect.center))
        self.test_sound_btn = pygame.Rect(rx+300, cy+3, 90, 35)
        self.draw_button("Test", self.test_sound_btn, Theme.ACCENT_BLUE, self.test_sound_btn.collidepoint(mp), self.font_small)
        self._draw_setting_slider(rx+20, cy+50, cw-40, "Volume", "sound_volume", self.settings.sound_volume, 0.0, 1.0, 0.1)

        # Right: Recognition
        ry = ty+185
        rc = pygame.Rect(rx, ry, cw, 155)
        cy = self.draw_card(rc, "Recognition Settings")
        self._draw_setting_slider(rx+20, cy+5, cw-40, "Window (s)", "window_seconds", self.settings.window_seconds, 1.0, 6.0, 0.5)
        self._draw_setting_slider(rx+20, cy+45, cw-40, "Confidence", "confidence_threshold", self.settings.confidence_threshold, 0.0, 0.5, 0.05)

        by = self.screen_height-65
        self.reset_btn = pygame.Rect(self.screen_width//2-170, by, 150, 45)
        self.apply_btn = pygame.Rect(self.screen_width//2+20, by, 150, 45)
        self.draw_button("Reset Default", self.reset_btn, Theme.ACCENT_RED, self.reset_btn.collidepoint(mp))
        self.draw_button("Apply", self.apply_btn, Theme.ACCENT_GREEN, self.apply_btn.collidepoint(mp))

    # ==========================================
    # SSVEP PARAMS PAGE (keyboard + parameter editor)
    # ==========================================
    def draw_ssvep_params_page(self):
        self.screen.fill(Theme.BG_PRIMARY)
        mp = pygame.mouse.get_pos()
        self.screen.blit(self.font_title.render("Settings", True, Theme.TEXT_PRIMARY), (30, 10))
        self.screen.blit(self.font_small.render("> SSVEP Parameters", True, Theme.TEXT_MUTED), (30, 48))

        margin, panel_w = 25, 320
        kb_w = self.screen_width - panel_w - margin*3

        # Keyboard card
        kc = pygame.Rect(margin, 75, kb_w, self.screen_height-140)
        cy = self.draw_card(kc, "Keyboard")
        self.ssvep_reset_btn = pygame.Rect(kc.right-100, kc.y+12, 80, 35)
        self.draw_button("Reset", self.ssvep_reset_btn, Theme.ACCENT_RED, self.ssvep_reset_btn.collidepoint(mp), self.font_small)
        self._draw_ssvep_keyboard(kc.x+20, cy+10, kc.width-40, kc.bottom-cy-30)

        # Parameters panel
        px = margin*2 + kb_w
        pc = pygame.Rect(px, 75, panel_w, self.screen_height-140)
        cy = self.draw_card(pc, "Parameters")

        if self.ssvep_selected_key:
            char = self.ssvep_selected_key
            params = self.beta_params[char]
            dn = "SPACE" if char == '_' else char.upper()
            self.screen.blit(self.font_header.render(f"Key: {dn}", True, Theme.ACCENT_GREEN), (px+25, cy+10))

            # Frequency field
            fy = cy+65
            self.screen.blit(self.font_body.render("Frequency (Hz)", True, Theme.TEXT_SECONDARY), (px+25, fy))
            self.ssvep_freq_rect = pygame.Rect(px+25, fy+30, panel_w-50, 45)
            fa = self.ssvep_editing_field == 'frequency'
            self.draw_rounded_rect(Theme.BG_SECONDARY, self.ssvep_freq_rect, 8, 2, Theme.ACCENT_BLUE if fa else Theme.BORDER_LIGHT)
            fd = self.ssvep_edit_text+"|" if fa else f"{params['frequency']:.1f}"
            self.screen.blit(self.font_header.render(fd, True, Theme.TEXT_PRIMARY), (self.ssvep_freq_rect.x+15, self.ssvep_freq_rect.y+8))

            # Phase field
            py_ = fy+100
            self.screen.blit(self.font_body.render("Phase (\u00d7 \u03c0)", True, Theme.TEXT_SECONDARY), (px+25, py_))
            self.ssvep_phase_rect = pygame.Rect(px+25, py_+30, panel_w-50, 45)
            pa = self.ssvep_editing_field == 'phase'
            self.draw_rounded_rect(Theme.BG_SECONDARY, self.ssvep_phase_rect, 8, 2, Theme.ACCENT_BLUE if pa else Theme.BORDER_LIGHT)
            pd = self.ssvep_edit_text+"|" if pa else f"{params['phase']/math.pi:.2f}"
            self.screen.blit(self.font_header.render(pd, True, Theme.TEXT_PRIMARY), (self.ssvep_phase_rect.x+15, self.ssvep_phase_rect.y+8))

            hy = py_+110
            for h in ["Click field to edit", "Enter to confirm", "ESC to cancel"]:
                self.screen.blit(self.font_small.render(h, True, Theme.TEXT_MUTED), (px+25, hy)); hy += 25
        else:
            s = self.font_body.render("Click a key to edit", True, Theme.TEXT_MUTED)
            self.screen.blit(s, s.get_rect(center=(px+panel_w//2, cy+100)))

        self.ssvep_back_btn = pygame.Rect(margin, self.screen_height-55, 120, 42)
        self.draw_button("<< Back", self.ssvep_back_btn, Theme.ACCENT_BLUE, self.ssvep_back_btn.collidepoint(mp), self.font_small)
        self.ssvep_next_btn = pygame.Rect(self.screen_width-margin-120, self.screen_height-55, 120, 42)
        self.draw_button("Next >>", self.ssvep_next_btn, Theme.ACCENT_BLUE, self.ssvep_next_btn.collidepoint(mp), self.font_small)

    def _draw_ssvep_keyboard(self, ax, ay, aw, ah):
        mp = pygame.mouse.get_pos()
        rows = len(self.keyboard_layout)
        mc = 11
        hs, vs = 8, 8

        # Calculate key size to FILL the available area
        kw = (aw - (mc - 1) * hs) // mc
        kh = (ah - (rows - 1) * vs) // rows

        # No max limit - fill the space
        total_h = rows * kh + (rows - 1) * vs
        total_w = mc * kw + (mc - 1) * hs
        sy = ay + (ah - total_h) // 2
        sx_offset = (aw - total_w) // 2

        self.ssvep_key_rects = {}

        for ri, row in enumerate(self.keyboard_layout):
            y = sy + ri * (kh + vs)
            if ri == 4:
                sw = kw * 5 + hs * 4
                tw = sw + hs + kw
                rx = ax + sx_offset + (total_w - tw) // 2
                self._draw_ssvep_key(pygame.Rect(rx, y, sw, kh), row[0], mp)
                self._draw_ssvep_key(pygame.Rect(rx + sw + hs, y, kw, kh), row[1], mp)
            else:
                rw = len(row) * kw + (len(row) - 1) * hs
                rx = ax + sx_offset + (total_w - rw) // 2
                for ci, ch in enumerate(row):
                    self._draw_ssvep_key(pygame.Rect(rx + ci * (kw + hs), y, kw, kh), ch, mp)

    def _draw_ssvep_key(self, rect, char, mp):
        self.ssvep_key_rects[char] = rect
        sel = char == self.ssvep_selected_key
        hov = rect.collidepoint(mp)
        bg = Theme.ACCENT_GREEN if sel else ((50, 55, 75) if hov else Theme.BG_SECONDARY)
        bc = Theme.ACCENT_GREEN if sel else (Theme.ACCENT_BLUE if hov else Theme.BORDER)
        self.draw_rounded_rect(bg, rect, 8, 1, bc)
        dn = "SP" if char == '_' else char.upper()
        tc = Theme.BG_PRIMARY if sel else Theme.TEXT_PRIMARY
        # Use larger font for key label
        s = self.font_key.render(dn, True, tc)
        self.screen.blit(s, s.get_rect(center=(rect.centerx, rect.centery - 10)))
        # Frequency label
        freq = self.beta_params[char]['frequency']
        fc = Theme.BG_PRIMARY if sel else Theme.TEXT_MUTED
        fs = self.font_small.render(f"{freq:.1f}", True, fc)
        self.screen.blit(fs, fs.get_rect(center=(rect.centerx, rect.bottom - 16)))

    # ==========================================
    # SIGNAL MONITOR PAGE
    # ==========================================
    def draw_monitor_page(self):
        self.screen.fill(Theme.BG_PRIMARY)
        mp = pygame.mouse.get_pos()
        W, H = self.screen_width, self.screen_height

        # ---- 顶部状态栏 ----
        bar = pygame.Rect(0, 0, W, 52)
        pygame.draw.rect(self.screen, Theme.BG_SECONDARY, bar)
        pygame.draw.line(self.screen, Theme.BORDER, (0, 52), (W, 52), 1)

        self.back_btn = pygame.Rect(15, 10, 90, 32)
        self.draw_button("<< Back", self.back_btn, Theme.BG_CARD,
                         self.back_btn.collidepoint(mp), self.font_small)

        title_s = self.font_body.render("EEG 信号监测", True, Theme.TEXT_PRIMARY)
        self.screen.blit(title_s, title_s.get_rect(center=(W // 2, 26)))

        conn = self.eeg_device and self.eeg_device.is_connected
        conn_str = "● 已连接" if conn else "○ 未连接"
        conn_col = Theme.ACCENT_GREEN if conn else Theme.ACCENT_RED
        cs = self.font_small.render(conn_str, True, conn_col)
        self.screen.blit(cs, (W - 130, 18))

        # ---- 控制栏 ----
        ctrl_y = 62
        self.monitor_pause_btn = pygame.Rect(15, ctrl_y, 80, 30)
        lbl = "▶ 继续" if self.eeg_monitor.paused else "⏸ 暂停"
        col = Theme.ACCENT_ORANGE if self.eeg_monitor.paused else Theme.BG_CARD
        self.draw_button(lbl, self.monitor_pause_btn, col,
                         self.monitor_pause_btn.collidepoint(mp), self.font_small)

        # 设备信息
        if self.eeg_device:
            info = (f"设备: {self.settings.device_type.upper()}  "
                    f"| {self.eeg_device.fs}Hz  "
                    f"| {self.eeg_device.n_channels}ch  "
                    f"| 缓冲: {len(self.eeg_device.buffer)}pts")
            
            # print(f"buffer:", len(self.eeg_device.buffer))
            self.screen.blit(self.font_small.render(info, True, Theme.TEXT_MUTED), (110, ctrl_y + 7))

        # ---- 波形区域 ----
        wave_rect = pygame.Rect(10, ctrl_y + 40, W - 20, H - ctrl_y - 50)

        # 更新监测数据
        if conn and self.eeg_device.is_streaming:
            self.eeg_monitor.update(self.eeg_device)

        n_ch = self.eeg_device.n_channels if self.eeg_device else self.settings.n_channels
        self.eeg_monitor.draw(self.screen, wave_rect, n_ch,
                              self.eeg_device.fs if self.eeg_device else self.settings.fs)

        # 未连接提示
        if not conn:
            hint = self.font_body.render("请先在主页连接设备并开始采集", True, Theme.TEXT_MUTED)
            self.screen.blit(hint, hint.get_rect(center=(W // 2, wave_rect.centery)))

    # ==========================================
    # DEMO / EXPERIMENT PAGES
    # ==========================================
    def draw_demo_page(self):
        self.screen.fill(Theme.BG_PRIMARY)
        # 更新监测数据
        if self.eeg_device and self.eeg_device.is_streaming:
            self.eeg_monitor.update(self.eeg_device)
        n_ch = self.eeg_device.n_channels if self.eeg_device else self.settings.n_channels
        fs   = self.eeg_device.fs if self.eeg_device else self.settings.fs
        kb_right = self.eeg_monitor.draw_sidebar(self.screen, self.screen_width, self.screen_height, n_ch, fs)
        # 键盘区域让出侧边栏
        self._draw_info_bar(kb_right)
        self._draw_keyboard(flickering=True, kb_right=kb_right)
        self.back_btn = pygame.Rect(20, self.screen_height-60, 100, 40)
        self.draw_button("Back", self.back_btn, Theme.BG_CARD, self.back_btn.collidepoint(pygame.mouse.get_pos()))

    def draw_experiment_page(self):
        self.screen.fill(Theme.BG_PRIMARY)
        if self.experiment_state is None: self._init_experiment()
        self._update_experiment()

        # 先绘制侧边栏，得到键盘可用右边界
        if self.eeg_device and self.eeg_device.is_streaming:
            self.eeg_monitor.update(self.eeg_device)
        n_ch = self.eeg_device.n_channels if self.eeg_device else self.settings.n_channels
        fs   = self.eeg_device.fs if self.eeg_device else self.settings.fs
        kb_right = self.eeg_monitor.draw_sidebar(self.screen, self.screen_width, self.screen_height, n_ch, fs)

        st = self.experiment_state['state']
        if st == 'ready': self._draw_ready()
        elif st in ('cue','flickering'):
            self._draw_info_bar(kb_right); self._draw_keyboard(flickering=(st=='flickering'), kb_right=kb_right)
        elif st == 'pause':
            self._draw_info_bar(kb_right); self._draw_keyboard(flickering=False, kb_right=kb_right); self._draw_processing()
        elif st == 'block_complete': self._draw_block_complete()
        elif st == 'rest': self._draw_rest()
        elif st == 'finished': self._draw_finished()

    def _draw_info_bar(self, kb_right=None):
        if kb_right is None:
            kb_right = self.screen_width - 20
        ir = pygame.Rect(20, 10, kb_right - 20, 70)
        self.draw_rounded_rect(Theme.BG_SECONDARY, ir, 10)
        if self.experiment_state:
            tgt = self.experiment_state.get('target_text','')
            res = self.experiment_state.get('result_text','')
            tr = self.experiment_state.get('current_trial',0)
            self.screen.blit(self.font_small.render("TARGET:", True, Theme.TEXT_MUTED), (40,17))
            self.screen.blit(self.font_body.render(tgt, True, Theme.TEXT_PRIMARY), (120,15))
            if tr < len(tgt):
                self.screen.blit(self.font_small.render("CUR:", True, Theme.TEXT_MUTED), (280,17))
                self.screen.blit(self.font_body.render(tgt[tr] if tgt[tr]!='_' else 'SP', True, Theme.ACCENT_RED), (330,15))
            self.screen.blit(self.font_small.render("RESULT:", True, Theme.TEXT_MUTED), (400,17))
            self.screen.blit(self.font_body.render(res or "---", True, Theme.ACCENT_GREEN), (480,15))
            acc = sum(1 for i,r in enumerate(res) if i<len(tgt) and r==tgt[i])/len(res) if res else 0
            itr = self.calculate_itr(acc if acc>0 else 0.8)
            self.screen.blit(self.font_small.render(f"ITR: {itr:.1f} | ACC: {acc*100:.0f}%", True, Theme.ACCENT_CYAN), (40,50))
            prog = tr/len(tgt) if tgt else 0
            self.screen.blit(self.font_small.render(f"Progress: {tr}/{len(tgt)}", True, Theme.TEXT_PRIMARY), (300,50))
            self.draw_progress_bar(pygame.Rect(kb_right - 180, 50, 150, 18), prog)

            # ★ Skip 按钮（仅在 flickering 状态显示）
            if self.experiment_state and self.experiment_state['state'] == 'flickering':
                self.skip_trial_btn = pygame.Rect(kb_right - 340, 12, 100, 36)
                self.draw_rounded_rect(Theme.ACCENT_ORANGE, self.skip_trial_btn, 8)
                skip_txt = self.font_small.render('SKIP >>', True, (255,255,255))
                self.screen.blit(skip_txt, skip_txt.get_rect(center=self.skip_trial_btn.center))
            else:
                self.skip_trial_btn = None

    def _draw_keyboard(self, flickering=False, kb_right=None):
        """Keyboard with correct frequency flickering using relative time"""
        if flickering:
            if self.flicker_start_time is None: self.flicker_start_time = time.time()
            ct = time.time() - self.flicker_start_time
        else:
            self.flicker_start_time = None; ct = 0

        if kb_right is None:
            kb_right = self.screen_width - 20
        kb_top, kb_bot, kb_l, kb_r = 90, self.screen_height-15, 20, kb_right
        kbw, kbh = kb_r-kb_l, kb_bot-kb_top
        rows, mc = len(self.keyboard_layout), 11
        hs, vs = 6, 6
        # Fill the entire available area - no max limits
        kw = (kbw - (mc - 1) * hs) // mc
        kh = (kbh - (rows - 1) * vs) // rows
        total_w = mc * kw + (mc - 1) * hs
        total_h = rows * kh + (rows - 1) * vs
        sy = kb_top + (kbh - total_h) // 2
        sx = kb_l + (kbw - total_w) // 2

        tgt_char = None
        if self.experiment_state:
            tgt = self.experiment_state.get('target_text','')
            trial = self.experiment_state.get('current_trial',0)
            state = self.experiment_state.get('state','')
            if trial < len(tgt) and state in ('cue', 'flickering'): tgt_char = tgt[trial]

        # In flickering mode with a target: only draw the target key (large, centered, flickering)
        # All other keys are drawn dimmed/greyed out without flickering
        if flickering and tgt_char is not None:
            self._draw_keyboard_single_target(tgt_char, ct, sx, sy, kw, kh, hs, vs, total_w, total_h)
            return

        for ri, row in enumerate(self.keyboard_layout):
            y = sy + ri * (kh + vs)
            if ri == 4:
                sw = kw * 5 + hs * 4
                tw = sw + hs + kw
                rx = sx + (total_w - tw) // 2
                self._draw_key(pygame.Rect(rx, y, sw, kh), row[0], ct, flickering, tgt_char)
                self._draw_key(pygame.Rect(rx + sw + hs, y, kw, kh), row[1], ct, flickering, tgt_char)
            else:
                rw = len(row) * kw + (len(row) - 1) * hs
                rx = sx + (total_w - rw) // 2
                for ci, ch in enumerate(row):
                    self._draw_key(pygame.Rect(rx + ci * (kw + hs), y, kw, kh), ch, ct, flickering, tgt_char)

    def _draw_key(self, rect, char, ct, flickering, tgt_char):
        params = self.beta_params.get(char, {'frequency': 10, 'phase': 0})
        freq, phase = params['frequency'], params['phase']
        is_tgt = char == tgt_char

        if flickering:
            raw = math.sin(2*math.pi*freq*ct + phase)
            intensity = 0.5 + 0.5*raw
            bg = (int(255*intensity),)*3
        elif is_tgt: bg = Theme.ACCENT_RED
        else: bg = Theme.BG_CARD

        self.draw_rounded_rect(bg, rect, 8, 1, Theme.BORDER)
        dn = "SP" if char=="_" else char.upper()
        if flickering:
            inv = 1.0-(0.5+0.5*raw)
            tc = (int(200*inv+30),)*3
        else: tc = Theme.TEXT_PRIMARY if not is_tgt else Theme.BG_PRIMARY
        s = self.font_key.render(dn, True, tc)
        self.screen.blit(s, s.get_rect(center=(rect.centerx, rect.centery-5)))
        if not flickering:
            fs = self.font_tiny.render(f"{freq:.1f}", True, Theme.TEXT_MUTED)
            self.screen.blit(fs, fs.get_rect(center=(rect.centerx, rect.bottom-14)))

    def _draw_keyboard_single_target(self, tgt_char, ct, sx, sy, kw, kh, hs, vs, total_w, total_h):
        """
        Flickering mode: draw ALL keys dimmed/grey (no flicker),
        then draw ONLY the target key very large and centered, flickering.
        """
        # --- Draw all keys dimmed ---
        for ri, row in enumerate(self.keyboard_layout):
            y = sy + ri * (kh + vs)
            if ri == 4:
                sw = kw * 5 + hs * 4
                tw = sw + hs + kw
                rx = sx + (total_w - tw) // 2
                for ch, rx2, rw2 in [(row[0], rx, sw), (row[1], rx + sw + hs, kw)]:
                    r = pygame.Rect(rx2, y, rw2, kh)
                    self.draw_rounded_rect((28, 32, 44), r, 8, 1, (45, 50, 65))
                    dn = "SP" if ch == "_" else ch.upper()
                    dim_s = self.font_key.render(dn, True, (55, 60, 75))
                    self.screen.blit(dim_s, dim_s.get_rect(center=r.center))
            else:
                rw = len(row) * kw + (len(row) - 1) * hs
                rx = sx + (total_w - rw) // 2
                for ci, ch in enumerate(row):
                    r = pygame.Rect(rx + ci * (kw + hs), y, kw, kh)
                    self.draw_rounded_rect((28, 32, 44), r, 8, 1, (45, 50, 65))
                    dn = "SP" if ch == "_" else ch.upper()
                    dim_s = self.font_key.render(dn, True, (55, 60, 75))
                    self.screen.blit(dim_s, dim_s.get_rect(center=r.center))

        # --- Draw the target key large and centered ---
        params = self.beta_params.get(tgt_char, {'frequency': 10, 'phase': 0})
        freq, phase = params['frequency'], params['phase']
        raw = math.sin(2 * math.pi * freq * ct + phase)
        intensity = 0.5 + 0.5 * raw

        # Big centered box: ~38% wide, 55% tall of keyboard area
        kb_right = sx + total_w
        kb_left = sx
        kb_top = sy
        kb_bot = sy + total_h
        box_w = int((kb_right - kb_left) * 0.38)
        box_h = int((kb_bot - kb_top) * 0.55)
        box_x = (kb_left + kb_right) // 2 - box_w // 2
        box_y = (kb_top + kb_bot) // 2 - box_h // 2
        big_rect = pygame.Rect(box_x, box_y, box_w, box_h)

        # Flickering background: black <-> white, no border
        bg = (int(255 * intensity),) * 3
        self.draw_rounded_rect(bg, big_rect, 16)

        # Large letter label
        dn = "SP" if tgt_char == "_" else tgt_char.upper()
        inv = 1.0 - intensity
        tc = (int(200 * inv + 30),) * 3
        big_font = pygame.font.Font(None, min(box_h - 40, 220))
        letter_s = big_font.render(dn, True, tc)
        self.screen.blit(letter_s, letter_s.get_rect(center=(big_rect.centerx, big_rect.centery - 18)))

        # Frequency label at bottom of box
        freq_font = pygame.font.Font(None, 40)
        freq_tc = (int(100 * inv + 80),) * 3
        freq_s = freq_font.render(f"{freq:.1f} Hz", True, freq_tc)
        self.screen.blit(freq_s, freq_s.get_rect(center=(big_rect.centerx, big_rect.bottom - 28)))

    def _draw_pre_flicker(self):
        """[DEPRECATED] Kept for backward compatibility only.

        This method used to render the "EEG Warming Up..." page that the
        experiment got stuck on. The state machine no longer enters the
        'pre_flicker' state; the experiment goes straight from 'ready'
        into 'cue'. This stub is preserved in case other code calls it,
        but it should not be reachable from the normal experiment flow.
        """
        W, H = self.screen_width, self.screen_height
        buf_len  = len(self.eeg_device.buffer) if self.eeg_device else 0
        buf_need = int(self.settings.window_seconds * self.settings.fs)
        fill_pct = min(1.0, buf_len / buf_need) if buf_need > 0 else 1.0
        secs_left = max(0, (buf_need - buf_len) / self.settings.fs) if self.settings.fs > 0 else 0
        blk_num = self.experiment_state.get('current_block', 0) + 1

        # ★ 画闪烁键盘（让大脑产生SSVEP）
        kb_right = self.eeg_monitor.draw_sidebar(
            self.screen, W, H,
            self.eeg_device.n_channels if self.eeg_device else self.settings.n_channels,
            self.eeg_device.fs if self.eeg_device else self.settings.fs
        )
        self._draw_keyboard(flickering=True, kb_right=kb_right)

        # 顶部进度条遮罩
        overlay = pygame.Surface((kb_right, 85), pygame.SRCALPHA)
        overlay.fill((15, 17, 23, 210))
        self.screen.blit(overlay, (0, 0))

        if fill_pct < 1.0:
            msg  = self.font_header.render(
                f"Block {blk_num}  |  EEG Warming Up...  {secs_left:.1f}s", True, Theme.ACCENT_ORANGE)
            sub  = self.font_small.render(
                "Focus on the keyboard — let your brain warm up!", True, Theme.TEXT_SECONDARY)
        else:
            msg  = self.font_header.render(
                f"Block {blk_num}  |  Ready!  Starting...", True, Theme.ACCENT_GREEN)
            sub  = self.font_small.render(
                "Keep focusing on the keyboard!", True, Theme.ACCENT_CYAN)

        self.screen.blit(msg, msg.get_rect(midleft=(20, 20)))
        self.screen.blit(sub, sub.get_rect(midleft=(20, 50)))

        # 进度条
        bar_w = min(400, kb_right - 40)
        bar_rect = pygame.Rect(kb_right - bar_w - 20, 12, bar_w, 18)
        pygame.draw.rect(self.screen, Theme.BG_CARD, bar_rect, border_radius=9)
        pygame.draw.rect(self.screen, Theme.BORDER, bar_rect, 1, border_radius=9)
        fill_color = Theme.ACCENT_GREEN if fill_pct >= 1.0 else Theme.ACCENT_ORANGE
        if fill_pct > 0:
            pygame.draw.rect(self.screen, fill_color,
                pygame.Rect(bar_rect.x+1, bar_rect.y+1,
                            int((bar_rect.w-2)*fill_pct), bar_rect.h-2), border_radius=8)
        pct_s = self.font_small.render(f"{fill_pct*100:.0f}%", True, Theme.TEXT_PRIMARY)
        self.screen.blit(pct_s, pct_s.get_rect(center=bar_rect.center))

    def _draw_ready(self):
        s = self.font_title.render("Get Ready...", True, Theme.ACCENT_CYAN)
        self.screen.blit(s, s.get_rect(center=(self.screen_width//2, self.screen_height//2)))

    def _draw_processing(self):
        ov = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        ov.fill((0,0,0,150)); self.screen.blit(ov, (0,0))
        s = self.font_header.render("Processing...", True, Theme.TEXT_PRIMARY)
        self.screen.blit(s, s.get_rect(center=(self.screen_width//2, self.screen_height//2)))

    def _draw_block_complete(self):
        tgt, res = self.experiment_state['target_text'], self.experiment_state['result_text']
        blk = self.experiment_state['current_block']
        cor = sum(1 for r,t in zip(res,tgt) if r==t)
        acc = (cor/len(tgt)*100) if tgt else 0
        itr = self.calculate_itr(acc/100)
        for txt, color, yoff in [(f"Block {blk+1} Complete!", Theme.ACCENT_GREEN, -80),
                                  (f"Target: {tgt} -> Result: {res}", Theme.TEXT_PRIMARY, 0),
                                  (f"Accuracy: {acc:.1f}% | ITR: {itr:.1f}", Theme.ACCENT_GREEN if acc>=80 else Theme.ACCENT_ORANGE, 60)]:
            s = self.font_header.render(txt, True, color)
            self.screen.blit(s, s.get_rect(center=(self.screen_width//2, self.screen_height//2+yoff)))

    def _draw_rest(self):
        rem = max(0, self.settings.rest_duration-(time.time()-self.experiment_state['state_start_time']))
        self.screen.blit(self.font_title.render("Take a Break", True, Theme.TEXT_PRIMARY),
                        self.font_title.render("Take a Break", True, Theme.TEXT_PRIMARY).get_rect(center=(self.screen_width//2, self.screen_height//2-80)))
        ts = pygame.font.Font(None, 100).render(f"{int(rem)}s", True, Theme.ACCENT_CYAN)
        self.screen.blit(ts, ts.get_rect(center=(self.screen_width//2, self.screen_height//2+20)))
        self.skip_btn = pygame.Rect(self.screen_width//2-60, self.screen_height-100, 120, 45)
        self.draw_button("Skip", self.skip_btn, Theme.BG_CARD, self.skip_btn.collidepoint(pygame.mouse.get_pos()))

    def _draw_finished(self):
        self.screen.blit(self.font_title.render("Experiment Complete!", True, Theme.ACCENT_GREEN),
                        self.font_title.render("Experiment Complete!", True, Theme.ACCENT_GREEN).get_rect(center=(self.screen_width//2, self.screen_height//2-100)))
        if self.block_history['accuracies']:
            aa = sum(self.block_history['accuracies'])/len(self.block_history['accuracies'])
            ai = sum(self.block_history['itrs'])/len(self.block_history['itrs'])
            s = self.font_header.render(f"Avg Acc: {aa:.1f}% | Avg ITR: {ai:.1f}", True, Theme.ACCENT_CYAN)
            self.screen.blit(s, s.get_rect(center=(self.screen_width//2, self.screen_height//2-30)))
        self.screen.blit(self.font_body.render("Press ESC to return", True, Theme.TEXT_SECONDARY),
                        self.font_body.render("Press ESC to return", True, Theme.TEXT_SECONDARY).get_rect(center=(self.screen_width//2, self.screen_height//2+80)))

    # ==========================================
    # EXPERIMENT LOGIC
    # ==========================================
    def get_block_role(self, block_idx):
        """Return 'calibration' or 'test' for a given block index.

        Following the dissertation protocol:
            block_idx 0 .. calibration_blocks-1  -> 'calibration'
            block_idx == calibration_blocks      -> 'test' (held-out)
        """
        return "calibration" if block_idx < self.settings.calibration_blocks else "test"

    def get_eval_targets(self):
        """Return the 18-target subset used for the offline CCA/TRCA evaluation.

        The GUI itself still flickers all 40 keys; this list is only the
        characters that are actually presented as cued targets in each block.
        """
        return list(self.settings.eval_target_sequence)

    # Legacy private alias kept for backward compatibility in case any
    # other code in the project still calls _get_block_role.
    def _get_block_role(self, block_idx):
        return self.get_block_role(block_idx)

    def _get_current_target_set_size(self):
        """Number of unique target characters used in the current block protocol."""
        chars = set("".join(self.settings.block_texts or []))
        return len(chars) if chars else len(self.char_to_freq)

    def _save_trial_compat(self, eeg_data, target_char, recognized_char,
                           frequency, confidence, block_idx, trial_idx,
                           correlations=None):
        """Save a single trial with the full 18-target subset metadata.

        Backward-compatible wrapper around DataSaver.add_trial_data:
        - Always passes the original positional/named fields the existing
          DataSaver supports (target_char, recognized_char, frequency, ...).
        - Tries to add the new 18-target evaluation metadata
          (block_role, target_frequency, fs, n_channels, eval_target_sequence)
          either as direct kwargs or, if DataSaver doesn't accept them,
          via an `extra_metadata` dict.
        - Falls back to the minimal call if neither path is supported,
          so this function never breaks data collection.
        """
        block_role = self.get_block_role(block_idx)
        target_frequency = self.char_to_freq.get(target_char, 0.0) if target_char else 0.0
        fs = self.eeg_device.fs if self.eeg_device else self.settings.fs
        n_channels = self.eeg_device.n_channels if self.eeg_device else self.settings.n_channels

        extra_metadata = {
            'block_role': block_role,
            'target_frequency': target_frequency,
            'fs': fs,
            'n_channels': n_channels,
            'eval_target_sequence': self.settings.eval_target_sequence,
            'calibration_blocks': self.settings.calibration_blocks,
            'test_blocks': self.settings.test_blocks,
        }

        # Attempt 1: pass the new fields as kwargs (works if DataSaver
        # uses **kwargs or has been extended to accept them).
        try:
            self.data_saver.add_trial_data(
                eeg_data=eeg_data,
                target_char=target_char,
                recognized_char=recognized_char,
                frequency=frequency,
                confidence=confidence,
                block_idx=block_idx,
                trial_idx=trial_idx,
                correlations=correlations,
                **extra_metadata,
            )
            return
        except TypeError:
            pass  # DataSaver signature doesn't accept these kwargs

        # Attempt 2: pass through a single extra_metadata dict.
        try:
            self.data_saver.add_trial_data(
                eeg_data=eeg_data,
                target_char=target_char,
                recognized_char=recognized_char,
                frequency=frequency,
                confidence=confidence,
                block_idx=block_idx,
                trial_idx=trial_idx,
                correlations=correlations,
                extra_metadata=extra_metadata,
            )
            return
        except TypeError:
            pass

        # Attempt 3: minimal compatible call. The offline_eval.py loader
        # can still recover block_role from block_idx vs calibration_blocks,
        # and target_frequency from target_char + char_to_freq mapping
        # (saved at session start via DataSaver.start_session(settings=...)).
        self.data_saver.add_trial_data(
            eeg_data=eeg_data,
            target_char=target_char,
            recognized_char=recognized_char,
            frequency=frequency,
            confidence=confidence,
            block_idx=block_idx,
            trial_idx=trial_idx,
            correlations=correlations,
        )

    def _init_experiment(self):
        # ----------------------------------------------------------------
        # SSVEP data-collection protocol (40-target interface prototype):
        #   - GUI shows the full 40-target keyboard layout
        #   - Each block uses the 18-target subset (EVAL_TARGET_SEQUENCE)
        #   - Blocks 0..calibration_blocks-1 = calibration
        #   - Final block (block_idx == calibration_blocks) = held-out test
        # No EEG warm-up screen is used. The experiment goes straight into
        # the standard ready -> cue -> flickering -> pause loop. Buffer
        # availability is handled at trial-save time, not as a UI block.
        # ----------------------------------------------------------------
        cb = len(self.block_history['results'])
        tgt = self.settings.block_texts[cb] if cb < len(self.settings.block_texts) else ""
        if cb == 0 and self.settings.save_data:
            self.settings.subject_id = self.subject_id_text
            self.data_saver.start_session(settings=self.settings, subject_id=self.subject_id_text)
        self.experiment_state = {
            'state': 'ready',  # ready -> cue -> flickering -> pause -> ...
            'current_block': cb,
            'block_role': self.get_block_role(cb),
            'current_trial': 0,
            'target_text': tgt,
            'result_text': '',
            'state_start_time': time.time(),
            'block_start_time': time.time()
        }
        if isinstance(self.eeg_device, SimulatedEEG) and tgt:
            self.eeg_device.set_target_frequency(self.char_to_freq.get(tgt[0], 10.0))

    def _update_experiment(self):
        if self.experiment_paused: return
        st = self.experiment_state['state']
        el = time.time() - self.experiment_state['state_start_time']

        # State machine: ready -> cue -> flickering -> pause -> next trial
        # On block end: -> block_complete -> rest -> next block
        # No EEG warm-up state; data is saved at end-of-flickering using
        # whatever buffer is currently available (offline analysis is done
        # later by offline_eval.py and is robust to short buffers).
        if st == 'ready' and el >= 2.0:
            self.experiment_state['state'] = 'cue'; self.experiment_state['state_start_time'] = time.time(); self.sound.play_cue()
        elif st == 'cue' and el >= self.settings.cue_duration:
            self.experiment_state['state'] = 'flickering'; self.experiment_state['state_start_time'] = time.time()
            self.flicker_start_time = None
            if self.recognizer: self.recognizer.reset()
        elif st == 'flickering' and el >= self.settings.flickering_duration:
            rc = self.get_recognized_char()
            tgt = self.experiment_state['target_text']
            tr = self.experiment_state['current_trial']
            tc = tgt[tr] if tr < len(tgt) else ''
            rf, rconf, rcorr = None, None, None
            with self.recognition_lock:
                if self.current_recognition:
                    rf = self.current_recognition.get('frequency')
                    rconf = self.current_recognition.get('confidence', 0)
                    rcorr = self.current_recognition.get('correlations')
            if rc is None and tr < len(tgt):
                rc = tc if np.random.random() < 0.8 else np.random.choice(list(self.char_to_freq.keys()))
                rf = self.char_to_freq.get(rc, 10.0); rconf = 0.5+np.random.random()*0.4
            if rc:
                self.experiment_state['result_text'] += rc
                self.sound.play_success() if rc == tc else self.sound.play_error()
            if self.settings.save_data:
                # Always save EEG epoch + metadata for offline CCA/TRCA evaluation,
                # even if online recognition is empty/unreliable.
                # Buffer may be shorter than window_samples for the very first trial:
                # we save whatever is available rather than blocking the UI.
                ed = self.eeg_device.get_data(self.recognizer.window_samples if self.recognizer else 1000) if self.eeg_device else None
                self._save_trial_compat(
                    eeg_data=ed, target_char=tc, recognized_char=rc or '',
                    frequency=rf or 0, confidence=rconf or 0,
                    block_idx=self.experiment_state['current_block'], trial_idx=tr,
                    correlations=rcorr,
                )
            self.experiment_state['current_trial'] += 1
            if self.experiment_state['current_trial'] >= len(tgt):
                self.experiment_state['state'] = 'block_complete'; self.sound.play_complete()
            else:
                self.experiment_state['state'] = 'pause'
                if isinstance(self.eeg_device, SimulatedEEG):
                    self.eeg_device.set_target_frequency(self.char_to_freq.get(tgt[self.experiment_state['current_trial']], 10.0))
            self.flicker_start_time = None; self.experiment_state['state_start_time'] = time.time()
        elif st == 'pause' and el >= self.settings.pause_duration:
            self.experiment_state['state'] = 'cue'; self.experiment_state['state_start_time'] = time.time(); self.sound.play_cue()
        elif st == 'block_complete' and el >= 2.0:
            res, tgt = self.experiment_state['result_text'], self.experiment_state['target_text']
            cor = sum(1 for r,t in zip(res,tgt) if r==t)
            acc = (cor/len(tgt)*100) if tgt else 0; itr = self.calculate_itr(acc/100)
            self.block_history['results'].append(res); self.block_history['accuracies'].append(acc); self.block_history['itrs'].append(itr)
            if self.settings.save_data:
                self.data_saver.add_block_result(block_idx=self.experiment_state['current_block'], target_text=tgt, result_text=res, accuracy=acc, itr=itr, duration=time.time()-self.experiment_state.get('block_start_time', time.time()))
            if self.experiment_state['current_block']+1 >= self.settings.total_blocks:
                self.experiment_state['state'] = 'finished'
                if self.settings.save_data: self.saved_files = self.data_saver.save_all()
            else: self.experiment_state['state'] = 'rest'
            self.experiment_state['state_start_time'] = time.time()
        elif st == 'rest' and el >= self.settings.rest_duration: self._next_block()


    def _skip_current_trial(self):
        """★ 跳过当前 trial，直接进入下一个字母"""
        if not self.experiment_state: return
        st = self.experiment_state
        tgt = st.get('target_text', '')
        tr  = st.get('current_trial', 0)
        tc  = tgt[tr] if tr < len(tgt) else ''

        # 跳过：result记为空字符串'?'
        st['result_text'] += '?'
        self.sound.play_error()

        # 保存trial数据（标记为跳过）
        if self.settings.save_data:
            self._save_trial_compat(
                eeg_data=None, target_char=tc, recognized_char='?',
                frequency=0, confidence=0,
                block_idx=st['current_block'], trial_idx=tr,
                correlations=None,
            )

        st['current_trial'] += 1
        if st['current_trial'] >= len(tgt):
            st['state'] = 'block_complete'
            self.sound.play_complete()
        else:
            st['state'] = 'pause'
            if isinstance(self.eeg_device, SimulatedEEG):
                next_tr = st['current_trial']
                self.eeg_device.set_target_frequency(
                    self.char_to_freq.get(tgt[next_tr], 10.0))
        self.flicker_start_time = None
        st['state_start_time'] = time.time()

    def _next_block(self):
        nb = self.experiment_state['current_block']+1
        if nb >= self.settings.total_blocks: self.experiment_state['state'] = 'finished'
        else:
            tgt = self.settings.block_texts[nb] if nb < len(self.settings.block_texts) else ""
            # No EEG warm-up: go straight into the standard ready -> cue
            # loop. The ready state already provides ~2s of preparation.
            self.experiment_state = {
                'state': 'ready',
                'current_block': nb,
                'block_role': self.get_block_role(nb),
                'current_trial': 0,
                'target_text': tgt,
                'result_text': '',
                'state_start_time': time.time(),
                'block_start_time': time.time()
            }
            if isinstance(self.eeg_device, SimulatedEEG) and tgt:
                self.eeg_device.set_target_frequency(self.char_to_freq.get(tgt[0], 10.0))

    # ==========================================
    # EVENT HANDLING
    # ==========================================
    def handle_click(self, pos):
        if self.current_page == "main": self._handle_main_click(pos)
        elif self.current_page == "settings": self._handle_settings_click(pos)
        elif self.current_page == "ssvep_params": self._handle_ssvep_click(pos)
        elif self.current_page == "monitor":
            if hasattr(self, 'back_btn') and self.back_btn.collidepoint(pos):
                self.current_page = "main"
            elif hasattr(self, 'monitor_pause_btn') and self.monitor_pause_btn.collidepoint(pos):
                self.eeg_monitor.paused = not self.eeg_monitor.paused
        elif self.current_page == "demo":
            # 侧边栏 toggle
            if self.eeg_monitor.toggle_btn_rect and self.eeg_monitor.toggle_btn_rect.collidepoint(pos):
                self.eeg_monitor.visible = not self.eeg_monitor.visible; return
            if hasattr(self, 'back_btn') and self.back_btn.collidepoint(pos): self.current_page = "main"; self.stop_recognition()
        elif self.current_page == "experiment":
            # 侧边栏 toggle
            if self.eeg_monitor.toggle_btn_rect and self.eeg_monitor.toggle_btn_rect.collidepoint(pos):
                self.eeg_monitor.visible = not self.eeg_monitor.visible; return
            if self.experiment_state and self.experiment_state['state'] == 'rest':
                if hasattr(self, 'skip_btn') and self.skip_btn.collidepoint(pos): self._next_block(); return
            # ★ Skip trial 按钮：flickering时跳过当前字母
            if self.experiment_state and self.experiment_state['state'] == 'flickering':
                if hasattr(self, 'skip_trial_btn') and self.skip_trial_btn and self.skip_trial_btn.collidepoint(pos):
                    self._skip_current_trial(); return

    def _handle_main_click(self, pos):
        # Dropdowns first (top layer)
        if self.device_dropdown_open:
            for i, opt in enumerate(['simulated','lsl','openbci','gds']):
                r = pygame.Rect(self.device_dropdown_rect.x+5, self.device_dropdown_rect.bottom+7+i*42, self.device_dropdown_rect.width-10, 36)
                if r.collidepoint(pos): self.settings.device_type = opt; self.device_dropdown_open = False; return
            self.device_dropdown_open = False; return
        if self.method_dropdown_open:
            for i, opt in enumerate(['cca','trca']):
                r = pygame.Rect(self.method_dropdown_rect.x+5, self.method_dropdown_rect.bottom+7+i*42, self.method_dropdown_rect.width-10, 36)
                if r.collidepoint(pos):
                    self.settings.recognition_method = opt
                    if self.recognizer: self.recognizer.set_method(opt)
                    self.method_dropdown_open = False; return
            self.method_dropdown_open = False; return

        if hasattr(self, 'subject_input_rect') and self.subject_input_rect.collidepoint(pos): self.subject_id_input_active = True; return
        else: self.subject_id_input_active = False
        if hasattr(self, 'save_toggle_rect') and self.save_toggle_rect.collidepoint(pos): self.settings.save_data = not self.settings.save_data; return
        if self.device_dropdown_rect.collidepoint(pos): self.device_dropdown_open = True; self.method_dropdown_open = False; return
        if self.method_dropdown_rect.collidepoint(pos): self.method_dropdown_open = True; self.device_dropdown_open = False; return

        if hasattr(self, 'connect_btn') and self.connect_btn.collidepoint(pos): self.connect_device()
        elif hasattr(self, 'disconnect_btn') and self.disconnect_btn.collidepoint(pos): self.disconnect_device()
        elif hasattr(self, 'ssvep_params_btn') and self.ssvep_params_btn.collidepoint(pos): self.current_page = "ssvep_params"
        elif hasattr(self, 'monitor_btn') and self.monitor_btn.collidepoint(pos):
            # 切换侧边栏默认显示状态
            self.eeg_monitor.visible = not self.eeg_monitor.visible
        elif hasattr(self, 'demo_btn') and self.demo_btn.collidepoint(pos):
            if not self.eeg_device: self.connect_device()
            self.start_recognition(); self.current_page = "demo"
        elif hasattr(self, 'settings_btn') and self.settings_btn.collidepoint(pos): self.current_page = "settings"
        elif hasattr(self, 'start_btn') and self.start_btn.collidepoint(pos):
            if not self.eeg_device: self.connect_device()
            self.start_recognition(); self.sound.play_start()
            self.experiment_state = None; self.block_history = {'results':[],'accuracies':[],'itrs':[]}; self.saved_files = []
            self.current_page = "experiment"

    def _handle_settings_click(self, pos):
        if hasattr(self, 'back_btn') and self.back_btn.collidepoint(pos): self.current_page = "main"; self.active_input_field = None; self.editing_block_idx = None; return

        # Block manager: Add block
        if hasattr(self, 'add_block_btn') and self.add_block_btn.collidepoint(pos):
            self.settings.block_texts.append('')
            self.settings.total_blocks = len(self.settings.block_texts)
            self.editing_block_idx = len(self.settings.block_texts) - 1
            self.editing_block_text = ''
            self.block_texts_input = ', '.join(self.settings.block_texts)
            return

        # Block manager: Delete block
        if hasattr(self, 'block_delete_btns'):
            for idx, rect in self.block_delete_btns.items():
                if rect.collidepoint(pos) and len(self.settings.block_texts) > 1:
                    self.settings.block_texts.pop(idx)
                    self.settings.total_blocks = len(self.settings.block_texts)
                    self.block_texts_input = ', '.join(self.settings.block_texts)
                    if self.editing_block_idx == idx:
                        self.editing_block_idx = None
                    elif self.editing_block_idx is not None and self.editing_block_idx > idx:
                        self.editing_block_idx -= 1
                    return

        # Block manager: Click block to edit
        if hasattr(self, 'block_item_rects'):
            for idx, rect in self.block_item_rects.items():
                if rect.collidepoint(pos):
                    # Don't start editing if clicking delete button
                    if hasattr(self, 'block_delete_btns') and idx in self.block_delete_btns and self.block_delete_btns[idx].collidepoint(pos):
                        continue
                    if self.editing_block_idx == idx:
                        return  # Already editing
                    # Save previous edit if any
                    if self.editing_block_idx is not None:
                        self._save_block_edit()
                    self.editing_block_idx = idx
                    self.editing_block_text = self.settings.block_texts[idx]
                    return

        # Click outside blocks: save current edit
        if hasattr(self, 'editing_block_idx') and self.editing_block_idx is not None:
            self._save_block_edit()
            self.editing_block_idx = None

        if hasattr(self, 'sound_toggle_rect') and self.sound_toggle_rect.collidepoint(pos):
            self.settings.sound_enabled = not self.settings.sound_enabled; self.sound.set_enabled(self.settings.sound_enabled); return
        if hasattr(self, 'test_sound_btn') and self.test_sound_btn.collidepoint(pos): self.sound.play_success(); return
        if hasattr(self, 'reset_btn') and self.reset_btn.collidepoint(pos): self._reset_settings(); return
        if hasattr(self, 'apply_btn') and self.apply_btn.collidepoint(pos): self._apply_settings(); self.sound.play_complete(); return
        if hasattr(self, 'setting_buttons'):
            for attr, b in self.setting_buttons.items():
                if b['minus'].collidepoint(pos): self._adjust_setting(attr, -b['step'], b['min'], b['max']); return
                if b['plus'].collidepoint(pos): self._adjust_setting(attr, b['step'], b['min'], b['max']); return

    def _save_block_edit(self):
        """Save the currently editing block text"""
        if self.editing_block_idx is not None and self.editing_block_idx < len(self.settings.block_texts):
            self.settings.block_texts[self.editing_block_idx] = self.editing_block_text
            self.settings.total_blocks = len(self.settings.block_texts)
            self.block_texts_input = ', '.join(self.settings.block_texts)

    def _handle_ssvep_click(self, pos):
        if hasattr(self, 'ssvep_back_btn') and self.ssvep_back_btn.collidepoint(pos): self.current_page = "main"; self.ssvep_editing_field = None; return
        if hasattr(self, 'ssvep_next_btn') and self.ssvep_next_btn.collidepoint(pos): self.current_page = "main"; self.ssvep_editing_field = None; return
        if hasattr(self, 'ssvep_reset_btn') and self.ssvep_reset_btn.collidepoint(pos): self.beta_params = self._generate_beta_params(); return
        if hasattr(self, 'ssvep_freq_rect') and self.ssvep_freq_rect.collidepoint(pos) and self.ssvep_selected_key:
            self.ssvep_editing_field = 'frequency'; self.ssvep_edit_text = f"{self.beta_params[self.ssvep_selected_key]['frequency']:.1f}"; return
        if hasattr(self, 'ssvep_phase_rect') and self.ssvep_phase_rect.collidepoint(pos) and self.ssvep_selected_key:
            self.ssvep_editing_field = 'phase'; self.ssvep_edit_text = f"{self.beta_params[self.ssvep_selected_key]['phase']/math.pi:.2f}"; return
        for char, rect in self.ssvep_key_rects.items():
            if rect.collidepoint(pos): self.ssvep_selected_key = char; self.ssvep_editing_field = None; return
        self.ssvep_editing_field = None

    def _adjust_setting(self, attr, delta, mn, mx):
        cur = getattr(self.settings, attr)
        nv = max(mn, min(mx, cur+delta))
        nv = int(round(nv)) if isinstance(cur, int) else round(nv, 2)
        setattr(self.settings, attr, nv)
        if attr == 'sound_volume': self.sound.set_volume(nv); self.sound.play_beep()

    def _apply_block_texts(self):
        ts = [t.strip() for t in self.block_texts_input.split(',') if t.strip()]
        if ts:
            self.settings.block_texts = ts
            self.settings.total_blocks = len(ts)
            self.settings.itr_n_targets = len(set("".join(ts))) or len(self.char_to_freq)

    def _reset_settings(self):
        self.settings = BCISettings(); self.block_texts_input = ', '.join(self.settings.block_texts)
        self.sound.set_enabled(self.settings.sound_enabled); self.sound.set_volume(self.settings.sound_volume)

    def _apply_settings(self):
        self._apply_block_texts()
        if self.recognizer:
            self.recognizer.window_samples = int(self.settings.window_seconds * self.settings.fs)
            self.recognizer.confidence_threshold = self.settings.confidence_threshold

    def _confirm_ssvep_edit(self):
        if not self.ssvep_selected_key or not self.ssvep_editing_field: return
        try:
            val = float(self.ssvep_edit_text)
            char = self.ssvep_selected_key
            if self.ssvep_editing_field == 'frequency':
                val = max(1.0, min(30.0, val))
                self.beta_params[char]['frequency'] = round(val, 1)
                self.char_to_freq[char] = round(val, 1)
                self.frequencies = sorted(set(self.char_to_freq.values()))
            elif self.ssvep_editing_field == 'phase':
                val = max(0.0, min(2.0, val))
                self.beta_params[char]['phase'] = round(val * math.pi, 4)
        except ValueError: pass
        self.ssvep_editing_field = None; self.ssvep_edit_text = ""

    # ==========================================
    # MAIN LOOP
    # ==========================================
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if self.current_page == "ssvep_params" and self.ssvep_editing_field:
                        if event.key == pygame.K_RETURN: self._confirm_ssvep_edit()
                        elif event.key == pygame.K_ESCAPE: self.ssvep_editing_field = None
                        elif event.key == pygame.K_BACKSPACE: self.ssvep_edit_text = self.ssvep_edit_text[:-1]
                        elif event.unicode in '0123456789.': self.ssvep_edit_text += event.unicode
                    elif self.current_page == "main" and self.subject_id_input_active:
                        if event.key == pygame.K_RETURN: self.subject_id_input_active = False
                        elif event.key == pygame.K_BACKSPACE: self.subject_id_text = self.subject_id_text[:-1]
                        elif event.key == pygame.K_ESCAPE: self.subject_id_input_active = False
                        elif event.unicode.isalnum() or event.unicode == '_':
                            if len(self.subject_id_text) < 20: self.subject_id_text += event.unicode
                    elif self.current_page == "settings" and self.active_input_field:
                        if event.key == pygame.K_RETURN: self._apply_block_texts(); self.active_input_field = None
                        elif event.key == pygame.K_BACKSPACE:
                            if self.active_input_field == 'block_texts': self.block_texts_input = self.block_texts_input[:-1]
                        elif event.key == pygame.K_ESCAPE: self.active_input_field = None
                        elif event.unicode.isprintable() and len(self.block_texts_input) < 100: self.block_texts_input += event.unicode
                    elif self.current_page == "settings" and hasattr(self, 'editing_block_idx') and self.editing_block_idx is not None:
                        if event.key == pygame.K_RETURN:
                            self._save_block_edit(); self.editing_block_idx = None
                        elif event.key == pygame.K_ESCAPE:
                            self.editing_block_idx = None
                        elif event.key == pygame.K_BACKSPACE:
                            self.editing_block_text = self.editing_block_text[:-1]
                        elif event.unicode.isprintable() and len(self.editing_block_text) < 50:
                            self.editing_block_text += event.unicode
                    elif event.key == pygame.K_ESCAPE:
                        if self.current_page in ("demo","experiment","monitor"): self.stop_recognition(); self.current_page = "main"
                        elif self.current_page in ("settings","ssvep_params"): self.current_page = "main"
                        else: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4 and self.current_page == "settings":  # scroll up
                        if hasattr(self, 'block_scroll'): self.block_scroll = max(0, self.block_scroll - 30)
                    elif event.button == 5 and self.current_page == "settings":  # scroll down
                        if hasattr(self, 'block_scroll'): self.block_scroll += 30
                    elif event.button == 1:
                        self.handle_click(event.pos)
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_width, self.screen_height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)

            if self.current_page == "main": self.draw_main_page()
            elif self.current_page == "settings": self.draw_settings_page()
            elif self.current_page == "ssvep_params": self.draw_ssvep_params_page()
            elif self.current_page == "monitor": self.draw_monitor_page()
            elif self.current_page == "demo": self.draw_demo_page()
            elif self.current_page == "experiment": self.draw_experiment_page()

            pygame.display.flip()
            self.clock.tick(self.fps)

        self.stop_recognition(); self.disconnect_device(); pygame.quit()


if __name__ == "__main__":
    print("="*60); print("Real-time SSVEP-BCI System"); print("="*60)
    system = RealTimeBCISystem()
    system.run()

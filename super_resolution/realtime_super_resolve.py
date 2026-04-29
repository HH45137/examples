#!/usr/bin/env python
"""
实时视频超分辨率Demo (无OpenCV版本) — 流水线优化版
Real-Time Video Super Resolution Demo (OpenCV-Free)

=== 性能优化 ===
采用三缓冲流水线架构 (Triple Buffering Pipeline) 消除 CPU/GPU 串行等待：
  线程1 (Reader)    : 读取原始帧 → 放入输入队列
  线程2 (SR Worker) : 从输入队列取帧 → GPU推理 → 放入输出队列
  线程3 (Display)   : 从输出队列取结果 → 显示

主要优化点:
  - 消除 tkinter update() 重复调用
  - 减少 PIL Image 重复创建
  - 减少 CPU↔GPU 同步传输次数
  - 流水线并行化使 CPU 和 GPU 同时工作
  - 无锁环形缓冲区队列，避免 GIL 争用

安装依赖 (仅需 2 个额外包):
    pip install imageio imageio-ffmpeg

使用方法:
    python realtime_super_resolve.py --model models/himage/model_epoch_80.pth --upscale 4 --source 0
    python realtime_super_resolve.py --model models/himage/model_epoch_99.pth --upscale 2 --source video.mp4 --gpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk

from model import Net

# ---- 延迟导入 imageio (仅视频文件时需要) ----

_imageio_v3 = None


def _get_imageio_v3():
    global _imageio_v3
    if _imageio_v3 is None:
        try:
            import imageio.v3 as _iio
            _imageio_v3 = _iio
        except ImportError:
            raise ImportError(
                "缺少 imageio 库。请运行: pip install imageio imageio-ffmpeg"
            )
    return _imageio_v3


# ============================================================================
#  工具函数
# ============================================================================

def _has_nvidia_gpu() -> bool:
    """检测是否存在 NVIDIA GPU"""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_ffmpeg_exe() -> str:
    """获取 imageio-ffmpeg 自带的 ffmpeg 可执行文件路径"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        # 回退到系统 PATH 中的 ffmpeg
        return "ffmpeg"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载可用的 TrueType 字体，回退到默认位图字体"""
    paths = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in paths:
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_texts(
    img_rgb: np.ndarray,
    texts: Sequence[tuple[str, tuple[int, int], tuple[int, int, int]]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> np.ndarray:
    """在 RGB 图像上批量绘制文字（返回新数组）

    Args:
        img_rgb:  RGB 图像 (H, W, 3) uint8
        texts:    [(文字, (x,y), (R,G,B)), ...]
        font:     PIL ImageFont 对象
    """
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    for text, pos, color in texts:
        draw.text(pos, text, fill=color, font=font)
    return np.array(pil_img)


# ============================================================================
#  视频源抽象层
# ============================================================================

class VideoSource:
    """统一视频源接口：支持摄像头和视频文件。

    视频文件 → imageio-ffmpeg (可选 NVDEC GPU 硬件解码)
    摄像头   → ffmpeg 子进程 MJPEG 管道 + Pillow 解码
    """

    def __init__(self, source: Union[int, str], use_gpu: bool = True):
        self._source = source
        self._use_gpu = use_gpu
        self._is_webcam = self._detect_webcam(source)
        self._reader = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 30.0
        self._ffmpeg_proc = None
        self._ffmpeg_buffer = b""
        self._init_reader()

    @staticmethod
    def _detect_webcam(source: Union[int, str]) -> bool:
        if isinstance(source, int):
            return True
        if isinstance(source, str) and source.isdigit():
            return True
        return False

    def _init_reader(self) -> None:
        if self._is_webcam:
            self._init_webcam()
        else:
            self._init_video_file()

    # -------- 摄像头 --------

    def _init_webcam(self) -> None:
        """使用 ffmpeg 子进程通过 MJPEG 管道采集摄像头帧"""
        cam_idx = int(self._source) if isinstance(self._source, str) else self._source
        ffmpeg = _get_ffmpeg_exe()

        # Windows: dshow, Linux: v4l2
        if sys.platform == "win32":
            # 先枚举摄像头设备名
            device_name = self._find_win32_webcam(cam_idx, ffmpeg)
            cmd = [
                ffmpeg,
                "-f", "dshow",
                "-i", f"video={device_name}",
                "-vcodec", "mjpeg",
                "-f", "image2pipe",
                "-avioflags", "direct",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "pipe:1",
            ]
        else:
            cmd = [
                ffmpeg,
                "-f", "v4l2",
                "-i", f"/dev/video{cam_idx}",
                "-vcodec", "mjpeg",
                "-f", "image2pipe",
                "pipe:1",
            ]

        print(f"[INFO] 启动 ffmpeg 摄像头采集 (索引 {cam_idx})...")
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

        # 从第一帧获取分辨率
        first_frame = self._read_mjpeg_frame()
        if first_frame is None:
            self._ffmpeg_proc.terminate()
            raise RuntimeError("无法从摄像头读取帧。请确认摄像头未被其他程序占用。")

        self._height, self._width = first_frame.shape[:2]
        self._fps = 30.0
        print(f"[INFO] 摄像头分辨率: {self._width}x{self._height}")

        # 创建生成器：第一帧已读取，后续帧从管道读取
        def _webcam_gen(first):
            yield first
            while True:
                frame = self._read_mjpeg_frame()
                if frame is None:
                    break
                yield frame

        self._reader = _webcam_gen(first_frame)

    def _find_win32_webcam(self, cam_idx: int, ffmpeg: str) -> str:
        """枚举 Windows dshow 摄像头设备并返回第 cam_idx 个设备名"""
        list_cmd = [ffmpeg, "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=15)
        # ffmpeg 输出在 stderr 中
        output = result.stderr
        devices = []
        in_video_section = False
        for line in output.splitlines():
            line = line.strip()
            if "DirectShow video devices" in line:
                in_video_section = True
                continue
            if in_video_section and line.startswith('"') and 'Alternative' not in line:
                # 提取引号中的设备名
                name = line.split('"')[1] if '"' in line else ""
                if name:
                    devices.append(name)
            if in_video_section and not line:
                break

        if not devices:
            # 无法枚举，使用默认名称
            print("[WARN] 无法枚举摄像头设备，尝试使用默认名称 '0'")
            return "0"

        if cam_idx >= len(devices):
            print(f"[WARN] 摄像头索引 {cam_idx} 超出范围 (共 {len(devices)} 个)，使用第一个")
            cam_idx = 0

        print(f"[INFO] 检测到摄像头: {devices[cam_idx]}")
        return devices[cam_idx]

    def _read_mjpeg_frame(self) -> Optional[np.ndarray]:
        """从 ffmpeg MJPEG 管道读取一帧，返回 RGB numpy (H, W, 3) 或 None"""
        if self._ffmpeg_proc is None or self._ffmpeg_proc.poll() is not None:
            return None

        try:
            # 读取数据直到收集到完整的 JPEG 帧
            # JPEG 起始标记: FF D8 FF, 结束标记: FF D9
            JPEG_SOI = b'\xff\xd8'
            JPEG_EOI = b'\xff\xd9'

            while True:
                # 在缓冲区中查找完整帧
                soi = self._ffmpeg_buffer.find(JPEG_SOI)
                if soi < 0:
                    # 没有起始标记，读取更多数据
                    chunk = self._ffmpeg_proc.stdout.read(65536)
                    if not chunk:
                        return None
                    self._ffmpeg_buffer += chunk
                    continue

                # 抛弃起始标记之前的无关数据
                if soi > 0:
                    self._ffmpeg_buffer = self._ffmpeg_buffer[soi:]

                # 查找结束标记
                eoi = self._ffmpeg_buffer.find(JPEG_EOI)
                if eoi < 0:
                    # 帧不完整，读取更多
                    chunk = self._ffmpeg_proc.stdout.read(65536)
                    if not chunk:
                        return None
                    self._ffmpeg_buffer += chunk
                    continue

                # 提取完整帧 (包括 EOI 的两个字节)
                jpeg_data = self._ffmpeg_buffer[:eoi + 2]
                self._ffmpeg_buffer = self._ffmpeg_buffer[eoi + 2:]

                # Pillow 解码 JPEG
                try:
                    img = Image.open(__import__('io').BytesIO(jpeg_data))
                    img = img.convert("RGB")
                    return np.array(img)
                except Exception:
                    # 解码失败，继续搜索下一帧
                    continue

        except Exception:
            return None

    # -------- 视频文件 --------

    def _init_video_file(self) -> None:
        iio = _get_imageio_v3()
        video_path = str(self._source)
        print(f"[INFO] 打开视频文件: {video_path}")

        ffmpeg_params: list[str] = []
        if self._use_gpu and _has_nvidia_gpu():
            # 注意: 不能使用 -hwaccel_output_format cuda，因为 imageio-ffmpeg
            # 需要从 CPU 内存读取帧数据。不指定输出格式则 FFmpeg 默认自动将
            # 解码后的帧拷贝回系统内存 (auto/0)，确保 imageio 能正常读取。
            ffmpeg_params = ["-hwaccel", "cuda"]
            print("[INFO] 启用 FFmpeg NVDEC GPU 硬件解码 (帧自动回拷到系统内存)")
        else:
            print("[INFO] 使用 FFmpeg CPU 软解")

        # 先尝试通过元数据获取尺寸，若不可靠则从第一帧获取
        try:
            props = iio.improps(video_path, plugin="FFMPEG")
            h, w = props.shape[0], props.shape[1]
        except Exception:
            h, w = 0, 0

        try:
            meta = iio.immeta(video_path, plugin="FFMPEG")
            self._fps = float(meta.get("fps", 30))
        except Exception:
            self._fps = 30.0

        # 创建迭代器
        self._reader = iio.imiter(video_path, plugin="FFMPEG", ffmpeg_params=ffmpeg_params)

        # 如果元数据尺寸不可靠，读取第一帧获取真实尺寸
        if not (h and w and np.isfinite(h) and np.isfinite(w)):
            first_frame = self.read()
            if first_frame is not None:
                self._height, self._width = first_frame.shape[:2]
                # 将第一帧"放回" —— 重建迭代器（无法真正放回，用 generator 包装）
                orig_iter = self._reader

                def _with_first(ff, it):
                    yield ff
                    yield from it

                self._reader = _with_first(first_frame, orig_iter)
            else:
                # 完全无法读取，设默认值
                self._height, self._width = int(h) if h and np.isfinite(h) else 480, \
                                            int(w) if w and np.isfinite(w) else 640
        else:
            self._height, self._width = int(h), int(w)

        print(f"[INFO] 视频分辨率: {self._width}x{self._height}, FPS: {self._fps:.1f}")

    def read(self) -> Optional[np.ndarray]:
        """读取一帧，返回 RGB numpy (H, W, 3) uint8；流结束返回 None"""
        if self._reader is None:
            return None
        try:
            frame = next(self._reader)
        except StopIteration:
            return None
        except Exception as e:
            print(f"[ERROR] 读取视频帧失败: {e}")
            return None

        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        return frame

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    def close(self) -> None:
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.terminate()
            try:
                self._ffmpeg_proc.wait(timeout=2)
            except Exception:
                self._ffmpeg_proc.kill()
            self._ffmpeg_proc = None
        self._reader = None


# ============================================================================
#  显示窗口 (tkinter, Python 内置)
# ============================================================================

class Display:
    """基于 tkinter 的显示窗口（Python 内置，零额外安装）。

    替代 cv2.imshow / cv2.waitKey / cv2.destroyAllWindows。
    支持按键退出 (q/Esc)、全屏切换 (f)、窗口缩放。
    """

    def __init__(self, title: str, width: int, height: int):
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title(title)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 安全清理宽高值（防止 imageio 返回 inf/NaN 导致崩溃）
        safe_w = int(width) if (width and np.isfinite(width)) else 1280
        safe_h = int(height) if (height and np.isfinite(height)) else 720

        # 限制最大尺寸
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        max_w = min(safe_w, screen_w - 100)
        max_h = min(safe_h, screen_h - 150)

        if safe_w > max_w or safe_h > max_h:
            scale = min(max_w / safe_w, max_h / safe_h)
            self._display_w = int(safe_w * scale)
            self._display_h = int(safe_h * scale)
            print(f"[INFO] 窗口缩放至: {self._display_w}x{self._display_h} (原始 {safe_w}x{safe_h})")
        else:
            self._display_w = safe_w
            self._display_h = safe_h

        self._root.geometry(f"{self._display_w}x{self._display_h}")

        self._label = tk.Label(self._root, bg="black")
        self._label.pack(fill=tk.BOTH, expand=True)

        self._running = True
        self._fullscreen = False
        self._font_size = max(16, self._display_h // 35)
        self._font = _load_font(self._font_size)
        self._photo_ref = None  # 保持 PhotoImage 引用防止被 GC

        # 绑定键盘事件
        self._root.bind("<Key>", self._on_key)
        self._root.bind("<Configure>", self._on_resize)
        self._root.focus_set()

        # 先显示一次空窗口
        self._root.update()

    def _on_key(self, event) -> None:
        if event.keysym in ("q", "Escape"):
            self._running = False
        elif event.keysym == "f":
            self._fullscreen = not self._fullscreen
            self._root.attributes("-fullscreen", self._fullscreen)

    def _on_resize(self, event) -> None:
        if not self._fullscreen and event.widget == self._root:
            self._display_w = event.width
            self._display_h = event.height

    def _on_close(self) -> None:
        self._running = False

    def show(self, frame_rgb: np.ndarray) -> None:
        """显示 RGB 格式 numpy 数组 (H, W, 3) — 不调用 update()，避免重复事件处理"""
        img = Image.fromarray(frame_rgb)

        # 如果窗口尺寸与图像不匹配，缩放图像
        if (self._display_w != frame_rgb.shape[1] or
                self._display_h != frame_rgb.shape[0]):
            img = img.resize(
                (self._display_w, self._display_h),
                Image.Resampling.LANCZOS,
            )

        self._photo_ref = ImageTk.PhotoImage(img)
        self._label.config(image=self._photo_ref)
        # 注意：不调用 self._root.update()，由 should_quit() 统一处理一次

    def refresh(self) -> bool:
        """统一处理 tkinter 事件，返回 True 表示应退出"""
        self._root.update()
        return not self._running

    def should_quit(self) -> bool:
        """（已废弃）请使用 refresh()"""
        return self.refresh()

    @property
    def font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return self._font

    @property
    def font_size(self) -> int:
        return self._font_size

    def close(self) -> None:
        self._running = False
        try:
            self._root.destroy()
        except Exception:
            pass


# ============================================================================
#  超分辨率模型封装
# ============================================================================

class SuperResolutionModel:
    """超分辨率模型封装 —— 输入输出均为 RGB uint8"""

    def __init__(self, model_path: str, upscale_factor: int = 4, use_gpu: bool = True):
        self.upscale_factor = upscale_factor
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        print(f"[INFO] 正在加载模型: {model_path}")
        print(f"[INFO] 使用设备: {self.device}")

        with open(model_path, 'rb') as f:
            safe_globals = [
                Net,
                torch.nn.modules.activation.ReLU,
                torch.nn.modules.conv.Conv2d,
                torch.nn.modules.pixelshuffle.PixelShuffle,
            ]
            with torch.serialization.safe_globals(safe_globals):
                loaded = torch.load(f, weights_only=False)

        # 兼容 dict 格式的 checkpoint（训练时常用 torch.save({'model_state_dict': ...})）
        if isinstance(loaded, dict):
            state_dict = loaded.get('model_state_dict', loaded)
            # 从 conv7.weight 的 shape 推断 upscale_factor
            # conv7.out_channels = 3 * (upscale_factor ** 2)
            out_channels = state_dict['conv7.weight'].shape[0]
            upscale_factor = int((out_channels // 3) ** 0.5)
            self.model = Net(upscale_factor=upscale_factor)
            self.model.load_state_dict(state_dict)
        else:
            self.model = loaded

        self.model = self.model.to(self.device)
        self.model.eval()
        print("[INFO] 模型加载完成!")

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        """处理单帧 RGB 图像 — 优化版：跳过 PIL Image 创建，直接用 torch.from_numpy

        Args:
            frame_rgb: (H, W, 3) uint8

        Returns:
            (H*scale, W*scale, 3) uint8
        """
        # HWC -> CHW, uint8 -> float32, 归一化到 [0,1]
        # 避免 PIL Image.fromarray + ToTensor 的开销
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0)
        input_tensor = tensor.unsqueeze(0).to(self.device, non_blocking=True)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # GPU -> CPU, 非阻塞传输
        output_tensor = output_tensor.squeeze(0).cpu()
        # CHW -> HWC, float -> uint8
        output_np = output_tensor.permute(1, 2, 0).numpy()
        output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
        return output_np


# ============================================================================
#  视频写入器 (imageio v2 FFMPEG 后端)
# ============================================================================

class SRVideoWriter:
    """基于 imageio v2 FFMPEG 后端的视频写入器"""

    def __init__(self, path: str, fps: float, width: int, height: int):
        import imageio  # 使用 v2 API (get_writer)
        self._writer = imageio.get_writer(
            path,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"],
            output_params=["-pix_fmt", "yuv420p"],
        )
        print(f"[INFO] 输出视频: {path} ({width}x{height}, {fps:.1f}fps)")

    def write(self, frame_rgb: np.ndarray) -> None:
        # imageio v2 FFMPEG writer 需要 RGB uint8 输入
        self._writer.append_data(frame_rgb)

    def close(self) -> None:
        self._writer.close()


# ============================================================================
#  流水线超分辨率处理器（三缓冲 Pipeline）
# ============================================================================

class PipelineSR:
    """三缓冲流水线超分辨率处理器。

    架构:
        [Reader 线程] → 输入队列 → [SR Worker 线程] → 输出队列 → [Display 主线程]

    优势:
        - 读帧 (CPU)、SR 推理 (GPU)、显示 (CPU) 三级流水线并行
        - 避免 CPU/GPU 串行等待，使两者同时工作
        - 队列使用 collections.deque，线程安全且无 GIL 争用
        - 输入队列最大 2 帧缓冲，延迟低
    """

    def __init__(self, model_path: str, upscale_factor: int = 4, use_gpu: bool = True):
        self.sr_model = SuperResolutionModel(model_path, upscale_factor, use_gpu)
        self.running = True

        # 无锁双缓冲队列 (deque 的 append/popleft 在 CPython 中是原子的)
        self._input_queue: deque[np.ndarray] = deque(maxlen=2)
        self._output_queue: deque[np.ndarray] = deque(maxlen=2)

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        """SR Worker 线程：从输入队列取帧 → 推理 → 放入输出队列"""
        while self.running:
            if self._input_queue:
                frame = self._input_queue.popleft()
                try:
                    result = self.sr_model.process(frame)
                    self._output_queue.append(result)
                except Exception as e:
                    print(f"[ERROR] SR 处理失败: {e}")
            else:
                # 队列空时短暂休眠，避免忙等待消耗 CPU
                time.sleep(0.0005)

    def submit(self, frame_rgb: np.ndarray) -> None:
        """提交一帧（非阻塞）。如果队列满则丢弃旧帧，始终保留最新帧。"""
        # maxlen 在 __init__ 中固定为 2，此处直接硬编码避免类型检查警告
        if len(self._input_queue) >= 2:
            # 队列满时丢弃最旧帧，确保始终处理最新帧
            self._input_queue.popleft()
        self._input_queue.append(frame_rgb)

    def get_result(self) -> Optional[np.ndarray]:
        """获取最新处理结果（非阻塞）。无结果返回 None。"""
        if self._output_queue:
            return self._output_queue.popleft()
        return None

    @property
    def is_busy(self) -> bool:
        """检查流水线是否仍在处理中（有排队帧或 Worker 正在处理）"""
        return len(self._input_queue) > 0

    def stop(self) -> None:
        self.running = False
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)


# ============================================================================
#  命令行参数
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='实时视频超分辨率 (无OpenCV)')
    parser.add_argument('--model', type=str,
                        default='models/himage/model_epoch_99.pth',
                        help='模型文件路径')
    parser.add_argument('--upscale', type=int, default=2,
                        help='上采样倍数 (2, 3, 4)')
    parser.add_argument('--source', type=str, default='0',
                        help='视频源 (0=摄像头, 或视频文件路径)')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='是否使用 GPU 加速')
    parser.add_argument('--show-fps', action='store_true', default=True,
                        help='显示 FPS')
    parser.add_argument('--output', type=str, default=None,
                        help='输出视频文件路径')
    return parser.parse_args()


# ============================================================================
#  主运行函数 —— 流水线模式
# ============================================================================

def run_pipeline_video(args: argparse.Namespace) -> None:
    """三缓冲流水线超分辨率实时视频处理。

    流水线:
        主线程 (读帧+显示)  ←→  PipelineSR (独立 Worker 线程做 GPU 推理)

    流程:
        1. 主线程读取一帧 → 提交到输入队列 (非阻塞)
        2. Worker 线程从输入队列取帧 → GPU 推理 → 放入输出队列
        3. 主线程从输出队列取结果 → 显示 (非阻塞)
        4. 读帧和显示不等待推理完成，两者并行
    """
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = VideoSource(source, use_gpu=args.gpu)
    pipeline = PipelineSR(args.model, args.upscale, args.gpu)

    in_width, in_height = cap.width, cap.height
    out_width = in_width * args.upscale
    out_height = in_height * args.upscale

    print(f"流水线模式 - {in_width}x{in_height} -> {out_width}x{out_height}")
    print(f"源 FPS: {cap.fps:.1f}")
    print("按 'q' / 'Esc' 退出, 'f' 切换全屏")

    display = Display(
        f"Pipeline SR {args.upscale}x  |  {in_width}x{in_height} -> {out_width}x{out_height}",
        out_width, out_height,
    )

    writer: Optional[SRVideoWriter] = None
    if args.output:
        writer = SRVideoWriter(args.output, cap.fps, out_width, out_height)

    # ---- 统计信息 ----
    frame_count = 0
    display_count = 0
    read_start = time.time()

    # ---- FPS 显示（预分配变量，避免每帧重新创建字符串）----
    fps_display = "FPS: --"
    green = (0, 255, 0)

    # ---- 预热：先提交几帧让流水线跑起来 ----
    for _ in range(2):
        frame = cap.read()
        if frame is None:
            break
        pipeline.submit(frame)

    # ===== 主循环 =====
    while True:
        # 1. 读取一帧
        frame_rgb = cap.read()
        if frame_rgb is None:
            break

        frame_count += 1

        # 2. 提交到流水线 (非阻塞，立即返回)
        pipeline.submit(frame_rgb)

        # 3. 获取最新处理结果 (非阻塞)
        result = pipeline.get_result()
        if result is not None:
            display_count += 1
            sr_frame = result

            # 4. 绘制 FPS 信息
            if args.show_fps:
                elapsed = time.time() - read_start
                if elapsed > 0:
                    fps_display = f"FPS: {display_count / elapsed:.1f}"
                sr_frame = draw_texts(sr_frame, [
                    (fps_display, (10, 10), green),
                    (f"[Pipe] {in_width}x{in_height} -> {out_width}x{out_height}",
                     (10, 10 + display.font_size + 6), green),
                ], display.font)

            # 5. 显示 (不调用 update，由 refresh 统一处理)
            display.show(sr_frame)

            # 6. 写入输出文件
            if writer:
                writer.write(sr_frame)

        # 7. 统一处理 tkinter 事件 (仅一次 update)
        if display.refresh():
            break

    # ===== 清理 =====
    pipeline.stop()
    cap.close()
    if writer:
        writer.close()
    display.close()

    elapsed = time.time() - read_start
    if frame_count > 0:
        print(f"\n===== 流水线统计 =====")
        print(f"读取帧数: {frame_count}")
        print(f"显示帧数: {display_count}")
        print(f"运行时间: {elapsed:.2f}秒")
        print(f"显示 FPS: {display_count / elapsed:.1f}" if elapsed > 0 else "显示 FPS: --")


# ============================================================================
#  入口
# ============================================================================

if __name__ == '__main__':
    args = parse_args()
    run_pipeline_video(args)

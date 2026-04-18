#!/usr/bin/env python
"""
实时视频超分辨率Demo
Real-Time Video Super Resolution Demo

使用方法:
    python realtime_super_resolve.py --model models/himage/model_epoch_80.pth --upscale 4 --source 0
    
参数:
    --model: 模型文件路径
    --upscale: 上采样倍数 (2, 3, 4)
    --source: 视频源 (0=摄像头, 视频文件路径)
    --gpu: 是否使用GPU加速
"""

import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import threading
import time

from model import Net


class SuperResolutionModel:
    """超分辨率模型封装类"""
    
    def __init__(self, model_path, upscale_factor=4, use_gpu=True):
        self.upscale_factor = upscale_factor
        self.use_gpu = use_gpu and self._check_gpu()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # 加载模型
        self._load_model(model_path)
        
    def _check_gpu(self):
        """检查GPU是否可用"""
        return torch.cuda.is_available()
        
    def _load_model(self, model_path):
        """加载训练好的模型"""
        print(f"正在加载模型: {model_path}")
        print(f"使用设备: {self.device}")
        
        with open(model_path, 'rb') as f:
            safe_globals = [
                Net,
                torch.nn.modules.activation.ReLU,
                torch.nn.modules.conv.Conv2d,
                torch.nn.modules.pixelshuffle.PixelShuffle,
            ]
            with torch.serialization.safe_globals(safe_globals):
                self.model = torch.load(f, weights_only=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("模型加载完成!")
        
    def process(self, frame):
        """处理单帧图像
        
        Args:
            frame: OpenCV BGR格式图像
            
        Returns:
            处理后的BGR格式图像
        """
        # BGR转RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # RGB转YCbCr (只处理Y通道)
        pil_img = Image.fromarray(frame_rgb).convert('YCbCr')
        y, cb, cr = pil_img.split()
        
        # 转换为tensor
        to_tensor = ToTensor()
        input_y = to_tensor(y).view(1, -1, y.size[1], y.size[0])
        input_y = input_y.to(self.device)
        
        # 模型推理
        with torch.no_grad():
            output_y = self.model(input_y)
        
        # 转换回图像
        output_y = output_y.cpu()[0].detach().numpy()
        output_y = (output_y * 255.0).clip(0, 255)
        output_y = Image.fromarray(np.uint8(output_y[0]), mode='L')
        
        # Cb, Cr通道上采样 (使用双三次插值)
        cb_up = cb.resize(output_y.size, Image.BICUBIC)
        cr_up = cr.resize(output_y.size, Image.BICUBIC)
        
        # 合并通道
        result = Image.merge('YCbCr', [output_y, cb_up, cr_up]).convert('RGB')
        
        # 转回BGR (OpenCV格式)
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        
        return result_bgr


class AsyncSuperResolution:
    """异步超分辨率处理器 - 使用独立线程处理帧"""
    
    def __init__(self, model_path, upscale_factor=4, use_gpu=True):
        self.sr_model = SuperResolutionModel(model_path, upscale_factor, use_gpu)
        self.current_frame = None
        self.processed_frame = None
        self.running = True
        self.frame_ready = False
        
        # 启动处理线程
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def _process_loop(self):
        """后台处理循环"""
        while self.running:
            if self.frame_ready and self.current_frame is not None:
                self.processed_frame = self.sr_model.process(self.current_frame)
                self.frame_ready = False
            time.sleep(0.001)  # 避免CPU占用过高
            
    def submit(self, frame):
        """提交新帧进行处理"""
        self.current_frame = frame
        self.frame_ready = True
        
    def get_result(self):
        """获取处理结果"""
        return self.processed_frame
    
    def is_processing(self):
        """检查是否正在处理"""
        return self.frame_ready
    
    def stop(self):
        """停止处理"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='实时视频超分辨率')
    parser.add_argument('--model', type=str, 
                        default='models/himage/model_epoch_99.pth',
                        help='模型文件路径')
    parser.add_argument('--upscale', type=int, default=2,
                        help='上采样倍数 (2, 3, 4)')
    parser.add_argument('--source', type=str, default='0',
                        help='视频源 (0=摄像头, 或视频文件路径)')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='是否使用GPU加速')
    parser.add_argument('--show-fps', action='store_true', default=True,
                        help='显示FPS')
    parser.add_argument('--output', type=str, default=None,
                        help='输出视频文件路径')
    return parser.parse_args()


def run_realtime_video(args):
    """运行实时视频超分辨率
    
    Args:
        args: 命令行参数
    """
    # 确定视频源
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # 初始化超分辨率模型
    sr = SuperResolutionModel(args.model, args.upscale, args.gpu)
    
    # 打开视频捕获
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {source}")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 计算输出尺寸
    out_width = width * args.upscale
    out_height = height * args.upscale
    
    print(f"原始分辨率: {width}x{height}")
    print(f"超分分辨率: {out_width}x{out_height}")
    print(f"FPS: {fps}")
    print("按 'q' 退出")
    
    # 创建视频写入器 (如果指定了输出)
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (out_width, out_height))
        print(f"输出视频: {args.output}")
    
    # 性能统计
    frame_count = 0
    total_time = 0
    fps_display = "FPS: --"
    
    # 窗口名称
    window_name = f'Super Resolution {args.upscale}x'
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 记录处理开始时间
        start_time = time.time()
        
        # 超分辨率处理
        sr_frame = sr.process(frame)
        
        # 计算处理时间
        process_time = time.time() - start_time
        total_time += process_time
        frame_count += 1
        
        # 计算FPS
        if frame_count % 10 == 0:
            avg_time = total_time / frame_count
            fps_display = f"FPS: {1/avg_time:.1f} | {avg_time*1000:.0f}ms"
        
        # 显示FPS
        if args.show_fps:
            cv2.putText(sr_frame, fps_display, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示分辨率信息
            res_text = f"{width}x{height} -> {out_width}x{out_height}"
            cv2.putText(sr_frame, res_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow(window_name, sr_frame)
        
        # 保存视频
        if writer:
            writer.write(sr_frame)
        
        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC
            break
    
    # 释放资源
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # 输出统计信息
    if frame_count > 0:
        print(f"\n===== 统计信息 =====")
        print(f"处理帧数: {frame_count}")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"平均处理时间: {total_time/frame_count*1000:.1f}毫秒/帧")
        print(f"平均FPS: {frame_count/total_time:.1f}")


def run_async_video(args):
    """使用异步模式运行视频超分 (降低延迟)"""
    
    # 确定视频源
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # 初始化异步超分辨率模型
    sr = AsyncSuperResolution(args.model, args.upscale, args.gpu)
    
    # 打开视频捕获
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {source}")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_width = width * args.upscale
    out_height = height * args.upscale
    
    print(f"异步模式 - 分辨率: {width}x{height} -> {out_width}x{out_height}")
    print("按 'q' 退出")
    
    window_name = f'Async Super Resolution {args.upscale}x'
    
    frame_count = 0
    display_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 提交帧进行处理
        sr.submit(frame)
        
        # 获取之前处理的结果
        result = sr.get_result()
        
        if result is not None:
            display_frame = result
            
            # 显示FPS
            if args.show_fps:
                cv2.putText(display_frame, "Async Mode", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    sr.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    run_realtime_video(args)
#!/usr/bin/env python
"""
PyTorch 模型 → ONNX 导出 → INT8 量化 一站式脚本

用法:
    # 1. 导出 ONNX FP32 + 动态量化 INT8
    python save_onnx_quantize.py export --checkpoint models/himage4/model_epoch_99.pth --upscale 4

    # 2. 只导出 ONNX FP32（不做量化）
    python save_onnx_quantize.py export --checkpoint models/himage4/model_epoch_99.pth --upscale 4 --no-quant

    # 3. 验证 ONNX 推理结果是否正确
    python save_onnx_quantize.py verify --onnx model_fp32.onnx --image image/test.png

    # 4. 对已有的 ONNX 模型做量化
    python save_onnx_quantize.py quantize --input model_fp32.onnx --output model_int8.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from model import Net


# ============================================================================
#  工具函数
# ============================================================================

def load_model(checkpoint_path: str, upscale_factor: int | None = None,
               device: torch.device = torch.device("cpu")) -> Net:
    """加载 PyTorch 模型，支持纯 state_dict 和完整 checkpoint"""
    print(f"[加载模型] {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model_state_dict', ckpt)
        # 从 conv7.weight 推断 upscale_factor
        if upscale_factor is None:
            out_channels = state_dict['conv7.weight'].shape[0]
            upscale_factor = int((out_channels // 3) ** 0.5)
            print(f"[推断] upscale_factor = {upscale_factor}")
    else:
        state_dict = ckpt.state_dict()
        # 尝试从模型对象获取
        upscale_factor = upscale_factor or getattr(ckpt, 'upscale_factor', 4)

    model = Net(upscale_factor=upscale_factor)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] 模型加载完成，upscale_factor={upscale_factor}")
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """加载图片并转为 ONNX 输入格式 (1,3,H,W) float32 [0,1]"""
    img = Image.open(image_path).convert('RGB')
    tensor = ToTensor()(img).unsqueeze(0)  # (1,3,H,W)
    return tensor.numpy().astype(np.float32)


# ============================================================================
#  导出 ONNX
# ============================================================================

def export_onnx(model: Net, output_path: str, dummy_h: int = 480,
                dummy_w: int = 640, opset: int = 17) -> str:
    """导出 PyTorch 模型为 ONNX（动态 H/W）"""
    dummy_input = torch.randn(1, 3, dummy_h, dummy_w)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {2: "height", 3: "width"},
            "output": {2: "height_out", 3: "width_out"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[ONNX] FP32 已导出 → {output_path}")
    print(f"       输入: (1,3,H,W)  H,W 动态")
    print(f"       opset: {opset}")
    return output_path


# ============================================================================
#  INT8 量化
# ============================================================================

def quantize_dynamic_onnx(input_path: str, output_path: str,
                          weight_type: str = "QInt8") -> str:
    """动态量化：仅量化 weight 为 INT8，activation 保留 FP32"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[ERROR] 请先安装 onnxruntime-quantization: pip install onnxruntime-quantization")
        sys.exit(1)

    wt = QuantType.QInt8 if weight_type.upper() == "QINT8" else QuantType.QUInt8
    quantize_dynamic(input_path, output_path, weight_type=wt)
    print(f"[量化] 动态量化完成 → {output_path}")
    print(f"       weight_type: {weight_type}")

    # 显示文件大小对比
    in_size = os.path.getsize(input_path) / 1024 / 1024
    out_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"       大小: {in_size:.2f}MB → {out_size:.2f}MB ({out_size/in_size*100:.1f}%)")
    return output_path


def quantize_static_onnx(input_path: str, output_path: str,
                         calib_images: list[str], weight_type: str = "QInt8") -> str:
    """静态量化：同时量化 weight 和 activation 为 INT8（需要校准数据）"""
    try:
        from onnxruntime.quantization import (
            quantize_static, QuantType, QuantFormat, CalibrationMethod,
            CalibrationDataReader,
        )
    except ImportError:
        print("[ERROR] 请先安装 onnxruntime-quantization: pip install onnxruntime-quantization")
        sys.exit(1)

    wt = QuantType.QInt8 if weight_type.upper() == "QINT8" else QuantType.QUInt8

    class ImageCalibReader(CalibrationDataReader):
        def __init__(self, image_paths: list[str]):
            self.data = [preprocess_image(p) for p in image_paths]
            self.iter = iter(self.data)

        def get_next(self):
            return {"input": next(self.iter, None)}

    if not calib_images:
        print("[WARN] 未提供校准图片，使用随机数据代替（精度可能较差）")
        # 生成随机校准数据
        class RandomCalibReader(CalibrationDataReader):
            def __init__(self, n: int = 50):
                self.data = [np.random.randn(1, 3, 480, 640).astype(np.float32)
                             for _ in range(n)]
                self.iter = iter(self.data)

            def get_next(self):
                return {"input": next(self.iter, None)}
        calib_reader = RandomCalibReader()
    else:
        calib_reader = ImageCalibReader(calib_images)

    quantize_static(
        input_path, output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=wt,
        weight_type=wt,
        calibration_method=CalibrationMethod.MinMax,
    )
    print(f"[量化] 静态量化完成 → {output_path}")

    in_size = os.path.getsize(input_path) / 1024 / 1024
    out_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"       大小: {in_size:.2f}MB → {out_size:.2f}MB ({out_size/in_size*100:.1f}%)")
    return output_path


# ============================================================================
#  ONNX 推理验证
# ============================================================================

def verify_onnx(onnx_path: str, image_path: str | None = None,
                use_gpu: bool = False, warmup: int = 10, benchmark: int = 100):
    """验证 ONNX 模型推理，并跑 benchmark"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] 请先安装 onnxruntime: pip install onnxruntime-gpu")
        sys.exit(1)

    # 准备输入
    if image_path and os.path.exists(image_path):
        input_data = preprocess_image(image_path)
        print(f"[输入] 加载图片: {image_path} → {input_data.shape}")
    else:
        input_data = np.random.randn(1, 3, 480, 640).astype(np.float32)
        print(f"[输入] 使用随机数据: {input_data.shape}")

    # 配置 Session
    providers = []
    if use_gpu:
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
            }),
            "CPUExecutionProvider",
        ]
        print("[后端] CUDAExecutionProvider + CPUExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
        print("[后端] CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_profiling = False

    session = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                    providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 打印模型信息
    print(f"\n[模型信息]")
    print(f"  输入: {session.get_inputs()[0]}")
    print(f"  输出: {session.get_outputs()[0]}")
    for p in session.get_providers():
        print(f"  提供者: {p}")

    # 推理一次验证
    result = session.run([output_name], {input_name: input_data})[0]
    print(f"\n[推理] 输出形状: {result.shape}")
    print(f"       数值范围: [{result.min():.4f}, {result.max():.4f}]")
    print(f"       均值: {result.mean():.6f}")

    # Benchmark
    if benchmark > 0:
        # Warmup
        for _ in range(warmup):
            session.run([output_name], {input_name: input_data})

        # Benchmark
        torch.cuda.synchronize() if use_gpu else None
        start = time.perf_counter()
        for _ in range(benchmark):
            session.run([output_name], {input_name: input_data})
        torch.cuda.synchronize() if use_gpu else None
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / benchmark * 1000
        fps = benchmark / elapsed
        print(f"\n[基准测试] {benchmark} 次推理")
        print(f"  总耗时: {elapsed:.3f}s")
        print(f"  平均耗时: {avg_ms:.2f}ms")
        print(f"  FPS: {fps:.1f}")

    # 如果是图片，保存输出
    if image_path and os.path.exists(image_path):
        out_img = result[0].transpose(1, 2, 0)  # CHW → HWC
        out_img = (out_img * 255.0).clip(0, 255).astype(np.uint8)
        save_path = onnx_path.replace('.onnx', '_output.png')
        Image.fromarray(out_img).save(save_path)
        print(f"\n[输出] 已保存: {save_path}")


# ============================================================================
#  与 PyTorch 推理对比
# ============================================================================

def compare_torch_onnx(checkpoint_path: str, onnx_path: str,
                       upscale_factor: int, image_path: str | None = None,
                       use_gpu: bool = False):
    """对比 PyTorch 与 ONNX Runtime 的推理结果"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] 请先安装 onnxruntime")
        return

    # 加载 PyTorch 模型
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, upscale_factor, device=device)

    # 准备输入
    if image_path and os.path.exists(image_path):
        input_data = preprocess_image(image_path)
    else:
        input_data = np.random.randn(1, 3, 480, 640).astype(np.float32)

    input_torch = torch.from_numpy(input_data).to(device)

    # PyTorch 推理
    with torch.no_grad():
        out_torch = model(input_torch).cpu().numpy()

    # ONNX Runtime 推理
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    out_onnx = session.run(None, {input_name: input_data})[0]

    # 比较
    diff = np.abs(out_torch - out_onnx)
    print(f"\n[对比] PyTorch vs ONNX Runtime")
    print(f"  PyTorch 输出形状: {out_torch.shape}")
    print(f"  ONNX 输出形状:    {out_onnx.shape}")
    print(f"  Max Abs Diff:      {diff.max():.6f}")
    print(f"  Mean Abs Diff:     {diff.mean():.6f}")
    print(f"  Max Rel Diff:      {(diff / (np.abs(out_torch) + 1e-8)).max():.6f}")

    # 保存对比图
    if image_path:
        out_t_img = (out_torch[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        out_o_img = (out_onnx[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(out_t_img).save(onnx_path.replace('.onnx', '_torch_output.png'))
        Image.fromarray(out_o_img).save(onnx_path.replace('.onnx', '_onnx_output.png'))
        print(f"  对比图已保存")


# ============================================================================
#  命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch → ONNX → INT8 量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- export ----
    p_export = sub.add_parser("export", help="导出 ONNX + 可选量化")
    p_export.add_argument("--checkpoint", required=True, help="PyTorch checkpoint 路径")
    p_export.add_argument("--upscale", type=int, default=None, help="上采样倍数（省略则自动推断）")
    p_export.add_argument("--output-dir", default=".", help="输出目录")
    p_export.add_argument("--no-quant", action="store_true", help="不执行量化，只导出 FP32")
    p_export.add_argument("--static", action="store_true", help="使用静态量化（需校准图片）")
    p_export.add_argument("--calib-images", nargs="*", default=[], help="静态量化校准图片路径")
    p_export.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")

    # ---- quantize ----
    p_quant = sub.add_parser("quantize", help="对已有的 ONNX 模型做量化")
    p_quant.add_argument("--input", required=True, help="输入 ONNX FP32 模型路径")
    p_quant.add_argument("--output", default=None, help="输出量化模型路径")
    p_quant.add_argument("--static", action="store_true", help="使用静态量化")
    p_quant.add_argument("--calib-images", nargs="*", default=[], help="静态量化校准图片路径")

    # ---- verify ----
    p_verify = sub.add_parser("verify", help="验证 ONNX 推理")
    p_verify.add_argument("--onnx", required=True, help="ONNX 模型路径")
    p_verify.add_argument("--image", default=None, help="测试图片路径")
    p_verify.add_argument("--gpu", action="store_true", help="使用 GPU 推理")
    p_verify.add_argument("--warmup", type=int, default=10, help="预热次数")
    p_verify.add_argument("--benchmark", type=int, default=100, help="基准测试次数")

    # ---- compare ----
    p_cmp = sub.add_parser("compare", help="对比 PyTorch vs ONNX 推理结果")
    p_cmp.add_argument("--checkpoint", required=True, help="PyTorch checkpoint 路径")
    p_cmp.add_argument("--onnx", required=True, help="ONNX 模型路径")
    p_cmp.add_argument("--upscale", type=int, default=None, help="上采样倍数")
    p_cmp.add_argument("--image", default=None, help="测试图片路径")
    p_cmp.add_argument("--gpu", action="store_true", help="使用 GPU")

    args = parser.parse_args()

    if args.command == "export":
        model = load_model(args.checkpoint, args.upscale)
        os.makedirs(args.output_dir, exist_ok=True)

        # 获取 upscale_factor
        upscale = args.upscale
        if upscale is None:
            state_dict = model.state_dict()
            out_channels = state_dict['conv7.weight'].shape[0]
            upscale = int((out_channels // 3) ** 0.5)

        base_name = f"model_{upscale}x"
        onnx_fp32 = os.path.join(args.output_dir, f"{base_name}.onnx")
        onnx_int8 = os.path.join(args.output_dir, f"{base_name}_int8.onnx")

        # 导出 FP32
        export_onnx(model, onnx_fp32, opset=args.opset)

        # 量化
        if not args.no_quant:
            if args.static:
                quantize_static_onnx(onnx_fp32, onnx_int8, args.calib_images)
            else:
                quantize_dynamic_onnx(onnx_fp32, onnx_int8)
        else:
            print("[跳过] 未执行量化")

    elif args.command == "quantize":
        output = args.output or args.input.replace('.onnx', '_int8.onnx')
        if args.static:
            quantize_static_onnx(args.input, output, args.calib_images)
        else:
            quantize_dynamic_onnx(args.input, output)

    elif args.command == "verify":
        verify_onnx(args.onnx, args.image, args.gpu, args.warmup, args.benchmark)

    elif args.command == "compare":
        if not args.upscale:
            # 从 checkpoint 推断
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state = ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt)
            args.upscale = int((state['conv7.weight'].shape[0] // 3) ** 0.5)
            print(f"[推断] upscale_factor = {args.upscale}")
        compare_torch_onnx(args.checkpoint, args.onnx, args.upscale, args.image, args.gpu)


if __name__ == '__main__':
    main()

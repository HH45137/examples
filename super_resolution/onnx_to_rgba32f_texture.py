"""
onnx_to_rgba32f_texture.py
从 ONNX 模型提取所有 Conv 层的 weight + bias，打包为单个 RGBA32F 纹理
输出：
  - weights_rgba32f.raw      原始 float32 像素数据（可直接上传 GPU）
  - weights_metadata.json    每层的偏移量和形状信息
"""

import numpy as np
import onnx
import json
import struct
from pathlib import Path


def extract_conv_weights(onnx_path: str) -> list[dict]:
    """
    从 ONNX 模型中提取所有 Conv 算子的 weight 和 bias
    返回列表，每个元素包含 name, weight, bias, shape
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # 建立 name → initializer 的映射
    initializers = {init.name: init for init in graph.initializer}
    
    # 找到所有 Conv 节点
    conv_layers = []
    for node in graph.node:
        if node.op_type != 'Conv':
            continue
        
        weight_name = node.input[1]  # Conv 的第二个输入是 weight
        bias_name = node.input[2] if len(node.input) > 2 else None
        
        weight_init = initializers[weight_name]
        weight_data = onnx.numpy_helper.to_array(weight_init).astype(np.float32)
        
        bias_data = None
        if bias_name and bias_name in initializers:
            bias_init = initializers[bias_name]
            bias_data = onnx.numpy_helper.to_array(bias_init).astype(np.float32)
        
        conv_layers.append({
            'name': node.name or weight_name,
            'weight_shape': tuple(weight_data.shape),
            'weight': weight_data.flatten(),
            'bias_shape': tuple(bias_data.shape) if bias_data is not None else None,
            'bias': bias_data.flatten() if bias_data is not None else None,
        })
        print(f"  [提取] {node.name}: weight {weight_data.shape}, bias {bias_data.shape if bias_data is not None else 'None'}")
    
    return conv_layers


def pack_to_rgba32f(layers: list[dict], tex_width: int = 512, tex_height: int = 256) -> tuple[np.ndarray, list[dict]]:
    """
    将所有层的 weight + bias 按顺序打包进 RGBA32F 纹理
    纹理数据格式: (height, width, 4) float32，R=chan0, G=chan1, B=chan2, A=chan3
    """
    total_pixels = tex_width * tex_height
    tex_data = np.zeros((tex_height, tex_width, 4), dtype=np.float32)
    
    # 将所有浮点数拼接为一个长向量
    all_floats = []
    metadata = []
    current_offset = 0  # 以 float 为单位的偏移
    
    for i, layer in enumerate(layers):
        meta = {
            'index': i,
            'name': layer['name'],
            'weight_shape': layer['weight_shape'],
            'bias_shape': layer['bias_shape'],
            'weight_offset_float': current_offset,
            'weight_count': len(layer['weight']),
        }
        
        all_floats.extend(layer['weight'].tolist())
        current_offset += len(layer['weight'])
        
        if layer['bias'] is not None:
            meta['bias_offset_float'] = current_offset
            meta['bias_count'] = len(layer['bias'])
            all_floats.extend(layer['bias'].tolist())
            current_offset += len(layer['bias'])
        else:
            meta['bias_offset_float'] = -1
            meta['bias_count'] = 0
        
        metadata.append(meta)
    
    # 填充到 RGBA 像素
    num_floats = len(all_floats)
    num_pixels_needed = (num_floats + 3) // 4  # 向上取整
    
    if num_pixels_needed > total_pixels:
        raise ValueError(f"纹理太小! 需要 {num_pixels_needed} 像素，当前 {total_pixels} 像素 "
                         f"({tex_width}×{tex_height})")
    
    for pixel_idx in range(num_pixels_needed):
        y = pixel_idx // tex_width
        x = pixel_idx % tex_width
        base = pixel_idx * 4
        for c in range(4):
            idx = base + c
            if idx < num_floats:
                tex_data[y, x, c] = all_floats[idx]
            else:
                tex_data[y, x, c] = 0.0  # padding
    
    print(f"\n[打包完成]")
    print(f"  总 float 数: {num_floats}")
    print(f"  占用像素数: {num_pixels_needed} / {total_pixels}")
    print(f"  纹理尺寸: {tex_width}×{tex_height}")
    print(f"  利用率: {num_pixels_needed/total_pixels*100:.1f}%")
    
    return tex_data, metadata


def save_outputs(tex_data: np.ndarray, metadata: list[dict], 
                 raw_path: str = "weights_rgba32f.raw",
                 meta_path: str = "weights_metadata.json"):
    """保存纹理数据和元数据"""
    # 保存 raw 二进制（可直接上传到 GPU）
    # 布局: RRRR...GGGG...BBBB...AAAA... (channel-planar) 
    # 或 R,G,B,A,R,G,B,A... (interleaved)
    # 这里用 interleaved，兼容大多数图形 API 的上传格式
    with open(raw_path, 'wb') as f:
        tex_data.astype(np.float32).tofile(f)
    print(f"\n[保存] 纹理原始数据 → {raw_path} ({tex_data.nbytes} bytes)")
    
    # 计算像素偏移（供 shader 使用）
    for meta in metadata:
        meta['weight_offset_pixel'] = meta['weight_offset_float'] // 4
        meta['weight_offset_subpixel'] = meta['weight_offset_float'] % 4
        if meta['bias_offset_float'] >= 0:
            meta['bias_offset_pixel'] = meta['bias_offset_float'] // 4
            meta['bias_offset_subpixel'] = meta['bias_offset_float'] % 4
    
    total_meta = {
        'texture_width': tex_data.shape[1],
        'texture_height': tex_data.shape[0],
        'format': 'RGBA32F',
        'pixel_layout': 'row-major, interleaved RGBA',
        'layers': metadata,
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(total_meta, f, indent=2, ensure_ascii=False)
    print(f"[保存] 元数据 → {meta_path}")


def save_preview_png(tex_data: np.ndarray, png_path: str = "weights_preview.png"):
    """
    将 RGBA32F 纹理数据归一化后保存为 PNG 预览图
    - 四通道以 2×2 网格展示（R左上 / G右上 / B左下 / A右下）
    - 使用 1%~99% 百分位裁剪，避免极端值冲淡对比度
    - 仅供肉眼查看纹理分布，不保证数值精度
    """
    try:
        from PIL import Image
    except ImportError:
        print("[跳过] 未安装 Pillow，无法生成 PNG 预览 (pip install Pillow)")
        return

    H, W, C = tex_data.shape  # (256, 512, 4)

    flat = tex_data.flatten()
    vmin = float(np.percentile(flat, 1))
    vmax = float(np.percentile(flat, 99))
    print(f"\n[PNG预览] 全局数值范围: [{flat.min():.4f}, {flat.max():.4f}]")
    print(f"          裁剪到 1%~99%:  [{vmin:.4f}, {vmax:.4f}]")

    def norm(ch: np.ndarray) -> np.ndarray:
        return np.clip((ch - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    # 2×2 网格布局
    canvas = np.zeros((H * 2, W * 2), dtype=np.float32)
    canvas[0:H,   0:W]   = norm(tex_data[:, :, 0])   # R 通道
    canvas[0:H,   W:W*2] = norm(tex_data[:, :, 1])   # G 通道
    canvas[H:H*2, 0:W]   = norm(tex_data[:, :, 2])   # B 通道
    canvas[H:H*2, W:W*2] = norm(tex_data[:, :, 3])   # A 通道

    # 十字分隔线
    canvas[H-1:H+1, :] = 1.0
    canvas[:, W-1:W+1] = 1.0

    img = (canvas * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(png_path)
    print(f"[保存] PNG 预览 → {png_path} ({W*2}×{H*2})")


def main():
    onnx_path = "model_3x.onnx"
    tex_width = 512
    tex_height = 256
    
    print(f"[读取] ONNX 模型: {onnx_path}")
    layers = extract_conv_weights(onnx_path)
    
    print(f"\n[打包] 目标纹理: {tex_width}×{tex_height} RGBA32F")
    tex_data, metadata = pack_to_rgba32f(layers, tex_width, tex_height)
    
    save_outputs(tex_data, metadata)
    save_preview_png(tex_data)


if __name__ == '__main__':
    main()

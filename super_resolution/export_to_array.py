import torch
import numpy as np
from model import Net

# 加载训练好的模型
model = Net(upscale_factor=3)
checkpoint = torch.load(r'models\himage5\model_epoch_100.pth', weights_only=False)
model.load_state_dict(checkpoint.state_dict())
model.eval()

# 转换为C数组并保存
output_file = open("weights.h", "w")

# 卷积层权重形状: (out_channels, in_channels, height, width)
# 偏置形状: (out_channels,)

layer_index = 1
for name, param in model.state_dict().items():
    if 'weight' in name:
        shape = param.shape
        data = param.detach().numpy().flatten()
        # 转为float32并量化（如果需要定点化）
        data = data.astype(np.float32)
        
        output_file.write(f"// {name}, shape: {shape}\n")
        output_file.write(f"const float conv{layer_index}_weight[] = " + "{\n")
        
        # 按PyTorch顺序输出: (out, in, h, w)
        for i, v in enumerate(data):
            if i % 16 == 0:
                output_file.write("\n    ")
            output_file.write(f"{v:.4f}f, ")
        
        output_file.write("\n};\n\n")
        
        # 保存对应的bias（如果存在）
        bias_name = name.replace('weight', 'bias')
        if bias_name in model.state_dict():
            bias_data = model.state_dict()[bias_name].detach().numpy()
            output_file.write(f"const float conv{layer_index}_bias[] = " + "{")
            for v in bias_data:
                output_file.write(f"{v:.4f}f, ")
            output_file.write("};\n\n")
        
        layer_index += 1

output_file.close()
print("权重已导出到 weights.h")

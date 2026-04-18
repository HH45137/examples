from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from model import Net

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--accel', action='store_true', help='Enables acceleration device, if available')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image).convert('RGB')

with open(opt.model, 'rb') as f:
    safe_globals = [
        Net,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.pixelshuffle.PixelShuffle,
    ]
    with torch.serialization.safe_globals(safe_globals):
        model = torch.load(f, weights_only=False)

img_to_tensor = ToTensor()
input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0])

if opt.accel:
    device = torch.accelerator.current_accelerator()
    model = model.to(device)
    input = input.to(device)

out = model(input)
out = out.cpu()

out_img = out[0].detach().numpy()
out_img = (out_img.transpose(1, 2, 0) * 255.0).clip(0, 255)
out_img = Image.fromarray(np.uint8(out_img))

out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)

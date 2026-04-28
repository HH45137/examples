# main.py (修改后)

from __future__ import print_function
import argparse
import os
import shutil
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--accel', action='store_true', help='Enables acceleration for training, if available')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset_root_dir', type=str, default=None)
parser.add_argument('--models_root_dir', type=str, default=None)
# === 新增参数 ===
parser.add_argument('--resume', type=str, default=None, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
opt = parser.parse_args()

print(opt)


def save_checkpoint(state, is_best, filename):
    """保存完整训练状态"""
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'model_best.pth')
        shutil.copyfile(filename, best_filename)
        print(f"===> New best model saved to {best_filename}")


torch.manual_seed(opt.seed)

if opt.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print('===> Loading datasets')
if opt.dataset_root_dir is None:
    print('Not found dataset!')
    exit(0)
train_set = get_training_set(opt.upscale_factor, opt.dataset_root_dir)
test_set = get_test_set(opt.upscale_factor, opt.dataset_root_dir)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# === 恢复逻辑 ===
start_epoch = 1
best_psnr = 0

if opt.resume:
    if os.path.isfile(opt.resume):
        print(f"===> Loading checkpoint '{opt.resume}'")
        checkpoint = torch.load(opt.resume, map_location=device)
        
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"===> Loaded checkpoint (epoch {checkpoint['epoch']})")
        print(f"===> Best PSNR so far: {best_psnr:.4f} dB")
    else:
        print(f"===> No checkpoint found at '{opt.resume}', starting from scratch")


def train(epoch):
    model.train()
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    avg_loss = epoch_loss / len(training_data_loader)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss))
    return avg_loss


def test():
    model.eval()
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    avg_psnr = avg_psnr / len(testing_data_loader)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    return avg_psnr


if __name__ == '__main__':
    for epoch in range(start_epoch, opt.nEpochs + 1):
        avg_loss = train(epoch)
        current_psnr = test()
        
        # 更新 best PSNR
        is_best = current_psnr > best_psnr
        if is_best:
            best_psnr = current_psnr
        
        # 保存 checkpoint
        if opt.models_root_dir is not None:
            checkpoint_path = f"{opt.models_root_dir}/checkpoint_epoch_{epoch}.pth"
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': current_psnr,
                'best_psnr': best_psnr,
            }, is_best=is_best, filename=checkpoint_path)
            
            # 始终保存一个 latest checkpoint 方便快速恢复
            latest_path = f"{opt.models_root_dir}/checkpoint_latest.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': current_psnr,
                'best_psnr': best_psnr,
            }, latest_path)
            print(f"Checkpoint saved to {checkpoint_path}")

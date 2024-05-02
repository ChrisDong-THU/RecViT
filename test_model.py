import torch
import numpy as np

from model import RecFieldViT

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Available CUDA device count: {device_count}")
    for i in range(device_count):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices are available.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RecFieldViT().to(device)

# 创建模拟输入数据
input_tensor = torch.randn(8, 1, 112, 192)  # [batch_size, height, width]
input_tensor = input_tensor.to(device)

# 执行前向传播
output = model(input_tensor)

# 检查输出
print(output.shape)
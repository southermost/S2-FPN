import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from nets.A2FPN_efficient_me import A2FPN
import os

# 初始化模型
model = A2FPN(class_num=4)

# 准备模型进行量化
model.eval()  # 将模型设为评估模式
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 选择量化配置
torch.quantization.prepare(model, inplace=True)  # 准备模型进行量化

# 假设有一个校准数据集 calibration_data
# 校准过程（需要一定的样本数据）
# for data in calibration_data:
#     model(data)  # 假设 calibration_data 是 DataLoader

# 完成量化
torch.quantization.convert(model, inplace=True)

# 保存量化后的模型
quantized_model_path = 'quantized_static_model.pth'
torch.save(model.state_dict(), quantized_model_path)

# 计算量化后模型文件大小（MB）
model_size_MB = os.path.getsize(quantized_model_path) / (1024 * 1024)
print(f"静态量化后模型文件大小: {model_size_MB:.2f} MB")

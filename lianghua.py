import torch
import os
import torch.quantization
from nets.A2FPN_efficient_me import A2FPN

# 加载训练好的模型
model = A2FPN(class_num=4)
model.load_state_dict(torch.load('F:/unet-pytorch-main/last_epoch_miou89.170.pth'))

# 设置为评估模式
model.eval()

# 设置量化配置
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 假设您有一个校准数据集 `calibration_data`（需要用来估算激活层的范围）
# 这里可以使用一个小批量数据来校准
# for data in calibration_data:
#     model(data)

# 完成量化
torch.quantization.convert(model, inplace=True)

# 保存量化后的模型
quantized_model_path = 'quantized_static_model.pth'
torch.save(model.state_dict(), quantized_model_path)

# 计算量化后模型文件大小（MB）
model_size_MB = os.path.getsize(quantized_model_path) / (1024 * 1024)
print(f"静态量化后模型文件大小: {model_size_MB:.2f} MB")


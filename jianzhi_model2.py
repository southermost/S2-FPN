import torch
from nets.A2FPN_efficient_me import A2FPN
import torch_pruning as tp
from utils.jianzhi import MyMagnitudeImportance


model = A2FPN(class_num=4)

example_inputs = torch.randn(1, 3, 256, 256)

# 1. 使用我们上述定义的重要性评估
imp = MyMagnitudeImportance()

# 2. 忽略无需剪枝的层，例如最后的分类层
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

# 3. 初始化剪枝器
iterative_steps = 5 # 迭代式剪枝，重复5次Pruning-Finetuning的循环完成剪枝。
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs, # 用于分析依赖的伪输入
    importance=imp, # 重要性评估指标
    iterative_steps=iterative_steps, # 迭代剪枝，设为1则一次性完成剪枝
    ch_sparsity=0.3, # 目标稀疏性，这里我们移除50%的通道 ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers, # 忽略掉最后的分类层
)

# 4. Pruning-Finetuning的循环
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    pruner.step() # 执行裁剪，本例子中我们每次会裁剪10%，共执行5次，最终稀疏度为50%
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("  Iter %d/%d, Params: %.2f M => %.2f M" % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6))
    print("  Iter %d/%d, MACs: %.2f G => %.2f G"% (i+1, iterative_steps, base_macs / 1e9, macs / 1e9))
    # finetune your model here
    # finetune(model)
    # ...
print(model)
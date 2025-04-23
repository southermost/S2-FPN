import torch
import torch_pruning as tp
from utils.jianzhi import MySlimmingImportance
from utils.jianzhi import MySlimmingPruner
from nets.A2FPN_efficient import A2FPN
model = A2FPN(class_num=4)
example_inputs = torch.randn(1, 3, 256, 256)

# 1. 使用我们上述定义的重要性评估
imp = MySlimmingImportance()

# 2. 忽略无需剪枝的层，例如最后的分类层
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

# 3. 初始化剪枝器
iterative_steps = 1 # 迭代式剪枝，本案例中我们一次剪枝到50%稀疏性
pruner = MySlimmingPruner(
    model,
    example_inputs, # 用于分析依赖的伪输入
    importance=imp, # 重要性评估指标
    iterative_steps=iterative_steps, # 迭代剪枝，设为1则一次性完成剪枝
    ch_sparsity=0.3, # 目标稀疏性，这里我们移除50%的通道 ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers, # 忽略掉最后的分类层
)

# 4. 稀疏训练（为了节省时间我们假装在训练，实际应用时只需要在optimizer.step前插入regularize即可）
for _ in range(100):
    # optimizer.zero_grad()
    # ...
    # loss.backward()
    pruner.regularize(model, reg=1e-5) # <== 插入该行进行稀疏化
    # optimizer.step()

# 5. Finetuning
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
pruner.step() # 执行裁剪，1次达到50%稀疏度（iterative_steps=1）
macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
print("Params: {:.2f} M => {:.2f} M".format(base_nparams / 1e6, nparams / 1e6))
print("MACs: {:.2f} G => {:.2f} G".format(base_macs / 1e9, macs / 1e9))
# finetune your model here
# finetune(model)
# ...

torch.save(model.state_dict(), 'Slim_medel.pth')
print(model)
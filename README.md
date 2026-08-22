# AEA-Net 纯净实现（中文说明）

本仓库提供 **AEA-Net：基于双向多层特征交互与注意力引导视图增强的细粒度视觉分类方法** 的纯净、可迁移实现。

仓库只保留完整 AEA-Net 的训练与评估路径，不包含数据集、ConvNeXt 预训练权重、AEA-Net 训练权重或历史实验结果。使用者准备好自己的分类数据集和兼容的 ConvNeXt-Base 权重后，即可在新数据集上训练。

## 方法组成

AEA-Net 以 ConvNeXt-Base 为特征提取骨干，并加入以下部分：

1. 上下文引导的自顶向下多层特征融合；
2. 自底向上的通道注意力传播；
3. 原始深层特征与增强深层特征之间的高效双线性池化；
4. 响应图引导的关键区域裁剪放大、局部水平翻转和高响应区域遮挡；
5. 根据各视图历史优化状态动态调整原图和三个增强视图的分类损失权重。

三种自适应视图只在训练阶段使用。验证和测试阶段只使用原始图像分支，不引入测试时数据增强。

## 目录结构

```text
AEANet-pure-zh/
├── aeanet/
│   ├── augment.py          # 注意力引导的三种视图生成策略
│   ├── data.py             # 通用 ImageFolder 数据读取
│   ├── engine.py           # 完整训练与评估循环
│   ├── loss_balancer.py    # 四视图动态损失权重
│   ├── model.py            # ConvNeXt-Base 与 AEA-Net 主模型
│   ├── modules.py          # 特征交互、注意力传播和 EBP 模块
│   └── __init__.py
├── configs/
│   └── seed0.json          # seed0 复现配置，不含本机路径和权重
├── train.py                # 训练入口
├── evaluate.py             # 独立评估入口
├── smoke_test.py           # 无数据、无权重的完整路径检查
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 3.9 或更高版本。首先根据本机 CUDA 版本安装匹配的 PyTorch 和 torchvision，然后执行：

```bash
pip install -r requirements.txt
```

主要依赖包括 PyTorch、torchvision、timm、NumPy、scikit-learn 和 Pillow。

## 数据集目录

数据集采用 torchvision 的 `ImageFolder` 目录形式：

```text
DATASET_ROOT/
├── train/
│   ├── class_000/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── class_001/
├── val/                    # 可选
│   ├── class_000/
│   └── class_001/
└── test/
    ├── class_000/
    └── class_001/
```

各数据划分中的类别文件夹名称和类别映射必须一致。

如果存在 `val/`，程序会直接使用它作为验证集。如果没有 `val/`，程序会按照类别进行分层抽样，从 `train/` 中确定性地划分验证集。划分比例由 `--val-ratio` 控制，划分随机种子由 `--split-seed` 控制。

## ConvNeXt-Base 预训练权重

本仓库不附带任何预训练文件。请单独下载兼容的 ConvNeXt-Base ImageNet 预训练权重，并通过 `--pretrained` 指定路径。

权重文件可以直接保存 ConvNeXt 状态字典，也可以将状态字典放在 `model` 字段下。加载时会自动忽略 ConvNeXt 原始分类头，因为 AEA-Net 会根据新数据集类别数建立新的分类头。

不提供 `--pretrained` 也可以从随机初始化开始训练，但对于常见的细粒度小样本数据集，一般建议使用预训练初始化。

## 在新数据集上训练

仓库提供了不含路径和权重的 seed0 配置：

```bash
python train.py \
  --config configs/seed0.json \
  --data-root /path/to/DATASET_ROOT \
  --dataset-name my_dataset \
  --pretrained /path/to/convnext_base_checkpoint.pth \
  --amp
```

命令行参数会覆盖 JSON 配置中的同名参数。如果不使用预训练权重，删除 `--pretrained` 即可。

seed0 配置的默认设置为：

- 输入尺寸：224 × 224；
- batch size：8；
- 优化器：AdamW；
- 峰值学习率：`4e-3`；
- 权重衰减：`0.05`；
- 训练轮数：150；
- warm-up：20轮；
- 训练随机种子：0；
- 数据划分随机种子：2026；
- 最优模型选择指标：验证集 Macro-F1。

如果希望为已预训练的 ConvNeXt 骨干设置较小学习率，可额外指定：

```bash
--backbone-lr 1e-5
```

该参数是可选项，没有固化在 seed0 配置中。

## 训练输出

每次训练会在 `--output-dir` 中生成：

- `config.json`：本次实际训练参数；
- `split_manifest.json`：类别、种子和数据集规模信息；
- `metrics.jsonl`：逐轮训练和验证指标；
- `checkpoint_last.pth`：最后一轮训练状态；
- `checkpoint_best.pth`：验证指标最优的训练状态；
- `test_metrics.json`：最终测试结果。

测试结果包括 Accuracy、Macro-Precision、Macro-Recall、Macro-F1、Weighted-F1、Balanced Accuracy、各类别召回率、混淆矩阵和逐图片预测结果。

## 断点恢复

```bash
python train.py \
  --config configs/seed0.json \
  --data-root /path/to/DATASET_ROOT \
  --resume runs/seed0/checkpoint_last.pth \
  --amp
```

恢复训练时应继续使用相同的数据集、划分种子、优化器设置和输出目录。

## 独立评估

```bash
python evaluate.py \
  --data-root /path/to/DATASET_ROOT \
  --checkpoint runs/seed0/checkpoint_best.pth \
  --amp \
  --output runs/seed0/evaluation.json
```

## 在 Python 中调用模型

```python
from aeanet import convnext_base_aeanet

model = convnext_base_aeanet(
    num_classes=100,
    attention_channels=16,
)

missing, unexpected = model.load_convnext_pretrained(
    "/path/to/convnext_base_checkpoint.pth"
)
```

训练阶段可使用：

```python
logits, response_maps = model(images, return_attention=True)
```

普通验证和推理阶段使用：

```python
logits, _ = model(images, return_attention=False)
```

## 快速完整性检查

`smoke_test.py` 使用缩小的合成网络，不需要数据集和任何权重：

```bash
python smoke_test.py
```

该测试会依次检查多层特征交互、注意力传播、EBP、三种自适应视图、四个分类损失、动态损失权重和反向传播。

## 可复现性说明

- `configs/seed0.json` 将训练随机种子设为0，将确定性数据划分种子设为2026；
- Python、NumPy、PyTorch 和 CUDA 随机数生成器均会设置种子；
- 关闭 cuDNN benchmark，并启用确定性模式；
- 仓库的 `.gitignore` 会排除数据集、训练输出和所有常见权重文件；
- 本仓库不保存 AEA-Net 训练参数和 ConvNeXt 预训练参数。

## 引用

如果本实现对你的研究有帮助，请引用对应的 AEA-Net 论文。论文正式发表后，可以在此处补充最终 BibTeX 信息。

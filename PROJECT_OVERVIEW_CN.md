# Segmentation Models PyTorch 项目详细说明

## 项目概述

**Segmentation Models PyTorch (SMP)** 是一个基于 PyTorch 的图像语义分割神经网络库。这是一个功能强大的开源 Python 库，专门用于图像分割任务。

### 核心功能

1. **简单易用的高级 API**：仅需两行代码即可创建神经网络模型
2. **12 种编码器-解码器架构**：包括 Unet、Unet++、Segformer、DPT 等
3. **800+ 预训练编码器**：支持卷积和 Transformer 架构，包括 timm 库支持
4. **完整的训练工具**：提供常用的损失函数和评估指标（Dice、Jaccard、Tversky 等）
5. **生产就绪**：支持 ONNX 导出、torch script/trace/compile

## 项目用途

这个项目主要用于：

- **医学图像分割**：分割 CT、MRI 等医学图像中的器官、病灶等
- **遥感图像分析**：卫星图像中的建筑物、道路、植被分割
- **自动驾驶**：道路场景分割，识别车辆、行人、道路等
- **工业检测**：产品缺陷检测和分割
- **背景移除**：图像抠图和背景替换
- **任意图像分割任务**：只要有像素级标注数据

## 项目文件组织结构

```
segmentation_models.pytorch/
│
├── segmentation_models_pytorch/    # 核心库代码
│   ├── __init__.py                 # 库入口，导出所有模型
│   ├── __version__.py              # 版本信息
│   │
│   ├── encoders/                   # 编码器（骨干网络）
│   │   ├── resnet.py              # ResNet 系列编码器
│   │   ├── efficientnet.py        # EfficientNet 系列
│   │   ├── mobilenet.py           # MobileNet 系列（轻量级）
│   │   ├── densenet.py            # DenseNet 系列
│   │   ├── vgg.py                 # VGG 系列
│   │   ├── mix_transformer.py     # Mix Transformer（用于 Segformer）
│   │   ├── timm_*.py              # timm 库支持的编码器
│   │   └── ...                    # 其他编码器实现
│   │
│   ├── decoders/                  # 解码器（分割模型架构）
│   │   ├── unet/                  # U-Net 模型
│   │   ├── unetplusplus/          # U-Net++ 模型
│   │   ├── fpn/                   # Feature Pyramid Network
│   │   ├── pspnet/                # Pyramid Scene Parsing Network
│   │   ├── deeplabv3/             # DeepLabV3/V3+ 模型
│   │   ├── linknet/               # LinkNet 模型
│   │   ├── manet/                 # MA-Net 模型
│   │   ├── pan/                   # Pyramid Attention Network
│   │   ├── upernet/               # UPerNet 模型
│   │   ├── segformer/             # Segformer 模型
│   │   └── dpt/                   # Dense Prediction Transformer
│   │
│   ├── losses/                    # 损失函数
│   │   ├── dice.py               # Dice Loss（分割常用）
│   │   ├── jaccard.py            # Jaccard Loss (IoU Loss)
│   │   ├── focal.py              # Focal Loss（处理类别不平衡）
│   │   ├── tversky.py            # Tversky Loss
│   │   ├── lovasz.py             # Lovasz Loss
│   │   ├── soft_bce.py           # Soft Binary Cross Entropy
│   │   ├── soft_ce.py            # Soft Cross Entropy
│   │   └── mcc.py                # Matthews Correlation Coefficient
│   │
│   ├── metrics/                   # 评估指标
│   │   └── functional.py         # IoU、F-score 等评估函数
│   │
│   ├── base/                      # 基础类和工具
│   │   ├── model.py              # 基础模型类
│   │   └── hub_mixin.py          # HuggingFace Hub 集成
│   │
│   ├── datasets/                  # 示例数据集
│   │   └── ...                   # 数据集加载工具
│   │
│   └── utils/                     # 工具函数（已弃用）
│       └── ...                   # 历史遗留的工具代码
│
├── examples/                       # 使用示例
│   ├── binary_segmentation_intro.ipynb          # 二分类分割入门
│   ├── camvid_segmentation_multiclass.ipynb     # 多类别分割
│   ├── cars segmentation (camvid).ipynb         # 汽车分割示例
│   ├── convert_to_onnx.ipynb                    # ONNX 导出
│   ├── segformer_inference_pretrained.ipynb     # Segformer 推理
│   ├── dpt_inference_pretrained.ipynb           # DPT 推理
│   ├── upernet_inference_pretrained.ipynb       # UPerNet 推理
│   └── save_load_model_and_share_with_hf_hub.ipynb  # 模型保存和分享
│
├── tests/                          # 单元测试
│   ├── models/                    # 模型测试
│   ├── encoders/                  # 编码器测试
│   ├── base/                      # 基础功能测试
│   └── ...
│
├── docs/                           # 文档源文件
│   ├── index.rst                  # 文档首页
│   ├── models.rst                 # 模型文档
│   ├── encoders.rst               # 编码器文档
│   ├── losses.rst                 # 损失函数文档
│   ├── metrics.rst                # 指标文档
│   └── ...
│
├── scripts/                        # 辅助脚本
│   └── models-conversions/        # 模型转换脚本
│
├── misc/                           # 杂项工具
│   ├── generate_table.py         # 生成编码器表格
│   └── ...
│
├── requirements/                   # 依赖配置
│
├── licenses/                       # 许可证信息
│
├── pics/                          # 项目图片资源
│
├── README.md                      # 项目说明（英文）
├── pyproject.toml                 # 项目配置和依赖
├── LICENSE                        # MIT 许可证
├── HALLOFFAME.md                  # 使用该库获奖的竞赛列表
├── Makefile                       # 开发命令快捷方式
└── .readthedocs.yaml              # Read the Docs 配置

```

## 核心组件详解

### 1. 编码器（Encoders / Backbones）

编码器是从分类模型改造而来的特征提取器，提供预训练权重：

- **卷积网络**：ResNet、EfficientNet、MobileNet、DenseNet、VGG 等
- **Transformer**：Vision Transformer (ViT)、Mix Transformer (MiT)
- **timm 支持**：可使用 timm 库中的 800+ 预训练模型
- **用途**：提取图像的多尺度特征，作为分割模型的骨干

### 2. 解码器（Decoders / Architectures）

12 种分割模型架构，每种都有特定的设计优势：

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **Unet** | 经典的 U 型结构，跳跃连接 | 通用分割，医学图像 |
| **Unet++** | 嵌套的 U-Net，密集连接 | 需要精细分割的场景 |
| **FPN** | 特征金字塔网络 | 多尺度目标分割 |
| **PSPNet** | 金字塔池化模块 | 场景解析 |
| **DeepLabV3/V3+** | 空洞卷积，ASPP 模块 | 需要大感受野的场景 |
| **Linknet** | 轻量级网络 | 实时分割 |
| **MAnet** | 位置和通道注意力 | 需要注意力机制的场景 |
| **PAN** | 金字塔注意力网络 | 复杂场景分割 |
| **Segformer** | Transformer 解码器 | 高性能分割 |
| **UPerNet** | 统一感知解析网络 | 场景理解 |
| **DPT** | 密集预测 Transformer | 利用 ViT 特性 |

### 3. 损失函数（Losses）

针对分割任务优化的损失函数：

- **Dice Loss**：基于 Dice 系数，对小目标友好
- **Jaccard Loss (IoU Loss)**：基于交并比
- **Focal Loss**：解决类别不平衡问题
- **Tversky Loss**：Dice Loss 的推广，可调整假阳性/假阴性权重
- **Lovasz Loss**：基于 Lovasz 扩展，优化 IoU
- **Cross Entropy**：标准交叉熵及其变体

### 4. 评估指标（Metrics）

- **IoU (Intersection over Union)**：交并比
- **F-score (Dice Score)**：F1 分数
- **Precision & Recall**：精确率和召回率

## 快速开始示例

### 创建模型（只需 2 行代码）

```python
import segmentation_models_pytorch as smp

# 创建一个 U-Net 模型
model = smp.Unet(
    encoder_name="resnet34",        # 选择编码器（如 mobilenet_v2、efficientnet-b7）
    encoder_weights="imagenet",     # 使用 ImageNet 预训练权重
    in_channels=1,                  # 输入通道数（1=灰度图，3=RGB）
    classes=3,                      # 输出类别数
)
```

### 数据预处理

```python
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# 获取预处理函数（与预训练权重匹配）
preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
```

### 完整训练示例

```python
import torch
import segmentation_models_pytorch as smp

# 1. 创建模型
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,  # 二分类分割
)

# 2. 定义损失函数和优化器
loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3. 训练循环
model.train()
for images, masks in dataloader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, masks)
    loss.backward()
    optimizer.step()
```

## 高级特性

### 1. 自定义输入通道

```python
# 单通道输入（如医学图像）
model = smp.FPN('resnet34', in_channels=1)

# 多通道输入（如多光谱卫星图像）
model = smp.Unet('resnet34', in_channels=10)
```

### 2. 辅助分类输出

在分割的同时进行图像分类：

```python
aux_params = dict(
    pooling='avg',           # 池化方式：'avg' 或 'max'
    dropout=0.5,             # Dropout 比例
    activation='sigmoid',    # 激活函数
    classes=4,               # 分类类别数
)
model = smp.Unet('resnet34', classes=4, aux_params=aux_params)
mask, label = model(x)  # 同时输出分割掩码和分类标签
```

### 3. 调整编码器深度

```python
# 使用更浅的编码器（更快，更轻量）
model = smp.Unet('resnet34', encoder_depth=4)
```

### 4. ONNX 导出

```python
import torch

# 导出为 ONNX 格式（用于生产部署）
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 5. HuggingFace Hub 集成

```python
# 从 HuggingFace Hub 加载预训练模型
model = smp.from_pretrained("path/to/model")

# 保存模型到 HuggingFace Hub
model.save_pretrained("my-awesome-model")
model.push_to_hub("username/my-awesome-model")
```

## 应用案例

### 竞赛获奖记录

该库在众多图像分割竞赛中被使用并获奖（详见 `HALLOFFAME.md`）：

- Kaggle Severstal: Steel Defect Detection（钢材缺陷检测）
- 多个医学图像分割竞赛
- 卫星图像分割竞赛
- 等等...

### 实际项目

- **withoutBG**：开源背景移除工具，使用 smp.Unet 构建图像抠图模型
- 医疗诊断辅助系统
- 自动驾驶感知系统
- 农业作物识别
- 城市规划遥感分析

## 技术要求

- **Python**: 3.9+
- **PyTorch**: 1.8+
- **依赖库**:
  - torchvision >= 0.9
  - timm >= 0.9
  - huggingface-hub >= 0.24
  - numpy, pillow, tqdm 等

## 安装方式

```bash
# 从 PyPI 安装（推荐）
pip install segmentation-models-pytorch

# 从 GitHub 安装最新版
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

## 开发和贡献

### 开发环境设置

```bash
make install_dev  # 创建虚拟环境，安装开发依赖
```

### 运行测试

```bash
make test         # 运行测试套件
make fixup        # 代码格式化和检查
```

### 生成文档

```bash
make table        # 生成编码器表格
```

## 文档资源

- **在线文档**: https://smp.readthedocs.io/
- **GitHub 仓库**: https://github.com/qubvel/segmentation_models.pytorch
- **PyPI 包**: https://pypi.org/project/segmentation-models-pytorch/
- **示例教程**: 见 `examples/` 目录中的 Jupyter Notebook

## 许可证

本项目主要采用 **MIT 许可证**，但部分文件可能受其他许可证约束。商业使用前请仔细检查 `licenses/` 目录中的许可证声明。

## 项目维护者

- **作者**: Pavel Iakubovskii
- **邮箱**: qubvel@gmail.com
- **组织**: qubvel-org

## 总结

**Segmentation Models PyTorch** 是一个功能完善、易于使用的图像分割库，具有以下优势：

✅ **简单易用**：高级 API，几行代码即可开始  
✅ **模型丰富**：12 种架构 + 800+ 编码器  
✅ **预训练权重**：加速训练，提高性能  
✅ **生产就绪**：支持 ONNX、TorchScript  
✅ **活跃维护**：持续更新，社区活跃  
✅ **实战验证**：众多竞赛和实际项目证明有效性  

无论是学术研究、竞赛参赛还是工业应用，这个库都是图像分割任务的优秀选择！

# 项目文件结构详细说明 | Detailed File Structure

## 完整目录树 | Complete Directory Tree

```
segmentation_models.pytorch/
│
├── 📁 .devcontainer/              # VS Code 开发容器配置
│                                  # VS Code dev container configuration
│
├── 📁 .github/                    # GitHub 配置文件
│   └── workflows/                 # CI/CD 工作流
│                                  # GitHub configuration and CI/CD workflows
│
├── 📁 segmentation_models_pytorch/   ⭐ 核心库代码 | Core Library Code
│   │
│   ├── __init__.py               # 库的主入口，导出所有公共 API
│   │                             # Main entry point, exports all public APIs
│   │
│   ├── __version__.py            # 版本号定义
│   │                             # Version information
│   │
│   ├── 📁 base/                  # 基础类和混入
│   │   ├── __init__.py          # Base classes and mixins
│   │   ├── model.py             # SegmentationModel 基类
│   │   │                        # Base segmentation model class
│   │   ├── modules.py           # 通用模块（激活函数、注意力等）
│   │   │                        # Common modules (activations, attention, etc.)
│   │   └── hub_mixin.py         # HuggingFace Hub 集成
│   │                            # HuggingFace Hub integration
│   │
│   ├── 📁 encoders/              # 编码器（骨干网络）| Encoders (Backbones)
│   │   ├── __init__.py          # 编码器注册和获取函数
│   │   │                        # Encoder registry and getter functions
│   │   │
│   │   ├── _base.py             # 编码器基类
│   │   │                        # Base encoder class
│   │   ├── _utils.py            # 编码器工具函数
│   │   │                        # Encoder utility functions
│   │   ├── _preprocessing.py    # 预处理函数
│   │   │                        # Preprocessing functions
│   │   ├── _legacy_pretrained_settings.py  # 旧版预训练权重配置
│   │   │                                   # Legacy pretrained weight settings
│   │   │
│   │   ├── resnet.py            # ResNet 系列 (ResNet-18/34/50/101/152)
│   │   │                        # ResNet family encoders
│   │   ├── efficientnet.py      # EfficientNet 系列 (B0-B7)
│   │   │                        # EfficientNet family encoders
│   │   ├── mobilenet.py         # MobileNet V2/V3（轻量级）
│   │   │                        # MobileNet encoders (lightweight)
│   │   ├── mobileone.py         # MobileOne（超快速）
│   │   │                        # MobileOne encoders (ultra-fast)
│   │   ├── densenet.py          # DenseNet 系列
│   │   │                        # DenseNet family encoders
│   │   ├── vgg.py               # VGG 系列
│   │   │                        # VGG family encoders
│   │   ├── senet.py             # SENet/SE-ResNet
│   │   │                        # SENet/SE-ResNet encoders
│   │   ├── xception.py          # Xception
│   │   ├── inceptionv4.py       # Inception V4
│   │   ├── inceptionresnetv2.py # Inception ResNet V2
│   │   ├── dpn.py               # Dual Path Network
│   │   │
│   │   ├── mix_transformer.py   # Mix Transformer (MiT) - 用于 Segformer
│   │   │                        # Mix Transformer - used in Segformer
│   │   │
│   │   ├── timm_universal.py    # timm 库通用编码器支持
│   │   │                        # Universal timm library encoder support
│   │   ├── timm_efficientnet.py # timm EfficientNet 变体
│   │   ├── timm_vit.py          # timm Vision Transformer
│   │   ├── timm_sknet.py        # timm SKNet
│   │   │
│   │   └── _*.py                # 私有实现文件
│   │                            # Private implementation files
│   │
│   ├── 📁 decoders/              # 解码器（模型架构）| Decoders (Architectures)
│   │   │
│   │   ├── 📁 unet/             # U-Net 模型
│   │   │   ├── __init__.py     # Classic U-Net architecture
│   │   │   ├── model.py        # 模型定义
│   │   │   └── decoder.py      # 解码器实现
│   │   │
│   │   ├── 📁 unetplusplus/     # U-Net++ 模型
│   │   │   ├── __init__.py     # Nested U-Net with dense connections
│   │   │   ├── model.py
│   │   │   └── decoder.py
│   │   │
│   │   ├── 📁 manet/            # MA-Net（多注意力网络）
│   │   │   ├── __init__.py     # Multi-attention network
│   │   │   ├── model.py
│   │   │   └── decoder.py
│   │   │
│   │   ├── 📁 linknet/          # LinkNet（轻量级）
│   │   │   ├── __init__.py     # Lightweight network for real-time
│   │   │   ├── model.py
│   │   │   └── decoder.py
│   │   │
│   │   ├── 📁 fpn/              # Feature Pyramid Network
│   │   │   ├── __init__.py     # Multi-scale feature pyramids
│   │   │   ├── model.py
│   │   │   └── decoder.py
│   │   │
│   │   ├── 📁 pspnet/           # Pyramid Scene Parsing Network
│   │   │   ├── __init__.py     # Scene parsing with pyramid pooling
│   │   │   ├── model.py
│   │   │   └── decoder.py
│   │   │
│   │   ├── 📁 pan/              # Pyramid Attention Network
│   │   │   ├── __init__.py     # Attention-based pyramid network
│   │   │   ├── model.py
│   │   │   └── decoder.py
│   │   │
│   │   ├── 📁 deeplabv3/        # DeepLabV3 和 DeepLabV3+
│   │   │   ├── __init__.py     # ASPP and atrous convolution
│   │   │   ├── model.py
│   │   │   └── decoder.py
│   │   │
│   │   ├── 📁 upernet/          # UPerNet（统一感知解析网络）
│   │   │   ├── __init__.py     # Unified perceptual parsing
│   │   │   ├── model.py
│   │   │   └── decoder.py
│   │   │
│   │   ├── 📁 segformer/        # Segformer（Transformer 解码器）
│   │   │   ├── __init__.py     # Transformer-based decoder
│   │   │   ├── model.py
│   │   │   ├── decoder.py
│   │   │   └── config.py       # 模型配置 | Model configs
│   │   │
│   │   └── 📁 dpt/              # Dense Prediction Transformer
│   │       ├── __init__.py     # Vision Transformer for dense prediction
│   │       ├── model.py
│   │       ├── decoder.py
│   │       └── config.py       # 模型配置 | Model configs
│   │
│   ├── 📁 losses/                # 损失函数 | Loss Functions
│   │   ├── __init__.py          # Loss function exports
│   │   ├── _functional.py       # 底层函数实现
│   │   │                        # Low-level function implementations
│   │   ├── constants.py         # 常量定义
│   │   │                        # Constants
│   │   │
│   │   ├── dice.py              # Dice Loss（对小目标友好）
│   │   │                        # Dice Loss (good for small objects)
│   │   ├── jaccard.py           # Jaccard Loss / IoU Loss
│   │   │                        # Intersection over Union loss
│   │   ├── focal.py             # Focal Loss（处理类别不平衡）
│   │   │                        # Focal Loss (handles class imbalance)
│   │   ├── tversky.py           # Tversky Loss（可调 FP/FN 权重）
│   │   │                        # Tversky Loss (adjustable FP/FN weights)
│   │   ├── lovasz.py            # Lovasz-Softmax Loss
│   │   │                        # Lovasz extension for IoU optimization
│   │   ├── mcc.py               # Matthews Correlation Coefficient Loss
│   │   ├── soft_bce.py          # Soft Binary Cross Entropy
│   │   └── soft_ce.py           # Soft Cross Entropy
│   │
│   ├── 📁 metrics/               # 评估指标 | Evaluation Metrics
│   │   ├── __init__.py
│   │   └── functional.py        # IoU、F-score、精确率、召回率等
│   │                            # IoU, F-score, Precision, Recall, etc.
│   │
│   ├── 📁 datasets/              # 示例数据集加载器
│   │   └── ...                  # Example dataset loaders
│   │
│   └── 📁 utils/                 # 工具函数（大部分已弃用）
│       └── ...                  # Utility functions (mostly deprecated)
│
├── 📁 examples/                   # 使用示例 | Usage Examples
│   ├── binary_segmentation_intro.ipynb          # 二分类分割入门教程
│   │                                           # Binary segmentation tutorial
│   ├── binary_segmentation_buildings.py         # 建筑物分割示例（Python 脚本）
│   │                                           # Building segmentation example
│   ├── camvid_segmentation_multiclass.ipynb     # 多类别分割（CamVid 数据集）
│   │                                           # Multi-class segmentation
│   ├── cars segmentation (camvid).ipynb         # 汽车分割示例
│   │                                           # Car segmentation example
│   ├── convert_to_onnx.ipynb                    # 导出模型为 ONNX 格式
│   │                                           # Export model to ONNX
│   ├── segformer_inference_pretrained.ipynb     # Segformer 预训练模型推理
│   │                                           # Segformer pretrained inference
│   ├── dpt_inference_pretrained.ipynb           # DPT 预训练模型推理
│   │                                           # DPT pretrained inference
│   ├── upernet_inference_pretrained.ipynb       # UPerNet 预训练模型推理
│   │                                           # UPerNet pretrained inference
│   └── save_load_model_and_share_with_hf_hub.ipynb  # 保存/加载模型及 HF Hub 分享
│                                                   # Save/load and HF Hub sharing
│
├── 📁 tests/                      # 单元测试 | Unit Tests
│   ├── conftest.py               # pytest 配置
│   │                             # pytest configuration
│   ├── utils.py                  # 测试工具函数
│   │                             # Test utilities
│   ├── test_base.py              # 基础功能测试
│   ├── test_losses.py            # 损失函数测试
│   ├── test_preprocessing.py     # 预处理测试
│   │
│   ├── 📁 models/                # 模型架构测试
│   │   └── test_*.py            # Tests for each architecture
│   │
│   ├── 📁 encoders/              # 编码器测试
│   │   └── test_*.py            # Tests for encoders
│   │
│   └── 📁 base/                  # 基础类测试
│       └── test_*.py            # Tests for base classes
│
├── 📁 docs/                       # 文档源文件 | Documentation Source
│   ├── conf.py                   # Sphinx 配置
│   │                             # Sphinx configuration
│   ├── index.rst                 # 文档首页
│   │                             # Documentation homepage
│   ├── models.rst                # 模型架构文档
│   │                             # Model architectures documentation
│   ├── encoders.rst              # 原生编码器列表
│   │                             # Native encoders table
│   ├── encoders_timm.rst         # timm 编码器列表
│   │                             # timm encoders table
│   ├── encoders_dpt.rst          # DPT 编码器列表
│   │                             # DPT encoders table
│   ├── losses.rst                # 损失函数文档
│   │                             # Loss functions documentation
│   ├── metrics.rst               # 评估指标文档
│   │                             # Metrics documentation
│   ├── quickstart.rst            # 快速开始指南
│   │                             # Quick start guide
│   ├── install.rst               # 安装说明
│   │                             # Installation guide
│   ├── save_load.rst             # 保存/加载模型
│   │                             # Save/load models guide
│   ├── insights.rst              # 使用建议和最佳实践
│   │                             # Insights and best practices
│   └── Makefile                  # 文档构建命令
│                                 # Documentation build commands
│
├── 📁 scripts/                    # 辅助脚本 | Utility Scripts
│   └── 📁 models-conversions/    # 模型权重转换脚本
│       ├── segformer-original-decoder-to-smp.py  # Segformer 权重转换
│       ├── upernet-hf-to-smp.py                  # UPerNet 权重转换
│       └── dpt-original-to-smp.py                # DPT 权重转换
│
├── 📁 misc/                       # 杂项工具 | Miscellaneous Tools
│   ├── generate_table.py         # 生成原生编码器表格
│   │                             # Generate native encoder table
│   ├── generate_table_timm.py    # 生成 timm 编码器表格
│   │                             # Generate timm encoder table
│   └── generate_test_models.py   # 生成测试模型
│                                 # Generate test models
│
├── 📁 requirements/               # 依赖配置文件 | Dependency Files
│   └── ...                       # Various requirement files
│
├── 📁 licenses/                   # 许可证信息 | License Information
│   └── LICENSES.md               # 详细许可证说明
│                                 # Detailed license information
│
├── 📁 pics/                       # 图片资源 | Image Resources
│   └── ...                       # Project images and assets
│
├── 📄 README.md                   # 项目主说明文档（英文）
│                                  # Main project documentation (English)
│
├── 📄 PROJECT_OVERVIEW_CN.md      # 项目详细说明（中文）⭐ 新增
│                                  # Detailed project overview (Chinese) ⭐ NEW
│
├── 📄 QUICK_REFERENCE.md          # 快速参考指南 ⭐ 新增
│                                  # Quick reference guide ⭐ NEW
│
├── 📄 FILE_STRUCTURE.md           # 文件结构说明（本文件）⭐ 新增
│                                  # File structure guide (this file) ⭐ NEW
│
├── 📄 HALLOFFAME.md               # 竞赛获奖记录
│                                  # Competition wins hall of fame
│
├── 📄 LICENSE                     # MIT 许可证
│                                  # MIT License
│
├── 📄 pyproject.toml              # 项目配置和依赖定义
│                                  # Project configuration and dependencies
│
├── 📄 Makefile                    # 开发命令快捷方式
│                                  # Development command shortcuts
│
├── 📄 .readthedocs.yaml           # Read the Docs 配置
│                                  # Read the Docs configuration
│
└── 📄 .gitignore                  # Git 忽略文件配置
                                   # Git ignore configuration
```

## 核心文件说明 | Key Files Explained

### 配置文件 | Configuration Files

| 文件 | 用途 |
|------|------|
| `pyproject.toml` | 项目元数据、依赖、构建配置 |
| `Makefile` | 开发命令（测试、格式化、文档生成） |
| `.readthedocs.yaml` | 在线文档构建配置 |
| `.gitignore` | Git 版本控制忽略规则 |

### 文档文件 | Documentation Files

| 文件 | 说明 |
|------|------|
| `README.md` | 项目主文档（英文），GitHub 首页显示 |
| `PROJECT_OVERVIEW_CN.md` | ⭐ 中文详细说明（新增） |
| `QUICK_REFERENCE.md` | ⭐ 快速参考指南（新增） |
| `FILE_STRUCTURE.md` | ⭐ 本文件，文件结构说明（新增） |
| `HALLOFFAME.md` | 使用本库获奖的竞赛列表 |
| `LICENSE` | MIT 开源许可证 |

### 代码统计 | Code Statistics

```
核心库代码行数估算 | Estimated Lines of Code:
- encoders/: ~5,000 行
- decoders/: ~8,000 行  
- losses/: ~1,500 行
- metrics/: ~500 行
- base/: ~1,000 行
总计 | Total: ~16,000 行代码

测试代码 | Test Code: ~5,000 行
文档 | Documentation: ~3,000 行
示例 | Examples: ~2,000 行

整体项目 | Total Project: ~26,000+ 行代码
```

## 文件命名规范 | File Naming Conventions

- `__init__.py`: 包初始化文件，定义公共 API
- `model.py`: 完整模型定义（编码器+解码器）
- `decoder.py`: 解码器实现
- `_*.py`: 私有模块（下划线开头表示内部使用）
- `test_*.py`: 测试文件
- `*.rst`: reStructuredText 文档源文件
- `*.ipynb`: Jupyter Notebook 示例

## 重要提示 | Important Notes

### 对于用户 | For Users
- 只需关注 `segmentation_models_pytorch/` 目录中的公共 API
- 从 `examples/` 开始学习使用方法
- 查阅 `docs/` 或在线文档了解详细 API

### 对于开发者 | For Developers
- `tests/` 包含完整的测试套件
- 使用 `make test` 运行测试
- 使用 `make fixup` 进行代码格式化
- 新增编码器需更新 `misc/generate_table.py`

### 对于贡献者 | For Contributors
- 遵循现有代码风格
- 为新功能添加测试
- 更新相关文档
- 查看 README.md 中的贡献指南

---

**创建日期 | Created**: 2025  
**用途 | Purpose**: 帮助理解项目文件组织结构 | Help understand project file organization

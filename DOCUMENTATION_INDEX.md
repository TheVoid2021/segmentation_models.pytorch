# 📚 项目文档导航 | Documentation Navigator

欢迎来到 **Segmentation Models PyTorch** 项目！  
Welcome to **Segmentation Models PyTorch** project!

本项目现在提供了完整的中英文文档，帮助您快速了解和使用这个强大的图像分割库。  
This project now provides comprehensive bilingual documentation to help you quickly understand and use this powerful image segmentation library.

---

## 📖 文档列表 | Documentation Index

### 1️⃣ 项目详细说明（中文）| Detailed Overview (Chinese)
**文件**: [`PROJECT_OVERVIEW_CN.md`](PROJECT_OVERVIEW_CN.md)  
**内容**: 
- 项目是什么，用来做什么
- 核心功能和特性详解
- 完整的文件组织结构说明
- 12 种模型架构介绍
- 800+ 编码器列表
- 损失函数和评估指标
- 代码示例和最佳实践
- 应用案例和竞赛获奖记录

**适合**: 想要全面了解项目的中文用户

---

### 2️⃣ 快速参考指南（英文）| Quick Reference (English)
**文件**: [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)  
**内容**:
- One-minute project overview
- Common use cases
- Directory structure quick map
- Available models and encoders
- Key code patterns
- Loss functions guide
- File organization explained
- Installation and setup
- Development commands

**适合**: 需要快速上手的开发者

---

### 3️⃣ 文件结构详细说明（中英双语）| File Structure (Bilingual)
**文件**: [`FILE_STRUCTURE.md`](FILE_STRUCTURE.md)  
**内容**:
- 完整的目录树结构
- 每个文件和目录的详细说明
- 核心文件说明
- 代码统计
- 文件命名规范
- 针对用户、开发者、贡献者的提示

**适合**: 想要深入了解项目组织的开发者

---

### 4️⃣ 官方 README（英文）| Official README (English)
**文件**: [`README.md`](README.md)  
**内容**:
- Official project description
- Installation instructions
- Quick start guide
- Examples and tutorials
- Model architectures table
- Encoder list
- API documentation
- Contributing guidelines

**适合**: GitHub 访客和英文用户

---

### 5️⃣ 竞赛获奖记录 | Hall of Fame
**文件**: [`HALLOFFAME.md`](HALLOFFAME.md)  
**内容**:
- 使用本库获奖的 Kaggle 竞赛
- 获奖者信息和解决方案链接
- 证明库的实战价值

**适合**: 想了解库实际应用效果的用户

---

## 🚀 快速开始 | Quick Start

### 第一次使用？| First Time User?

1. **了解项目**（中文用户）→ 阅读 [`PROJECT_OVERVIEW_CN.md`](PROJECT_OVERVIEW_CN.md)
2. **了解项目**（英文用户）→ 阅读 [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) 或 [`README.md`](README.md)
3. **安装库**:
   ```bash
   pip install segmentation-models-pytorch
   ```
4. **运行示例** → 查看 `examples/` 目录中的 Jupyter Notebook
5. **深入学习** → 访问 [在线文档](https://smp.readthedocs.io/)

### 想要贡献代码？| Want to Contribute?

1. 阅读 [`FILE_STRUCTURE.md`](FILE_STRUCTURE.md) 了解项目结构
2. 阅读 [`README.md`](README.md) 的贡献指南部分
3. 设置开发环境:
   ```bash
   make install_dev
   ```
4. 运行测试:
   ```bash
   make test
   ```

---

## 📊 项目概览 | Project at a Glance

| 项目特性 | 详细信息 |
|---------|---------|
| **名称** | Segmentation Models PyTorch (SMP) |
| **用途** | 图像语义分割神经网络库 |
| **语言** | Python 3.9+ |
| **框架** | PyTorch 1.8+ |
| **许可证** | MIT License |
| **模型数量** | 12 种架构 × 800+ 编码器 = 9600+ 组合 |
| **代码量** | ~26,000 行（含测试和文档） |
| **安装量** | PyPI 每月下载数万次 |
| **维护状态** | ✅ 活跃维护中 |

---

## 🎯 常见问题 | FAQ

### Q1: 这个项目是干什么的？
**A**: 这是一个图像分割库，用于训练深度学习模型来识别图像中每个像素属于哪个类别。比如在医学图像中分割器官，在自动驾驶中识别道路和车辆等。

### Q2: 我应该使用哪个模型？
**A**: 
- 通用场景：**Unet**（经典、稳定）
- 高精度需求：**Unet++** 或 **Segformer**
- 实时应用：**Linknet**（快速）
- 多尺度目标：**FPN** 或 **PSPNet**
- 最先进性能：**Segformer** 或 **DPT**

### Q3: 我应该使用哪个编码器？
**A**:
- 通用场景：**ResNet-34/50**（平衡性能和速度）
- 移动设备：**MobileNet V2/V3**（轻量级）
- 高精度：**EfficientNet-B4/B7** 或 **ResNet-101**
- 最新技术：**Mix Transformer** (用于 Segformer)

### Q4: 如何开始？
**A**: 查看 `examples/binary_segmentation_intro.ipynb`，这是最好的入门教程。

### Q5: 支持哪些任务？
**A**:
- ✅ 二分类分割（前景/背景）
- ✅ 多类别分割（多个类别）
- ✅ 实例分割的语义部分
- ✅ 医学图像分割
- ✅ 遥感图像分割
- ✅ 任意像素级分类任务

---

## 🔗 相关链接 | Related Links

- 📦 **PyPI 包**: https://pypi.org/project/segmentation-models-pytorch/
- 💻 **GitHub 仓库**: https://github.com/qubvel/segmentation_models.pytorch
- 📖 **在线文档**: https://smp.readthedocs.io/
- 🏆 **HuggingFace Models**: 
  - [Segformer](https://huggingface.co/collections/smp-hub/segformer-6749eb4923dea2c355f29a1f)
  - [UPerNet](https://huggingface.co/collections/smp-hub/upernet-67fadcdbe08418c6ea94f768)
  - [DPT](https://huggingface.co/collections/smp-hub/dpt-67f30487327c0599a0c62d68)

---

## 📝 文档贡献 | Documentation Contributions

这些新增的文档文件由 GitHub Copilot 创建，旨在：  
These new documentation files were created by GitHub Copilot to:

✅ 为中文用户提供完整的项目说明  
✅ Provide complete project overview for Chinese users

✅ 提供快速参考指南  
✅ Provide quick reference guide

✅ 详细解释文件组织结构  
✅ Explain file organization in detail

✅ 降低新用户的学习门槛  
✅ Lower the learning curve for new users

✅ 方便开发者理解和贡献代码  
✅ Help developers understand and contribute code

---

## 🙏 致谢 | Acknowledgments

- **原作者**: Pavel Iakubovskii
- **社区贡献者**: 所有为项目做出贡献的开发者
- **赞助商**: withoutBG 等支持项目的公司

---

**最后更新**: 2025年12月  
**Last Updated**: December 2025

如有问题或建议，欢迎在 GitHub 上提出 Issue 或 Pull Request！  
For questions or suggestions, feel free to open an Issue or Pull Request on GitHub!

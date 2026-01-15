# Qwen3-0.6B GRPO GSM8K Demo

这是一个使用TRL Transformers Trainer在单卡上基于Qwen3-0.6B模型和GSM8K数据集进行GRPO(Generalized Relative Preference Optimization)训练的演示仓库。

## 项目概述

本项目旨在学习和复现DeepSeek-R1-Zero的训练方法，通过在较小的模型(Qwen3-0.6B)上进行GRPO训练，来理解其核心原理和实现方式。

## 主要组件

- **基础模型**: Qwen3-0.6B - 一个轻量级的语言模型
- **数据集**: GSM8K - 小学数学推理数据集
- **训练方法**: GRPO (Generalized Relative Preference Optimization)
- **训练框架**: TRL (Transformers Reinforcement Learning)

## 项目结构

```
Qwen3-0.6B-GRPO-GSM8K-Demo/
├── README.md
├── LICENSE
└── .gitignore
```

## 开发计划

- [ ] 设置环境和依赖
- [ ] 实现数据加载和预处理
- [ ] 配置GRPO训练器
- [ ] 实现训练循环
- [ ] 添加评估和日志记录
- [ ] 优化和实验

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- TRL (Transformers Reinforcement Learning)
- 其他依赖将在实现过程中添加

## 使用方法

(待实现代码完成后更新)

## 许可证

请参阅 [LICENSE](LICENSE) 文件了解详细信息。

## 参考资料

- [DeepSeek-R1-Zero](https://github.com/deepseek-ai/DeepSeek-R1)
- [TRL文档](https://huggingface.co/docs/trl)
- [Qwen3模型](https://huggingface.co/Qwen/Qwen3-0.6B)
- [GSM8K数据集](https://huggingface.co/datasets/gsm8k)
# 🤖 最中幻想LLM训练场

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-2.3%2B-green" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p align="center">
  <b>一个完整的可视化大语言模型训练平台</b><br>
  基于 PyTorch 从零训练字符级 GPT，支持实时可视化、对话测试与 DeepSeek AI 数据集生成。
</p>

---

## ✨ 功能特性

| 模块 | 功能 |
|------|------|
| 🏋️ **真实训练** | 基于 PyTorch 的字符级 GPT 模型，支持 Transformer 结构、注意力机制、梯度累积、学习率调度 |
| 📊 **实时可视化** | 训练损失/验证损失双曲线、学习率 LR、困惑度 PPL、训练速度 tokens/s 四图同步更新 |
| 🤖 **DeepSeek 数据集生成** | 调用 DeepSeek API 自动生成训练数据，支持「基于当前数据集风格生成」模式 |
| 💬 **对话测试场** | 维护多轮对话历史，支持温度、Top-K、Top-P 采样参数实时调节 |
| 💾 **模型管理** | 自动保存最佳模型、手动保存/加载、模型文件下载、列表管理 |
| ⚙️ **CPU 控制** | 可限制 PyTorch 使用的 CPU 线程数，避免占用全部算力 |
| 📋 **运行日志** | 黑色终端风格日志面板，实时记录训练状态与推理参数 |

---

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- 推荐使用 CUDA，CPU 亦可运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
# Windows
python app.py

# 或双击
run.bat
```

访问 http://127.0.0.1:5000 即可打开训练平台。

---

## 📖 使用指南

### 1️⃣ 准备数据集

**方式 A：上传自己的 `.txt` 文件**
- 在左侧「📁 数据集」区域点击上传
- 支持任意纯文本文件，自动构建字符级词表

**方式 B：使用 DeepSeek 自动生成**
- 填写 DeepSeek API Key
- 输入生成主题（如：科幻小说对话、古诗词、技术文档）
- 勾选「基于当前数据集风格生成」可让 AI 学习已有文风
- 点击「✨ 生成数据集」，等待约 10~30 秒

### 2️⃣ 配置训练参数

左侧「⚙️ 训练超参数」面板提供完整调节：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `n_embd` | 模型嵌入维度 | 512 |
| `n_layer` | Transformer 层数 | 8 |
| `n_head` | 注意力头数 | 8 |
| `block_size` | 上下文长度 | 256 |
| `batch_size` | 批次大小 | 32 |
| `learning_rate` | 学习率 | 3e-4 |
| `scheduler` | 学习率调度 | cosine |
| `cpu_threads` | CPU 线程限制 | 0（不限） |

### 3️⃣ 开始训练

点击「🚀 开始训练」，中间面板将实时显示：
- 📈 训练/验证损失曲线
- 📉 学习率变化
- 🧮 困惑度 PPL
- ⚡ 训练速度 tokens/s
- ✍️ 实时生成样例

训练过程中自动保存最佳模型到 `models/best_model.pt`。

### 4️⃣ 对话测试

训练完成后，在右侧「🎯 实战测试场」：
- 输入消息进行多轮对话
- 实时调节温度、Top-K、Top-P 控制生成风格
- 每次推理参数自动记录到日志面板

---

## 🏗️ 项目结构

```
zzhx-llm-ai-train/
├── app.py                 # Flask 后端主程序
├── model_manager.py       # GPT 模型定义与训练管理
├── utils.py               # CharTokenizer 与数据加载工具
├── requirements.txt       # Python 依赖
├── run.bat / run.sh       # 启动脚本
├── data/                  # 数据集目录（.gitignore 排除大文件）
├── models/                # 模型保存目录（.gitignore 排除大文件）
├── templates/
│   └── index.html         # 前端主页面
└── static/
    ├── script.js          # 前端交互逻辑
    └── style.css          # 样式表
```

---

## 🛠️ 技术栈

- **后端**: Flask + PyTorch
- **模型**: 字符级 GPT (Causal Self-Attention + LayerNorm + GELU)
- **训练优化**: AdamW、梯度累积、梯度裁剪、CosineAnnealingLR + LinearLR 预热
- **前端**: Bootstrap 5 + Chart.js
- **数据生成**: DeepSeek Chat API (`deepseek-chat`)
- **通信**: Server-Sent Events (SSE) 实时推送训练进度

---

## 📸 界面预览

> 启动服务后访问 http://127.0.0.1:5000 即可体验完整功能。

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/awesome-feature`)
3. 提交更改 (`git commit -am 'Add awesome feature'`)
4. 推送分支 (`git push origin feature/awesome-feature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/qingshanjiluo">qingshanjiluo</a>
</p>

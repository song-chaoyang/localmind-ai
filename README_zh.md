# OpenMind

<div align="center">

<img src="assets/images/hero-logo.svg" alt="OpenMind" width="700">

**免费替代 ChatGPT + Claude + Gemini。本地运行。无需 GPU。一键安装。**

[![PyPI version](https://img.shields.io/pypi/v/openmind.svg)](https://pypi.org/project/openmind/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[English](README.md) | **中文**

</div>

---

## 🎬 效果演示

<div align="center">

<img src="assets/images/demo-screenshot.svg" alt="OpenMind 演示" width="700">

</div>

## ✨ 为什么选择 OpenMind？

| | ChatGPT | Claude | Gemini | **OpenMind** |
|---|---------|--------|--------|-------------|
| **价格** | $20/月 | $20/月 | $20/月 | **永久免费** |
| **隐私** | ❌ 数据上传 | ❌ 数据上传 | ❌ 数据上传 | **✅ 100% 本地** |
| **离线** | ❌ | ❌ | ❌ | **✅ 支持离线** |
| **多模型** | 仅 GPT | 仅 Claude | 仅 Gemini | **✅ 所有模型** |
| **可定制** | ❌ | ❌ | ❌ | **✅ 完全开源** |
| **无需注册** | ❌ 必须注册 | ❌ 必须注册 | ❌ 必须注册 | **✅ 零注册** |
| **数据归属** | 他们的服务器 | 他们的服务器 | 他们的服务器 | **✅ 你的机器** |

---

## 🚀 60 秒上手

### 一行命令安装（推荐）

```bash
pip install openmind && openmind
```

就这样！OpenMind 会自动：
1. ✅ 下载默认模型
2. ✅ 启动精美的聊天界面
3. ✅ 在浏览器中打开 `http://localhost:3000`

### 指定模型

```bash
openmind --model llama3      # 通用对话（推荐）
openmind --model mistral      # 更轻量、更快
openmind --model deepseek-coder  # 编程最强
```

<div align="center">

<img src="assets/images/install-demo.svg" alt="安装演示" width="700">

</div>

> 💡 **无需安装 Ollama。** OpenMind 内置一切，安装即用。

---

## 🖥️ 精美界面

<div align="center">

<img src="assets/images/ui-chat.svg" alt="聊天界面" width="700">

</div>

- 💬 **实时流式输出** — 逐字显示回复
- 🌙 **深色/浅色模式** — 护眼，随时切换
- 📱 **移动端适配** — 手机、平板、桌面都能用
- 📂 **文件上传** — 拖拽上传文档、图片、代码
- 🔍 **历史搜索** — 秒找任何历史对话
- 🎨 **Markdown 渲染** — 代码高亮、表格、数学公式

---

## 🤖 多模型一键切换

<div align="center">

<img src="assets/images/model-switcher.svg" alt="模型切换" width="700">

</div>

| 模型 | 大小 | 内存需求 | 最适合 |
|------|------|---------|--------|
| 🦙 **Llama 3** | 8B | 8GB | 通用对话 |
| 🌊 **Mistral** | 7B | 8GB | 快速响应 |
| 🐼 **DeepSeek** | 7B | 8GB | 编程推理 |
| 🔮 **Qwen 2** | 7B | 8GB | 中英双语 |
| 💎 **Phi-3** | 3.8B | 4GB | 轻量任务 |
| 🧠 **Gemma 2** | 9B | 8GB | 复杂推理 |

> 💡 **没有 GPU？没问题。** 所有模型支持 CPU 运行，普通笔记本即可。

---

## 📄 RAG — 和你的文档对话

上传任何文件，向它提问。文档永远不会离开你的机器。

<div align="center">

<img src="assets/images/rag-demo.svg" alt="RAG 演示" width="700">

</div>

支持格式：`.txt` `.md` `.csv` `.json` `.py` `.js` `.pdf` `.docx` 等

---

## 🔌 插件生态

```bash
openmind plugin install web-search        # 网页搜索
openmind plugin install code-interpreter   # 代码执行
openmind plugin install image-generator    # 图像生成
```

---

## 🏗️ 架构

<div align="center">

<img src="assets/images/architecture.svg" alt="架构图" width="700">

</div>

- **后端**: Python + FastAPI（异步、快速、轻量）
- **模型运行**: 内置 llama.cpp（无外部依赖）
- **前端**: 原生 HTML/CSS/JS（无构建步骤，秒开）
- **RAG**: 内置嵌入 + 向量搜索
- **存储**: SQLite（零配置）

---

## 🛠️ 开发者

```python
from openmind import OpenMind

mind = OpenMind(model="llama3")

# 对话
response = mind.chat("用简单的语言解释量子计算")

# 流式输出
for chunk in mind.chat_stream("写一个 Python 爬虫"):
    print(chunk, end="", flush=True)

# 文档问答
mind.ingest("report.pdf")
answer = mind.chat("报告的主要发现是什么？")

# 切换模型
mind.switch_model("mistral")
```

---

## 🗺️ 路线图

### v0.1（当前）— 基础 ✅
- [x] 一键安装运行
- [x] 内置模型运行时
- [x] 精美 Web UI
- [x] 多模型支持
- [x] RAG 文档问答
- [x] 插件系统
- [x] REST API

### v0.2 — 智能化
- [ ] AI Agent 模式
- [ ] 多模型自动路由
- [ ] 语音输入/输出
- [ ] 图像理解

### v0.3 — 桌面应用
- [ ] macOS .dmg
- [ ] Windows .exe
- [ ] Linux .AppImage
- [ ] 系统托盘 + 自动更新

### v1.0 — 平台
- [ ] 插件市场
- [ ] 团队协作
- [ ] 模型微调 UI
- [ ] 移动端 App

---

## 📜 许可证

[MIT 许可证](LICENSE) — 个人和商业使用均免费。

---

<div align="center">

**由 OpenMind 社区用 ❤️ 打造**

*"AI 应该是免费的、私有的、人人可用的。"*

[⭐ 在 GitHub 上给我们加星](https://github.com/song-chaoyang/localmind-ai) — 只需一键！

</div>

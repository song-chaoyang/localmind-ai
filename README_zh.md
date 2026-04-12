# LocalMind

<div align="center">

# 🧠 LocalMind

**你的私有 AI 操作系统 — 一条命令，完整 AI 体验**

[![PyPI version](https://img.shields.io/pypi/v/localmind.svg)](https://pypi.org/project/localmind/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Discord](https://img.shields.io/discord/XXXXXXXXXX?color=5865F2&label=Discord)](https://discord.gg/localmind)

[English](README.md) | **中文** | [日本語](README_ja.md)

*"AI 界的 Linux — 夺回你对人工智能的控制权。"*

</div>

<div align="center">

<img src="assets/images/hero-logo.svg" alt="LocalMind Hero" width="700">

</div>

---

## 📖 目录

- [✨ LocalMind 是什么？](#-localmind-是什么)
- [🚀 快速开始](#-快速开始)
- [🎯 核心功能](#-核心功能)
- [🏗️ 架构](#-架构)
- [🔌 插件系统](#-插件系统)
- [📚 文档](#-文档)
- [🤝 参与贡献](#-参与贡献)
- [🗺️ 路线图](#-路线图)
- [❓ 常见问题](#-常见问题)
- [📜 许可证](#-许可证)

---

## ✨ LocalMind 是什么？

**LocalMind** 是一个开源的、隐私优先的 AI 操作系统，完全运行在你的本地机器上。它将模型管理、智能 Agent、插件生态和工作流编排统一到一个简洁优雅的界面中。

### 🎯 我们解决的问题

当今的 AI 领域是碎片化的：
- 🔴 **Ollama** 能运行模型，但没有 Agent 能力
- 🔴 **LangChain** 能构建 Agent，但配置复杂
- 🔴 **Open WebUI** 提供聊天界面，但缺乏自动化
- 🔴 **AutoGPT** 能自动化，但不可靠且难以控制

**LocalMind 将这一切整合在一起** — 一个平台，无限可能，零数据外泄。

<div align="center">

<img src="assets/images/feature-overview.svg" alt="功能概览" width="700">

</div>

### 💡 为什么选择 LocalMind？

| 功能 | LocalMind | Ollama | LangChain | Open WebUI | AutoGPT |
|------|-----------|--------|-----------|------------|---------|
| 本地模型管理 | ✅ | ✅ | ❌ | ✅ | ❌ |
| AI Agent 系统 | ✅ | ❌ | ✅ | ❌ | ✅ |
| 插件生态 | ✅ | ❌ | 部分 | ❌ | ❌ |
| 可视化工作流构建器 | ✅ | ❌ | ❌ | ❌ | ❌ |
| RAG 管道 | ✅ | ❌ | ✅ | ✅ | ❌ |
| 记忆与上下文 | ✅ | ❌ | 部分 | 部分 | ✅ |
| 多模型支持 | ✅ | ✅ | ✅ | ✅ | ❌ |
| Web UI | ✅ | ❌ | ❌ | ✅ | ❌ |
| CLI 界面 | ✅ | ✅ | ❌ | ❌ | ✅ |
| 隐私优先 | ✅ | ✅ | ❌ | ✅ | ✅ |
| 一键安装 | ✅ | ✅ | ❌ | ✅ | ❌ |

---

## 🚀 快速开始

### 安装

```bash
# 一条命令安装
pip install localmind

# 或安装所有可选依赖
pip install "localmind[all]"

# 或从源码克隆安装
git clone https://github.com/song-chaoyang/localmind-ai.git
cd localmind
pip install -e ".[all]"
```

### 启动你的 AI 操作系统

```bash
# 启动 Web UI（默认）
localmind start

# 仅 CLI 模式
localmind start --cli

# 指定模型启动
localmind start --model llama3

# 无头模式（仅 API 服务）
localmind start --headless --port 8080
```

就这样！在浏览器中打开 `http://localhost:3000`，开始与你的私有 AI 对话。

<div align="center">

<img src="assets/images/cli-demo.svg" alt="CLI 演示" width="700">

</div>

> 💡 **提示**：请确保已安装并运行 [Ollama](https://ollama.ai)。然后运行 `ollama pull llama3` 下载模型。

### 第一次对话

```python
from localmind import LocalMind

# 初始化你的 AI 操作系统
mind = LocalMind()

# 下载并加载模型（首次运行可能需要几分钟）
mind.load_model("llama3")

# 开始对话
response = mind.chat("你好！你能帮我分析数据吗？")
print(response)
```

### 使用 AI Agent

```python
from localmind import LocalMind
from localmind.agents import ResearchAgent, CodeAgent, DataAgent

mind = LocalMind()
mind.load_model("llama3")

# 创建专业 Agent
researcher = ResearchAgent(mind)
coder = CodeAgent(mind)
analyst = DataAgent(mind)

# 让 Agent 协作
result = mind.collaborate(
    agents=[researcher, coder, analyst],
    task="研究最新的 AI 趋势并构建一个演示"
)
print(result)
```

### 构建工作流

```python
from localmind import LocalMind
from localmind.core import Workflow

mind = LocalMind()
mind.load_model("llama3")

# 定义工作流
workflow = Workflow("每日报告生成器")

# 添加步骤
workflow.add_step("fetch_news", agent="research", input="今日 AI 新闻")
workflow.add_step("summarize", agent="writer", depends_on="fetch_news")
workflow.add_step("format_report", agent="formatter", depends_on="summarize")

# 执行
result = workflow.run(mind)
print(result)
```

<div align="center">

<img src="assets/images/workflow-demo.svg" alt="工作流演示" width="700">

</div>

---

## 🎯 核心功能

### 🤖 智能 Agent 系统
- **内置 Agent**：研究、编程、数据分析、写作、翻译等
- **自定义 Agent**：用简单的 Python 类创建你自己的 Agent
- **多 Agent 协作**：Agent 之间协同完成复杂任务
- **工具集成**：Agent 可以使用网页搜索、文件读写、代码执行等

<div align="center">

<img src="assets/images/agent-collaboration.svg" alt="Agent 协作" width="700">

</div>

### 🔌 插件生态
- **一键安装**：`localmind plugin install <名称>`
- **社区插件**：在插件市场中浏览和分享
- **简单开发**：使用我们的 SDK 构建插件
- **热重载**：无需重启即可开发插件

### 📊 RAG（检索增强生成）
- **文档摄入**：支持 PDF、DOCX、TXT、Markdown、代码文件
- **向量存储**：内置向量数据库，支持多种后端
- **智能分块**：自动文档切分和嵌入
- **混合搜索**：结合语义搜索和关键词搜索

### 🧠 记忆系统
- **短期记忆**：对话上下文管理
- **长期记忆**：持久化知识存储
- **语义记忆**：自动组织的知识图谱
- **情景记忆**：记住过去的交互

### 🎨 精美 Web UI
- **现代界面**：简洁、响应式设计
- **实时流式输出**：实时查看生成的回复
- **深色/浅色模式**：随你喜好
- **移动端友好**：任何设备上使用

<div align="center">

<img src="assets/images/web-ui.svg" alt="Web UI" width="700">

</div>

### 🔒 隐私与安全
- **100% 本地运行**：所有数据留在你的机器上
- **零遥测**：我们不追踪任何东西
- **沙箱执行**：代码在隔离环境中运行
- **加密存储**：数据静态加密

---

## 🏗️ 架构

<div align="center">

<img src="assets/images/architecture.svg" alt="架构图" width="700">

</div>

```
localmind/
├── src/
│   ├── core/           # 核心引擎、配置、工作流引擎
│   │   ├── engine.py   # LocalMind 主引擎
│   │   ├── config.py   # 配置管理
│   │   ├── memory.py   # 记忆系统
│   │   ├── workflow.py # 工作流编排
│   │   └── events.py   # 事件总线系统
│   ├── models/         # 模型管理与抽象
│   ├── agents/         # Agent 系统
│   ├── plugins/        # 插件系统
│   ├── api/            # REST 和 WebSocket API
│   ├── ui/             # Web UI
│   └── utils/          # 共享工具
├── tests/              # 完整的测试套件
├── examples/           # 使用示例
├── docs/               # 文档
└── scripts/            # 实用脚本
```

### 设计原则

1. **模块化**：每个组件都是可插拔、可替换的
2. **可扩展**：轻松添加新模型、Agent、插件和工具
3. **高性能**：异步优先、懒加载、高效内存使用
4. **安全**：沙箱执行、加密存储、零数据泄露
5. **优雅**：简洁的代码、完善的文档、愉悦的体验

---

## 🔌 插件系统

### 安装插件

```bash
# 从插件市场安装
localmind plugin install web-search
localmind plugin install code-executor
localmind plugin install image-generator

# 从 Git 仓库安装
localmind plugin install https://github.com/user/my-plugin

# 列出已安装的插件
localmind plugin list
```

### 创建插件

```python
from localmind.plugins import Plugin, plugin_metadata

@plugin_metadata(
    name="my-awesome-plugin",
    version="1.0.0",
    description="Does awesome things",
    author="Your Name"
)
class MyAwesomePlugin(Plugin):
    def on_load(self):
        """插件加载时调用"""
        self.logger.info("MyAwesomePlugin loaded!")

    def register_tools(self):
        """注册 Agent 可使用的工具"""
        return [
            {
                "name": "my_tool",
                "description": "My custom tool",
                "function": self.my_tool_function
            }
        ]

    def my_tool_function(self, query: str) -> str:
        return f"Result for: {query}"
```

---

## 📚 文档

- [📖 入门指南](docs/getting-started.md)
- [🔧 配置参考](docs/configuration.md)
- [🤖 Agent 开发指南](docs/agent-development.md)
- [🔌 插件开发指南](docs/plugin-development.md)
- [🧠 记忆系统详解](docs/memory-system.md)
- [📊 RAG 管道指南](docs/rag-pipeline.md)
- [🌐 API 参考](docs/api-reference.md)
- [🚀 部署指南](docs/deployment.md)

---

## 🤝 参与贡献

我们 ❤️ 贡献！LocalMind 由社区构建，为社区服务。

### 快速贡献指南

1. **Fork** 本仓库
2. **创建** 功能分支：`git checkout -b feature/amazing-feature`
3. **提交** 你的更改：`git commit -m 'Add amazing feature'`
4. **推送** 到分支：`git push origin feature/amazing-feature`
5. **发起** Pull Request

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 我们需要帮助的领域

- 🌍 **翻译**：帮助将 LocalMind 翻译成你的语言
- 📚 **文档**：改进文档、添加教程
- 🐛 **Bug 修复**：查看标记为 "good first issue" 的 Issue
- ✨ **新功能**：提出并实现新功能
- 🔌 **插件**：构建和分享插件
- 🧪 **测试**：提高测试覆盖率

---

## 🗺️ 路线图

### v0.1.0（当前版本）— 基础 ✅
- [x] 核心引擎和配置系统
- [x] 模型管理（Ollama 集成）
- [x] 基础聊天界面（CLI + Web UI）
- [x] 内置 Agent（研究、编程、数据、写作）
- [x] 插件系统与 SDK
- [x] RAG 管道与文档摄入
- [x] 记忆系统（短期 + 长期）
- [x] REST API 和 WebSocket 支持

### v0.2.0 — 智能化 🔄
- [ ] 多 Agent 协作框架
- [ ] 可视化工作流构建器（拖拽式）
- [ ] 高级 RAG（混合搜索）
- [ ] 代码执行沙箱
- [ ] 网页浏览 Agent 工具
- [ ] 图像生成集成
- [ ] 语音转文字 / 文字转语音

### v0.3.0 — 生态系统 📋
- [ ] 插件市场
- [ ] 模型市场
- [ ] Agent 分享平台
- [ ] 工作流模板
- [ ] 社区中心
- [ ] 移动端 App（iOS/Android）

### v1.0.0 — 革命 🚀
- [ ] 分布式 AI（多节点）
- [ ] 联邦学习
- [ ] AI 模型微调 UI
- [ ] 企业级功能（SSO、RBAC、审计日志）
- [ ] 云同步（可选，加密）
- [ ] 硬件加速优化

---

## ❓ 常见问题

### Q: LocalMind 会把我的数据发送到外部吗？
**A: 不会。** LocalMind 被设计为 100% 本地运行。除非你明确配置，否则不会有任何数据离开你的机器。零遥测。

### Q: LocalMind 支持哪些模型？
**A:** LocalMind 支持所有 Ollama 支持的模型，包括 Llama 3、Mistral、Qwen、DeepSeek、Phi-3、Gemma 等数百个模型。也支持自定义 GGUF 模型。

### Q: 硬件要求是什么？
**A:**
- **最低配置**：8GB 内存，任何现代 CPU（会比较慢）
- **推荐配置**：16GB+ 内存，Apple Silicon M1+ 或 NVIDIA GPU
- **最佳配置**：32GB+ 内存，8GB+ 显存的独立 GPU

### Q: LocalMind 和 Ollama 有什么区别？
**A:** Ollama 是一个模型运行器。LocalMind 是一个完整的 AI 操作系统，它使用 Ollama（及其他后端）来运行模型，然后在此基础上增加了 Agent、插件、RAG、记忆、工作流和精美的 UI。

### Q: 可以在 LocalMind 中使用云模型（OpenAI、Anthropic）吗？
**A:** 可以，LocalMind 支持多个模型提供商。你可以对隐私敏感的任务使用本地模型，需要更强算力时使用云模型。每个 Agent 可以单独配置。

### Q: LocalMind 是免费的吗？
**A:** 是的！LocalMind 在 MIT 许可证下 100% 免费开源。没有隐藏费用，没有高级付费层级，没有数据收集。

---

## 🙏 致谢

- [Ollama](https://github.com/ollama/ollama) — 优秀的本地模型运行器
- [Hugging Face](https://huggingface.co/) — 开放 AI 社区
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 应用框架灵感来源
- [LlamaIndex](https://github.com/run-llama/llama_index) — RAG 管道灵感来源
- [Open WebUI](https://github.com/open-webui/open-webui) — 精美 UI 灵感来源
- [Sentence Transformers](https://www.sbert.net/) — 嵌入模型

---

## 📜 许可证

LocalMind 基于 [MIT 许可证](LICENSE) 发布。

---

<div align="center">

**由 LocalMind 社区用 ❤️ 打造**

*"AI 的未来是本地的、私有的、自由的。"*

[⭐ 在 GitHub 上给我们加星](https://github.com/song-chaoyang/localmind-ai) — 这比你想象的更有帮助！

</div>

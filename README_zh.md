<div align="center">

<img src="assets/images/hero-logo.svg" alt="NexusMind Logo" width="120"/>

# NexusMind

**你的第二大脑。永远在线。自动进化。**

开源 AI 助手 -- 记住一切，从每次交互中学习，在你睡觉时持续工作。

<img src="assets/images/install-demo.svg" alt="Install Demo" width="600"/>

`pip install nexusmind && nexusmind`

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Self-Hosted](https://img.shields.io/badge/Deploy-Self_Hosted-9cf?logo=linux&logoColor=white)
![Multi-Provider](https://img.shields.io/badge/LLM-Multi_Provider-orange)
![No GPU Required](https://img.shields.io/badge/GPU-Not_Required-success)

---

<img src="assets/images/feature-grid.svg" alt="Feature Grid" width="700"/>

</div>

## 为什么选择 NexusMind？

每次关闭浏览器标签页，AI 助手就会重置。**NexusMind 不会。**

### 痛点

- ChatGPT 每次会话都会忘记你的项目上下文
- Claude Code 不会记住你的编码风格
- 没有任何 AI 工具能从你的重复工作流中学习
- 你无法安排 AI 任务在你离开时自动运行

### 解决方案

**NexusMind 融合了 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 和 [Claw Code](https://github.com/ultraworkers/claw-code) 的最佳理念：**

| 功能 | Hermes Agent | Claw Code | **NexusMind** |
|------|:------------:|:---------:|:-------------:|
| 持久化记忆 | :white_check_mark: | :x: | :white_check_mark: |
| 自动技能进化 | :white_check_mark: | :x: | :white_check_mark: |
| 离线任务调度 | :white_check_mark: | :x: | :white_check_mark: |
| 多智能体协作 | :x: | :white_check_mark: | :white_check_mark: |
| 多 LLM 提供商 | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 精美 Web UI | :white_check_mark: | :x: | :white_check_mark: |
| 零配置启动 | :x: | :x: | :white_check_mark: |
| 自托管部署 | :white_check_mark: | :white_check_mark: | :white_check_mark: |

> **一句话总结：** NexusMind 是唯一一个同时提供持久化记忆、自动技能进化、离线任务调度和多智能体协作的开源 AI 助手 -- 全部集成在一个精美、可自托管的包中。

---

## 核心功能

### 持久化记忆

<img src="assets/images/memory-system.svg" alt="Memory System" width="500" align="right"/>

NexusMind 记住关于你和你的项目的一切：

- 你的技术栈、编码风格和个人偏好
- 跨会话的项目上下文
- 自动提取实体（人物、项目、技术）
- 语义搜索所有记忆

<br clear="right"/>

### 自动技能进化

<img src="assets/images/skill-evolution.svg" alt="Skill Evolution" width="500" align="right"/>

NexusMind 从每次交互中学习：

- 检测工作流中的重复模式
- 自动创建可复用技能
- 技能随使用不断改进（追踪成功率）
- 在不同实例间导出/导入技能
- 生成你的 **"技能 DNA"** 指纹

<br clear="right"/>

### 休眠模式代理

<img src="assets/images/scheduler-demo.svg" alt="Scheduler Demo" width="500" align="right"/>

安排任务在你睡觉时运行：

- "每天早上 9 点审查我的 PR"
- "每 30 分钟运行一次测试套件"
- "每周五发送周报摘要"
- 结果推送到 Telegram、Discord 或 Slack

<br clear="right"/>

### 多智能体协作

<img src="assets/images/multi-agent.svg" alt="Multi-Agent System" width="500" align="right"/>

专业智能体协同工作：

- **Coder Agent** -- 代码生成、代码审查、调试
- **Research Agent** -- 文档编写、网络调研
- **Data Agent** -- 数据分析、可视化
- 所有智能体共享同一记忆系统

<br clear="right"/>

### 多提供商支持

连接任意 LLM 提供商：

| 提供商 | 模型 | 费用 | 需要 GPU |
|--------|------|------|:--------:|
| **Ollama** | Llama 3, Mistral, Qwen, Phi | 免费 | :x: |
| **OpenAI** | GPT-4, GPT-4o | 付费 | :x: |
| **Anthropic** | Claude 3.5, Claude Opus | 付费 | :x: |
| **OpenRouter** | 100+ 模型 | 不等 | :x: |

一条命令切换提供商 -- 无需修改任何代码。

---

## 快速开始

### 前置要求

- Python 3.10 或更高版本
- 一个 LLM 提供商（Ollama 用于免费本地推理，或云提供商的 API Key）

### 安装

```bash
pip install nexusmind
```

### 使用 Ollama 启动（免费本地，无需 GPU）

```bash
# 1. 安装 Ollama: https://ollama.ai
# 2. 拉取模型
ollama pull llama3

# 3. 启动 NexusMind
nexusmind start
```

### 使用 OpenAI 启动

```bash
export OPENAI_API_KEY=sk-...
nexusmind start --provider openai --model gpt-4o
```

### 使用 Anthropic 启动

```bash
export ANTHROPIC_API_KEY=sk-ant-...
nexusmind start --provider anthropic --model claude-opus-4-20250514
```

打开 **http://localhost:3000** 开始聊天！

---

## 文档

### CLI 命令

```bash
nexusmind start            # 启动 Web UI + API 服务
nexusmind chat             # 交互式终端聊天
nexusmind model list       # 列出可用模型
nexusmind model pull       # 拉取模型 (Ollama)
nexusmind memory search    # 搜索记忆
nexusmind skill list       # 列出已学习技能
nexusmind skill export     # 导出技能为 JSON
nexusmind skill import     # 从 JSON 导入技能
nexusmind schedule list    # 列出定时任务
nexusmind schedule add     # 添加定时任务
nexusmind ingest <file>    # 将文档导入记忆
```

### API 端点

```bash
# 聊天
POST   /api/v1/chat              # 发送消息
POST   /api/v1/chat/stream       # 流式响应 (SSE)

# 模型
GET    /api/v1/models            # 列出可用模型
GET    /api/v1/models/{id}       # 获取模型详情

# 记忆
GET    /api/v1/memory            # 浏览记忆
POST   /api/v1/memory/search     # 语义搜索
DELETE /api/v1/memory/{id}       # 删除记忆

# 技能
GET    /api/v1/skills            # 列出已学习技能
GET    /api/v1/skills/{id}       # 获取技能详情
POST   /api/v1/skills/export     # 导出技能
POST   /api/v1/skills/import     # 导入技能

# 调度器
GET    /api/v1/scheduler/tasks   # 列出定时任务
POST   /api/v1/scheduler/tasks   # 创建定时任务
DELETE /api/v1/scheduler/tasks/{id}  # 删除任务

# 文档导入
POST   /api/v1/ingest            # 将文档导入记忆
```

### 配置

NexusMind 开箱即用，零配置即可运行。如需高级设置，在项目根目录创建 `nexusmind.yaml`：

```yaml
provider:
  name: ollama
  model: llama3
  base_url: http://localhost:11434

memory:
  max_memories: 10000
  embedding_model: all-MiniLM-L6-v2

scheduler:
  enabled: true
  timezone: UTC

server:
  host: 0.0.0.0
  port: 3000

notifications:
  telegram:
    bot_token: ""
    chat_id: ""
  discord:
    webhook_url: ""
```

---

## 架构

NexusMind 采用模块化、可扩展的架构设计：

```
nexusmind/
├── api/            # FastAPI 服务 & WebSocket 端点
├── core/
│   ├── engine.py       # 主编排引擎
│   ├── memory.py       # 持久化记忆（向量搜索）
│   ├── skills.py       # 自动技能进化系统
│   ├── scheduler.py    # 基于 Cron 的任务调度器
│   ├── agents.py       # 多智能体协作
│   ├── providers.py    # LLM 提供商抽象层
│   └── rag.py          # 检索增强生成
├── static/         # Web UI (HTML/CSS/JS)
├── utils/          # 共享工具
└── cli.py          # 命令行界面
```

**核心设计理念：**

- **提供商无关**：无需修改代码即可切换任意 LLM
- **记忆优先**：每次交互都丰富共享知识库
- **技能驱动**：重复模式自动转化为可复用、可改进的技能
- **事件溯源**：所有操作和决策的完整审计日志

---

## 路线图

- [ ] 插件系统（自定义智能体）
- [ ] 语音输入/输出支持
- [ ] 移动端响应式 Web UI
- [ ] Docker 一键部署
- [ ] 技能市场（与社区共享技能）
- [ ] Git 集成（自动提交、PR 审查）
- [ ] MCP（Model Context Protocol）支持

---

## 贡献

我们欢迎各种贡献！无论是修复 Bug、添加新功能还是改进文档。

### 入门指南

1. Fork 本仓库
2. 克隆你的 Fork：`git clone https://github.com/your-username/nexusmind.git`
3. 安装开发依赖：`pip install -e ".[dev]"`
4. 运行测试：`pytest tests/`
5. 创建功能分支：`git checkout -b feature/my-feature`
6. 提交并推送：`git commit -m "Add my feature" && git push`
7. 发起 Pull Request

### 代码规范

- 遵循 PEP 8
- 所有函数添加类型注解
- 为新功能编写测试
- 保持 PR 小而聚焦

---

## 许可证

本项目基于 **MIT 许可证** -- 个人和商业使用均免费。

详见 [LICENSE](LICENSE)。

---

## 灵感来源

NexusMind 站在巨人的肩膀上：

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) -- 持久化记忆与技能进化理念
- [Claw Code](https://github.com/ultraworkers/claw-code) -- 自主多智能体编码
- [Open WebUI](https://github.com/open-webui/open-webui) -- 精美的 AI 界面设计
- [MemGPT](https://github.com/cpacker/memgpt) -- 虚拟上下文管理
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) -- 自主 AI 智能体框架

---

<div align="center">

**由 NexusMind 贡献者用 :heart: 打造**

如果你觉得这个项目有用，请给我们一个 :star:！

</div>

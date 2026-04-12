# Contributing to LocalMind

First of all, thank you for considering contributing to LocalMind! 🎉

LocalMind is built by the community, and every contribution — whether it's a bug fix, new feature, documentation improvement, or plugin — makes LocalMind better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Coding Standards](#coding-standards)
- [Plugin Development](#plugin-development)
- [Agent Development](#agent-development)
- [Documentation](#documentation)
- [Recognition](#recognition)

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating. We are committed to providing a welcoming and inclusive experience for everyone.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/localmind.git
   cd localmind
   ```
3. **Create** a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```
4. **Install** development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
5. **Run** tests to verify your setup:
   ```bash
   pytest
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) (for local model support)
- Node.js 18+ (for Web UI development)
- Git

### Project Structure

```
localmind/
├── src/localmind/       # Main source code
│   ├── core/            # Core engine
│   ├── models/          # Model management
│   ├── agents/          # Agent system
│   ├── plugins/         # Plugin system
│   ├── api/             # API layer
│   ├── ui/              # Web UI
│   └── utils/           # Utilities
├── tests/               # Test files
├── examples/            # Example code
├── docs/                # Documentation
└── scripts/             # Utility scripts
```

### Running in Development Mode

```bash
# Start the development server with auto-reload
localmind start --dev

# Run with verbose logging
localmind start --log-level debug

# Run only the API server
localmind start --headless --port 8080 --reload
```

## How to Contribute

### Types of Contributions

1. **Bug Fixes** 🐛 — Fix issues in existing functionality
2. **New Features** ✨ — Add new capabilities
3. **Documentation** 📚 — Improve guides, tutorials, API docs
4. **Plugins** 🔌 — Create new plugins
5. **Agents** 🤖 — Build new agents
6. **Translations** 🌍 — Translate to your language
7. **Tests** 🧪 — Improve test coverage
8. **Performance** ⚡ — Optimize speed and resource usage

### Reporting Bugs

Found a bug? Please [open an issue](https://github.com/song-chaoyang/localmind-ai/issues/new?template=bug_report.md) with:

1. **Clear title** describing the bug
2. **Steps to reproduce** the issue
3. **Expected behavior** vs. actual behavior
4. **Environment details** (OS, Python version, model used)
5. **Logs** or error messages (if any)
6. **Screenshots** (if applicable)

### Suggesting Features

Have an idea? Please [open an issue](https://github.com/song-chaoyang/localmind-ai/issues/new?template=feature_request.md) with:

1. **Clear description** of the proposed feature
2. **Motivation** — Why would this be useful?
3. **Use cases** — How would people use it?
4. **Possible implementation** (optional)

### Pull Request Guidelines

1. **Keep it small** — Small, focused PRs are easier to review
2. **Add tests** — Ensure new code is covered by tests
3. **Update docs** — Update relevant documentation
4. **Follow conventions** — Match the existing code style
5. **One PR per concern** — Don't mix bug fixes with new features
6. **Write good commit messages** — Use conventional commits format:
   ```
   feat: add new agent type
   fix: resolve memory leak in chat
   docs: update plugin development guide
   test: add integration tests for RAG pipeline
   refactor: simplify model manager
   ```
7. **Pass all checks** — Ensure CI passes before requesting review

## Coding Standards

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [type hints](https://docs.python.org/3/library/typing.html) for all function signatures
- Write [docstrings](https://peps.python.org/pep-0257/) for all public classes and functions
- Maximum line length: 100 characters
- Use `isort` for import sorting and `black` for formatting

```bash
# Auto-format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

### Testing

- Write tests for all new functionality
- Aim for >80% code coverage
- Use `pytest` with `pytest-asyncio` for async tests
- Follow the Arrange-Act-Assert pattern

```python
import pytest
from localmind.core import Engine

class TestEngine:
    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        engine = Engine()
        assert engine is not None
        assert engine.config is not None

    @pytest.mark.asyncio
    async def test_chat_response(self):
        """Test that chat returns a response."""
        engine = Engine()
        response = await engine.chat("Hello")
        assert response is not None
        assert len(response.text) > 0
```

## Plugin Development

See [docs/plugin-development.md](docs/plugin-development.md) for the complete plugin development guide.

Quick start:

```python
from localmind.plugins import Plugin, plugin_metadata

@plugin_metadata(
    name="my-plugin",
    version="1.0.0",
    description="My awesome plugin"
)
class MyPlugin(Plugin):
    def on_load(self):
        self.logger.info("Plugin loaded!")

    def register_tools(self):
        return [{"name": "my_tool", "function": self.my_tool}]

    def my_tool(self, query: str) -> str:
        return f"Result: {query}"
```

## Agent Development

See [docs/agent-development.md](docs/agent-development.md) for the complete agent development guide.

Quick start:

```python
from localmind.agents import BaseAgent, agent_metadata

@agent_metadata(
    name="my-agent",
    description="My custom agent",
    capabilities=["research", "analysis"]
)
class MyAgent(BaseAgent):
    async def execute(self, task: str, context: dict) -> str:
        # Your agent logic here
        result = await self.think(task, context)
        return result
```

## Documentation

Documentation improvements are always welcome! Please:

1. Check for existing docs in the `docs/` directory
2. Follow the existing documentation style
3. Include code examples where possible
4. Test any code snippets you add

## Recognition

Contributors are recognized in:

- The [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- Release notes for significant contributions
- The project README for major contributions

Thank you for making LocalMind better! 🙏

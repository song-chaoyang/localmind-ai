# Contributing to NexusMind

Thank you for your interest in contributing to NexusMind! This guide will help you get started.

## Development Environment Setup

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/your-username/nexusmind.git
   cd nexusmind
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[all,dev]"
   ```

3. **Verify the setup:**

   ```bash
   pytest tests/ -v
   ```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Before submitting a PR, run:

```bash
ruff check src/
ruff format src/
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
type(scope): description

feat(core): add new memory provider
fix(api): resolve connection timeout issue
docs: update README installation instructions
test(core): add unit tests for scheduler
chore: update dependencies
```

Common types:
- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation changes
- `test` - Test additions or modifications
- `chore` - Maintenance tasks
- `refactor` - Code refactoring
- `perf` - Performance improvements

## Pull Request Process

1. Create a feature branch from `main`:

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Make your changes, write tests, and ensure all tests pass.

3. Commit using conventional commit format.

4. Push to your fork and open a Pull Request against `main`.

5. Ensure CI checks pass (lint + test).

6. Address any review feedback.

## Reporting Issues

Please use our [bug report](.github/ISSUE_TEMPLATE/bug_report.md) and [feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates when opening issues.

## Questions?

Feel free to open an issue with the `question` label if you have any questions about contributing.

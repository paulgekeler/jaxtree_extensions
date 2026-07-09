# Contributing to Jaxtree Extensions

Welcome! Thank you for considering contributing to `jaxtree-extensions`.

This project uses a modern `uv` based setup and dev workflow.

## Development Setup

We use **`uv`** to manage virtual environments, dependencies, and formatting tasks.

### Prerequisite
Ensure `uv` is installed on your system. If not:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environment Setup
Clone the repository and synchronize the environment (including all development and documentation dependencies):
```bash
uv sync --all-groups
```
This automatically sets up a `.venv` virtual environment in the project directory.

---

## Code Quality Standards

Before submitting a pull request, please ensure your changes satisfy our quality checks.

### Code Formatting & Linting (`ruff`)
We use `ruff` to enforce standard formatting and docstring styling (Google convention).

* **Check code formatting and style**:
  ```bash
  uv run ruff check .
  ```
* **Auto-fix style violations and format code**:
  ```bash
  uv run ruff check --fix .
  uv run ruff format .
  ```

### Type Checking (`mypy`)
We enforce strict static type-checking across the codebase.
```bash
uv run mypy jaxtree_extensions
```

### Running Unit Tests (`pytest`)
All utilities must be covered by comprehensive unit tests, including integration checks for custom pytree structures like Equinox modules.
```bash
uv run pytest
```

---

## Documentation

API documentation is generated using MkDocs and Griffe from docstrings.

* **Build the documentation**:
  ```bash
  uv run --group docs mkdocs build
  ```
* **Preview the documentation locally**:
  ```bash
  uv run --group docs mkdocs serve
  ```

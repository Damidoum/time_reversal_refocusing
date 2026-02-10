# Agent Coding Guidelines for Time Reversal Project

This document outlines the coding standards, workflows, and commands for agents working in this repository.

## 1. Project Structure & Environment

- **Root Directory**: The project uses a `src` layout.
- **Package Name**: `time_reversal` (located in `src/time_reversal/`).
- **Dependency Management**: `pyproject.toml` is the source of truth.
- **Python Version**: >= 3.10

### Key Locations
- `src/time_reversal/`: Source code.
- `tests/`: Unit and integration tests.
- `pyproject.toml`: Configuration for build, linting, and testing.

## 2. Development Commands

Agents must use the following commands to verify changes.

### Installation
```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Testing
**Framework**: `pytest`

- **Run all tests**:
  ```bash
  pytest
  ```
- **Run a single test file**:
  ```bash
  pytest tests/test_model.py
  ```
- **Run a specific test case** (Crucial for verifying specific fixes):
  ```bash
  pytest tests/test_model.py::test_forward_pass
  ```
- **Run with coverage**:
  ```bash
  pytest --cov=time_reversal
  ```

### Linting & Formatting
**Tools**: `ruff` (unified linter/formatter), `mypy` (static type checking).

- **Format code**:
  ```bash
  ruff format .
  ```
- **Lint code** (and fix auto-fixable issues):
  ```bash
  ruff check --fix .
  ```
- **Type check**:
  ```bash
  mypy src/
  ```

## 3. Code Style & Conventions

### General Principles
- **Functional/Stateless**: Prefer pure functions where possible, especially for data transformations.
- **Modularity**: Keep files focused. If a file exceeds 300 lines, consider refactoring.
- **Explicit is better than implicit**: Avoid "magic" behavior or heavy reliance on `**kwargs` without typing.

### Imports
Follow this order (enforced by `ruff`):
1. Standard library imports
2. Third-party library imports (e.g., `numpy`, `torch`)
3. Local application imports

```python
import os
from typing import Optional

import torch
import torch.nn as nn

from time_reversal.utils import helper_function
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `DiffusionModel`)
- **Functions/Variables**: `snake_case` (e.g., `calculate_loss`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_BATCH_SIZE`)
- **Private members**: Prefix with `_` (e.g., `_internal_cache`)

### Type Hinting
- **Strict Typing**: All function signatures must be typed.
- **Tensor Typing**: Use `torch.Tensor` or `numpy.ndarray`. Where helpful, indicate shape in docstrings.

```python
def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (B, C, H, W)
        t: Time step
    """
    ...
```

### Error Handling
- Use specific exception types (e.g., `ValueError`, `RuntimeError`) rather than generic `Exception`.
- meaningful error messages are mandatory.

```python
if x.shape[0] != self.in_channels:
    raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[0]}")
```

### Documentation
- Use Google-style or NumPy-style docstrings.
- Docstrings are required for all public classes and functions.

## 4. Generative Modeling Specifics
- **Reproducibility**: Allow passing `torch.Generator` or setting seeds for stochastic functions.
- **Device Management**: Do not hardcode `.cuda()` or `.cpu()`. Use `device` arguments or `tensor.to(device)`.

## 5. Agent Workflow
1. **Explore**: Use `ls`, `grep`, and `read` to understand the context.
2. **Plan**: Propose changes before executing.
3. **Edit**: Use `edit` or `write` to modify code.
4. **Verify**: ALWAYS run the relevant test command (e.g., `pytest tests/test_specific.py`) after changes.
5. **Lint**: Run `ruff check .` to ensure no style regressions.

# Contributing to PlugLLM

Thank you for considering contributing to PlugLLM! This document provides guidelines and instructions for contributing to this project.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Adding New LLM Providers](#adding-new-llm-providers)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guide](#style-guide)
- [Questions](#questions)

## Code of Conduct

This project follows a simple principle: **be kind and respectful**. We welcome contributions from everyone, regardless of experience level. Please treat others with respect and provide constructive feedback.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of LLM APIs

### Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/plugllm.git
cd plugllm
```

## Development Setup

### 1. Create a Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n plugllm python=3.9
conda activate plugllm
```

### 2. Install Development Dependencies
```bash
pip install -e ".[dev]"
```

### 3. Set Up Pre-commit Hooks (Optional)
```bash
pip install pre-commit
pre-commit install
```

## Making Changes

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes
- Keep changes focused on a single feature/bug
- Write clear, documented code
- Update tests if needed
- Update documentation

### 3. Test Your Changes
```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_base.py

# Run with coverage
pytest --cov=plugllm tests/
```

## Adding New LLM Providers

To add support for a new LLM provider:

1. Create a new file in `plugllm/providers/` (e.g., `new_provider.py`)
2. Create a class that inherits from `BaseLLM`
3. Implement all abstract methods:
   - `generate()`
   - `agenerate()`
   - `stream()`
   - `astream()`

Example structure:
```python
from plugllm.base import BaseLLM, ChatResponse, Message

class NewProviderLLM(BaseLLM):
    """LLM provider for NewProvider"""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)
        # Initialize your client here
    
    def generate(self, prompt, **kwargs) -> ChatResponse:
        # Implement sync generation
        pass
    
    async def agenerate(self, prompt, **kwargs) -> ChatResponse:
        # Implement async generation
        pass
    
    def stream(self, prompt, **kwargs):
        # Implement sync streaming
        pass
    
    async def astream(self, prompt, **kwargs):
        # Implement async streaming
        pass
```

4. Add tests in `tests/test_new_provider.py`
5. Update `README.md` with the new provider

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with detailed output
pytest -v

# Run with coverage report
pytest --cov=plugllm --cov-report=html
```

### Writing Tests
- Place tests in the `tests/` directory
- Name test files with `test_` prefix (e.g., `test_base.py`)
- Use descriptive test names
- Mock external API calls

Example test:
```python
def test_ask_with_conversation():
    llm = MockLLM(model="test-model")
    response = llm.ask("Hello", conversational_name="test")
    assert response.content == "Hello back!"
```

## Submitting Changes

### 1. Commit Your Changes
```bash
git add .
git commit -m "Brief description of changes"
```

Use clear commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for code refactoring

Example:
```
feat: add conversational_name parameter to ask method
fix: resolve memory leak in streaming responses
docs: update README with new usage examples
```

### 2. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 3. Create a Pull Request
1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the pull request template
5. Submit the pull request

### Pull Request Guidelines
- Keep PRs focused on a single change
- Include tests for new features
- Update documentation
- Ensure all tests pass
- Respond to review comments

## Style Guide

### Code Style
We follow PEP 8 with some exceptions:
- Use 4 spaces for indentation
- Maximum line length: 88 characters (like Black)
- Use descriptive variable names

### Docstrings
Use Google-style docstrings:
```python
def ask(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> ChatResponse:
    """
    Send a prompt to the LLM and get a response.
    
    Args:
        user_prompt: The user's question or prompt
        system_prompt: Optional system instruction
        **kwargs: Additional generation parameters
    
    Returns:
        ChatResponse object containing the response
    
    Raises:
        ValueError: If user_prompt is empty
    """
    pass
```

### Type Hints
Always use type hints for function parameters and returns:
```python
def process_message(message: str, max_tokens: int = 100) -> List[str]:
    pass
```

## Questions or Need Help?

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Contact**: Open an issue with your question

## License
By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to PlugLLM! 🚀

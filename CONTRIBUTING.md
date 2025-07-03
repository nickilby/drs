# Contributing to vCenter DRS Compliance Dashboard

Thank you for your interest in contributing to the vCenter DRS Compliance Dashboard! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vcenter-drs-compliance-dashboard.git
   cd vcenter-drs-compliance-dashboard
   ```
3. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or for bug fixes:
   git checkout -b fix/your-bug-fix-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip
- A vCenter server for testing
- Database (PostgreSQL, MySQL, or SQLite)

### Local Development Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt  # If available
   ```

3. **Set up environment variables**:
   ```bash
   export VCENTER_HOST=your-vcenter-server
   export VCENTER_USERNAME=your-username
   export VCENTER_PASSWORD=your-password
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=vcenter_drs
   export DB_USER=your-db-user
   export DB_PASSWORD=your-db-password
   ```

4. **Run the application**:
   ```bash
   streamlit run vcenter_drs/app.py
   ```

## Making Changes

### Code Structure

- `vcenter_drs/` - Main application package
  - `app.py` - Streamlit application entry point
  - `db/` - Database layer
  - `rules/` - DRS compliance rules engine
  - `utils/` - Utility functions
- `tests/` - Test files
- `docs/` - Documentation

### Guidelines

1. **Keep changes focused**: Each commit should address a single issue or feature
2. **Write clear commit messages**: Use conventional commit format
3. **Add tests**: New features should include tests
4. **Update documentation**: Keep docs in sync with code changes

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=vcenter_drs

# Run specific test file
pytest tests/test_specific_module.py

# Run tests with verbose output
pytest -v
```

### Test Guidelines

- Write unit tests for new functionality
- Ensure all tests pass before submitting
- Aim for good test coverage
- Use meaningful test names
- Mock external dependencies

### Type Checking

```bash
# Run mypy type checking
mypy vcenter_drs/

# Run mypy with strict mode
mypy --strict vcenter_drs/
```

## Submitting Changes

### Pull Request Process

1. **Ensure your code is ready**:
   - All tests pass
   - Code follows style guidelines
   - Documentation is updated
   - No new warnings or errors

2. **Push your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   - Use the provided PR template
   - Link related issues
   - Add appropriate labels
   - Request reviews from maintainers

4. **Address feedback**:
   - Respond to review comments
   - Make requested changes
   - Update the PR as needed

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(dashboard): add new compliance metric visualization
fix(db): resolve connection timeout issue
docs(readme): update installation instructions
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Keep functions small and focused
- Use meaningful variable and function names
- Add docstrings for public functions and classes

### Code Formatting

```bash
# Format code with black
black vcenter_drs/

# Sort imports with isort
isort vcenter_drs/

# Check code style with flake8
flake8 vcenter_drs/
```

## Documentation

### Documentation Guidelines

- Keep documentation up to date with code changes
- Use clear, concise language
- Include code examples where helpful
- Update README.md for significant changes
- Add inline comments for complex logic

### Documentation Structure

- `README.md` - Project overview and quick start
- `CONTRIBUTING.md` - This file
- `CODE_OF_CONDUCT.md` - Community guidelines
- `docs/` - Detailed documentation
- Inline docstrings - Code documentation

## Reporting Issues

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for solutions
3. **Try to reproduce** the issue locally
4. **Gather relevant information** (logs, error messages, etc.)

### Issue Template

Use the provided issue template when reporting issues. Include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment information
- Error messages and logs
- Screenshots (if applicable)

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the docs folder and README

## Recognition

Contributors will be recognized in the project's README and release notes. Significant contributions may be eligible for maintainer status.

Thank you for contributing to the vCenter DRS Compliance Dashboard! 
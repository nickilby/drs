# Makefile for vCenter DRS
# Provides common commands for development, testing, and deployment

.PHONY: help install dev-install test lint format clean run check-connectivity init-db

# Default target
help:
	@echo "Available commands:"
	@echo "  install        - Install production dependencies"
	@echo "  dev-install    - Install development dependencies"
	@echo "  test           - Run tests with coverage"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black and isort"
	@echo "  clean          - Clean up generated files"
	@echo "  run            - Run the Streamlit dashboard"
	@echo "  check-connectivity - Test vCenter connectivity"
	@echo "  init-db        - Initialize database schema"
	@echo "  collect-data   - Collect data from vCenter"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev-install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --cov=vcenter_drs --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 vcenter_drs/ --max-line-length=88 --extend-ignore=E203,W503
	mypy vcenter_drs/ --ignore-missing-imports

# Format code
format:
	black vcenter_drs/ --line-length=88
	isort vcenter_drs/ --profile=black

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -f last_collection_time.txt

# Run the Streamlit dashboard
run:
	streamlit run streamlit_app.py

# Test vCenter connectivity
check-connectivity:
	python main.py check

# Initialize database schema
init-db:
	python -c "from db.metrics_db import MetricsDB; db = MetricsDB(); db.connect(); db.init_schema(); db.close()"

# Collect data from vCenter
collect-data:
	python main.py

# Setup development environment
setup-dev: dev-install format lint
	@echo "Development environment setup complete!"

# Pre-commit checks
pre-commit: format lint test
	@echo "All pre-commit checks passed!"

# Create virtual environment
venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Install in development mode
dev: venv
	@echo "Activating virtual environment and installing dependencies..."
	. venv/bin/activate && pip install -e ".[dev]"
	@echo "Development environment ready!" 
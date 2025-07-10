"""
Basic tests for AI Optimizer Module

This module tests the basic structure and imports of the AI optimizer.
"""

import pytest
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ai_optimizer_import():
    """Test that we can import the AI optimizer module."""
    try:
        from ai_optimizer import __version__, __author__
        assert __version__ == "1.0.0"
        assert __author__ == "vCenter DRS Team"
        print("✅ AI optimizer module imports successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import AI optimizer module: {e}")


def test_ai_config_import():
    """Test that we can import the AI config module."""
    try:
        from ai_optimizer.config import AIConfig
        config = AIConfig()
        assert config is not None
        print("✅ AI config module imports successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import AI config module: {e}")


def test_ai_config_prometheus_url():
    """Test that Prometheus URL is correctly configured."""
    try:
        from ai_optimizer.config import AIConfig
        config = AIConfig()
        url = config.get_prometheus_url()
        assert 'prometheus.zengenti.com' in url
        assert ':9090' in url
        print(f"✅ Prometheus URL configured correctly: {url}")
    except Exception as e:
        pytest.fail(f"Failed to get Prometheus URL: {e}")


def test_ai_config_validation():
    """Test that AI config validation works."""
    try:
        from ai_optimizer.config import AIConfig
        config = AIConfig()
        is_valid = config.validate()
        assert is_valid is True
        print("✅ AI config validation passes")
    except Exception as e:
        pytest.fail(f"Failed to validate AI config: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_ai_optimizer_import()
    test_ai_config_import()
    test_ai_config_prometheus_url()
    test_ai_config_validation()
    print("🎉 All basic AI optimizer tests passed!") 
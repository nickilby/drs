"""
Tests for the configuration module.

This module contains unit tests for the configuration management functionality.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch

from vcenter_drs.config import ConfigManager, VCenterConfig, DatabaseConfig, AppConfig


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_init_with_default_path(self):
        """Test ConfigManager initialization with default path."""
        with patch('os.path.exists', return_value=False):
            config = ConfigManager()
            assert config.config_path.endswith('credentials.json')
    
    def test_init_with_custom_path(self):
        """Test ConfigManager initialization with custom path."""
        custom_path = "/custom/path/config.json"
        config = ConfigManager(custom_path)
        assert config.config_path == custom_path
    
    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        test_config = {
            "host": "test-vcenter.com",
            "username": "test-user",
            "password": "test-pass",
            "db_host": "test-db.com",
            "db_user": "db-user",
            "db_password": "db-pass",
            "db_database": "test_db"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name
        
        try:
            # Clear environment variables to ensure file values are used
            env_vars_to_clear = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_DATABASE", 
                                "VCENTER_HOST", "VCENTER_USERNAME", "VCENTER_PASSWORD"]
            with patch.dict(os.environ, {}, clear=True):
                config = ConfigManager(config_path)
                assert config.vcenter.host == "test-vcenter.com"
                assert config.database.host == "test-db.com"
        finally:
            os.unlink(config_path)
    
    def test_environment_variable_override(self):
        """Test that environment variables override file configuration."""
        test_config = {
            "host": "file-vcenter.com",
            "username": "file-user",
            "password": "file-pass"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name
        
        try:
            with patch.dict(os.environ, {
                'VCENTER_HOST': 'env-vcenter.com',
                'VCENTER_USERNAME': 'env-user'
            }):
                config = ConfigManager(config_path)
                assert config.vcenter.host == "env-vcenter.com"
                assert config.vcenter.username == "env-user"
                assert config.vcenter.password == "file-pass"  # Not overridden
        finally:
            os.unlink(config_path)
    
    def test_validate_success(self):
        """Test configuration validation with valid config."""
        test_config = {
            "host": "test-vcenter.com",
            "username": "test-user",
            "password": "test-pass",
            "db_host": "test-db.com",
            "db_user": "db-user",
            "db_password": "db-pass",
            "db_database": "test_db"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name
        
        try:
            config = ConfigManager(config_path)
            assert config.validate() is True
        finally:
            os.unlink(config_path)
    
    def test_validate_failure(self):
        """Test configuration validation with missing required fields."""
        test_config = {
            "host": "test-vcenter.com",
            # Missing username, password, and database config
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name
        
        try:
            config = ConfigManager(config_path)
            assert config.validate() is False
        finally:
            os.unlink(config_path)


class TestVCenterConfig:
    """Test cases for VCenterConfig dataclass."""
    
    def test_default_values(self):
        """Test VCenterConfig default values."""
        config = VCenterConfig(
            host="test.com",
            username="user",
            password="pass"
        )
        assert config.host == "test.com"
        assert config.username == "user"
        assert config.password == "pass"
        assert config.port == 443
        assert config.ssl_verify is False
    
    def test_custom_values(self):
        """Test VCenterConfig with custom values."""
        config = VCenterConfig(
            host="test.com",
            username="user",
            password="pass",
            port=8443,
            ssl_verify=True
        )
        assert config.port == 8443
        assert config.ssl_verify is True


class TestDatabaseConfig:
    """Test cases for DatabaseConfig dataclass."""
    
    def test_default_values(self):
        """Test DatabaseConfig default values."""
        config = DatabaseConfig(
            host="test.com",
            user="user",
            password="pass",
            database="test_db"
        )
        assert config.host == "test.com"
        assert config.user == "user"
        assert config.password == "pass"
        assert config.database == "test_db"
        assert config.port == 3306
    
    def test_custom_port(self):
        """Test DatabaseConfig with custom port."""
        config = DatabaseConfig(
            host="test.com",
            user="user",
            password="pass",
            database="test_db",
            port=3307
        )
        assert config.port == 3307


class TestAppConfig:
    """Test cases for AppConfig dataclass."""
    
    def test_default_values(self):
        """Test AppConfig default values."""
        config = AppConfig()
        assert config.collection_interval == 300
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.batch_size == 100
        assert config.max_connections == 10
        assert config.log_level == "INFO"
        assert config.log_file is None
        assert config.page_title == "vCenter DRS Compliance Dashboard"
        assert config.page_icon == "🏢"
        assert config.layout == "wide" 
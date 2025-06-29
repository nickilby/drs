"""
Configuration Management for vCenter DRS

This module centralizes all configuration settings for the vCenter DRS system.
It provides a clean interface for accessing configuration values and supports
both file-based and environment variable configuration.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str
    user: str
    password: str
    database: str
    port: int = 3306


@dataclass
class VCenterConfig:
    """vCenter configuration settings."""
    host: str
    username: str
    password: str
    port: int = 443
    ssl_verify: bool = False


@dataclass
class AppConfig:
    """Application configuration settings."""
    # Data collection settings
    collection_interval: int = 300  # seconds
    max_retries: int = 3
    timeout: int = 30
    
    # Performance settings
    batch_size: int = 100
    max_connections: int = 10
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # UI settings
    page_title: str = "vCenter DRS Compliance Dashboard"
    page_icon: str = "🏢"
    layout: str = "wide"


class ConfigManager:
    """
    Centralized configuration manager for the vCenter DRS system.
    
    This class handles loading configuration from multiple sources:
    1. Environment variables
    2. Configuration file
    3. Default values
    
    The configuration is loaded once and cached for performance.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (defaults to credentials.json)
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 'credentials.json'
            )
        
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Load from file
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
        else:
            self._config = {}
        
        # Override with environment variables
        self._config.update({
            'host': os.getenv('VCENTER_HOST', self._config.get('host')),
            'username': os.getenv('VCENTER_USERNAME', self._config.get('username')),
            'password': os.getenv('VCENTER_PASSWORD', self._config.get('password')),
            'db_host': os.getenv('DB_HOST', self._config.get('db_host')),
            'db_user': os.getenv('DB_USER', self._config.get('db_user')),
            'db_password': os.getenv('DB_PASSWORD', self._config.get('db_password')),
            'db_database': os.getenv('DB_DATABASE', self._config.get('db_database')),
        })
    
    @property
    def vcenter(self) -> VCenterConfig:
        """Get vCenter configuration."""
        return VCenterConfig(
            host=self._config.get('host', ''),
            username=self._config.get('username', ''),
            password=self._config.get('password', ''),
            port=int(self._config.get('port', 443)),
            ssl_verify=bool(self._config.get('ssl_verify', False))
        )
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig(
            host=self._config.get('db_host', 'localhost'),
            user=self._config.get('db_user', ''),
            password=self._config.get('db_password', ''),
            database=self._config.get('db_database', 'vcenter_drs'),
            port=int(self._config.get('db_port', 3306))
        )
    
    @property
    def app(self) -> AppConfig:
        """Get application configuration."""
        return AppConfig(
            collection_interval=int(self._config.get('collection_interval', 300)),
            max_retries=int(self._config.get('max_retries', 3)),
            timeout=int(self._config.get('timeout', 30)),
            batch_size=int(self._config.get('batch_size', 100)),
            max_connections=int(self._config.get('max_connections', 10)),
            log_level=self._config.get('log_level', 'INFO'),
            log_file=self._config.get('log_file'),
            page_title=self._config.get('page_title', 'vCenter DRS Compliance Dashboard'),
            page_icon=self._config.get('page_icon', '🏢'),
            layout=self._config.get('layout', 'wide')
        )
    
    def reload(self) -> None:
        """Reload configuration from sources."""
        self._load_config()
    
    def validate(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        vcenter = self.vcenter
        database = self.database
        
        required_fields = [
            (vcenter.host, 'vCenter host'),
            (vcenter.username, 'vCenter username'),
            (vcenter.password, 'vCenter password'),
            (database.host, 'Database host'),
            (database.user, 'Database user'),
            (database.password, 'Database password'),
            (database.database, 'Database name'),
        ]
        
        for value, name in required_fields:
            if not value:
                print(f"Missing required configuration: {name}")
                return False
        
        return True


# Global configuration instance
config = ConfigManager() 
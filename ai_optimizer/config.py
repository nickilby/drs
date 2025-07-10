"""
Configuration Management for AI Optimizer

This module centralizes all configuration settings for the AI optimization system.
It provides settings for Prometheus connection, analysis windows, and ML parameters.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PrometheusConfig:
    """Prometheus configuration settings."""
    url: str = "http://prometheus.zengenti.com"
    port: int = 9090
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class AnalysisConfig:
    """Analysis window configuration."""
    cpu_trend_hours: int = 1
    storage_trend_days: int = 3
    ram_trend_hours: int = 6
    io_trend_days: int = 2
    ready_time_window: int = 1  # hours


@dataclass
class MLConfig:
    """Machine learning configuration."""
    model_save_path: str = "ai_optimizer/models"
    training_episodes: int = 1000
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    batch_size: int = 32


@dataclass
class OptimizationConfig:
    """Optimization parameters."""
    ideal_host_usage_min: float = 0.30  # 30%
    ideal_host_usage_max: float = 0.70  # 70%
    ram_priority_weight: float = 1.0
    ready_time_priority_weight: float = 0.8
    cpu_priority_weight: float = 0.6
    io_priority_weight: float = 0.4
    max_recommendations: int = 10


class AIConfig:
    """
    Centralized configuration manager for the AI optimization system.
    
    This class handles loading configuration from multiple sources:
    1. Environment variables
    2. Configuration file
    3. Default values
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from environment variables and defaults."""
        # Prometheus configuration
        self.prometheus = PrometheusConfig(
            url=os.getenv('PROMETHEUS_URL', 'http://prometheus.zengenti.com'),
            port=int(os.getenv('PROMETHEUS_PORT', '9090')),
            timeout=int(os.getenv('PROMETHEUS_TIMEOUT', '30')),
            retry_attempts=int(os.getenv('PROMETHEUS_RETRY_ATTEMPTS', '3'))
        )
        
        # Analysis configuration
        self.analysis = AnalysisConfig(
            cpu_trend_hours=int(os.getenv('CPU_TREND_HOURS', '1')),
            storage_trend_days=int(os.getenv('STORAGE_TREND_DAYS', '3')),
            ram_trend_hours=int(os.getenv('RAM_TREND_HOURS', '6')),
            io_trend_days=int(os.getenv('IO_TREND_DAYS', '2')),
            ready_time_window=int(os.getenv('READY_TIME_WINDOW', '1'))
        )
        
        # ML configuration
        self.ml = MLConfig(
            model_save_path=os.getenv('MODEL_SAVE_PATH', 'ai_optimizer/models'),
            training_episodes=int(os.getenv('TRAINING_EPISODES', '1000')),
            learning_rate=float(os.getenv('LEARNING_RATE', '0.001')),
            discount_factor=float(os.getenv('DISCOUNT_FACTOR', '0.95')),
            exploration_rate=float(os.getenv('EXPLORATION_RATE', '0.1')),
            batch_size=int(os.getenv('BATCH_SIZE', '32'))
        )
        
        # Optimization configuration
        self.optimization = OptimizationConfig(
            ideal_host_usage_min=float(os.getenv('IDEAL_HOST_USAGE_MIN', '0.30')),
            ideal_host_usage_max=float(os.getenv('IDEAL_HOST_USAGE_MAX', '0.70')),
            ram_priority_weight=float(os.getenv('RAM_PRIORITY_WEIGHT', '1.0')),
            ready_time_priority_weight=float(os.getenv('READY_TIME_PRIORITY_WEIGHT', '0.8')),
            cpu_priority_weight=float(os.getenv('CPU_PRIORITY_WEIGHT', '0.6')),
            io_priority_weight=float(os.getenv('IO_PRIORITY_WEIGHT', '0.4')),
            max_recommendations=int(os.getenv('MAX_RECOMMENDATIONS', '10'))
        )
    
    def validate(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate Prometheus URL
            if not self.prometheus.url:
                print("Missing Prometheus URL")
                return False
            
            # Validate analysis windows
            if self.analysis.cpu_trend_hours <= 0:
                print("CPU trend hours must be positive")
                return False
            
            if self.analysis.storage_trend_days <= 0:
                print("Storage trend days must be positive")
                return False
            
            # Validate optimization parameters
            if not (0 <= self.optimization.ideal_host_usage_min <= 1):
                print("Ideal host usage min must be between 0 and 1")
                return False
            
            if not (0 <= self.optimization.ideal_host_usage_max <= 1):
                print("Ideal host usage max must be between 0 and 1")
                return False
            
            if self.optimization.ideal_host_usage_min >= self.optimization.ideal_host_usage_max:
                print("Ideal host usage min must be less than max")
                return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def get_prometheus_url(self) -> str:
        """Get the full Prometheus URL."""
        return f"{self.prometheus.url}:{self.prometheus.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'prometheus': {
                'url': self.prometheus.url,
                'port': self.prometheus.port,
                'timeout': self.prometheus.timeout,
                'retry_attempts': self.prometheus.retry_attempts
            },
            'analysis': {
                'cpu_trend_hours': self.analysis.cpu_trend_hours,
                'storage_trend_days': self.analysis.storage_trend_days,
                'ram_trend_hours': self.analysis.ram_trend_hours,
                'io_trend_days': self.analysis.io_trend_days,
                'ready_time_window': self.analysis.ready_time_window
            },
            'ml': {
                'model_save_path': self.ml.model_save_path,
                'training_episodes': self.ml.training_episodes,
                'learning_rate': self.ml.learning_rate,
                'discount_factor': self.ml.discount_factor,
                'exploration_rate': self.ml.exploration_rate,
                'batch_size': self.ml.batch_size
            },
            'optimization': {
                'ideal_host_usage_min': self.optimization.ideal_host_usage_min,
                'ideal_host_usage_max': self.optimization.ideal_host_usage_max,
                'ram_priority_weight': self.optimization.ram_priority_weight,
                'ready_time_priority_weight': self.optimization.ready_time_priority_weight,
                'cpu_priority_weight': self.optimization.cpu_priority_weight,
                'io_priority_weight': self.optimization.io_priority_weight,
                'max_recommendations': self.optimization.max_recommendations
            }
        }


# Global configuration instance
ai_config = AIConfig() 
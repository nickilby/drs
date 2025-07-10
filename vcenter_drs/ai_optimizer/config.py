"""Configuration Management for AI Optimizer"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PrometheusConfig:
    """Prometheus connection configuration"""
    url: str = "http://prometheus.zengenti.com"
    port: int = 9090
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class AnalysisConfig:
    """Analysis window configuration"""
    cpu_trend_hours: int = 1
    storage_trend_days: int = 3
    ram_trend_hours: int = 1
    io_trend_days: int = 3
    ready_time_window: int = 1


@dataclass
class OptimizationConfig:
    """Optimization parameters"""
    ideal_host_usage_min: float = 0.3  # 30%
    ideal_host_usage_max: float = 0.7  # 70%
    max_recommendations: int = 5
    cpu_priority_weight: float = 1.0
    ram_priority_weight: float = 1.0
    ready_time_priority_weight: float = 1.0
    io_priority_weight: float = 1.0


@dataclass
class MLConfig:
    """Machine learning model configuration"""
    training_episodes: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 32
    exploration_rate: float = 0.1


class AIConfig:
    """Main configuration class for AI Optimizer"""
    
    def __init__(self):
        self.prometheus = PrometheusConfig()
        self.analysis = AnalysisConfig()
        self.optimization = OptimizationConfig()
        self.ml = MLConfig()

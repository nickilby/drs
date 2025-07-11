"""
AI Optimizer Module for vCenter DRS

This module provides AI-driven VM placement optimization using reinforcement learning.
It analyzes performance metrics from Prometheus to make informed placement decisions
while respecting existing compliance rules.
"""

__version__ = "1.0.0"
__author__ = "vCenter DRS Team"

from .data_collector import PrometheusDataCollector
from .config import AIConfig

__all__ = [
    "PrometheusDataCollector",
    "AIConfig"
] 
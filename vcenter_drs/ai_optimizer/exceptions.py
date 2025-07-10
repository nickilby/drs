"""Custom exceptions for AI Optimizer"""


class AIOptimizerError(Exception):
    """Base exception for AI Optimizer errors"""
    pass


class PrometheusConnectionError(AIOptimizerError):
    """Raised when unable to connect to Prometheus"""
    pass


class DataCollectionError(AIOptimizerError):
    """Raised when data collection fails"""
    pass


class ModelTrainingError(AIOptimizerError):
    """Raised when model training fails"""
    pass


class OptimizationError(AIOptimizerError):
    """Raised when optimization fails"""
    pass


class ConfigurationError(AIOptimizerError):
    """Raised when configuration is invalid"""
    pass

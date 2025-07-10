"""
Custom Exceptions for AI Optimizer

This module defines custom exception classes for the AI optimization system.
These exceptions provide more specific error handling and better debugging
capabilities than generic Python exceptions.
"""

from typing import Optional, Any


class AIOptimizerError(Exception):
    """Base exception class for AI optimizer errors."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        self.details = details


class PrometheusConnectionError(AIOptimizerError):
    """Raised when connection to Prometheus fails."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        """
        Initialize the Prometheus connection error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Prometheus connection failed: {message}", details)


class DataCollectionError(AIOptimizerError):
    """Raised when data collection from Prometheus fails."""
    
    def __init__(self, metric_type: str, object_name: str, message: str, details: Optional[Any] = None):
        """
        Initialize the data collection error.
        
        Args:
            metric_type: Type of metric being collected
            object_name: Name of the object (VM/Host)
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Failed to collect {metric_type} data for '{object_name}': {message}", details)
        self.metric_type = metric_type
        self.object_name = object_name


class ModelTrainingError(AIOptimizerError):
    """Raised when ML model training fails."""
    
    def __init__(self, model_name: str, message: str, details: Optional[Any] = None):
        """
        Initialize the model training error.
        
        Args:
            model_name: Name of the model being trained
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Model training failed for '{model_name}': {message}", details)
        self.model_name = model_name


class PlacementRecommendationError(AIOptimizerError):
    """Raised when placement recommendation generation fails."""
    
    def __init__(self, vm_name: str, message: str, details: Optional[Any] = None):
        """
        Initialize the placement recommendation error.
        
        Args:
            vm_name: Name of the VM being placed
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Placement recommendation failed for VM '{vm_name}': {message}", details)
        self.vm_name = vm_name


class ConfigurationError(AIOptimizerError):
    """Raised when AI configuration is invalid or missing."""
    
    def __init__(self, config_key: str, message: str, details: Optional[Any] = None):
        """
        Initialize the configuration error.
        
        Args:
            config_key: Configuration key that caused the error
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Configuration error for '{config_key}': {message}", details)
        self.config_key = config_key


class ValidationError(AIOptimizerError):
    """Raised when data validation fails."""
    
    def __init__(self, field: str, value: Any, message: str, details: Optional[Any] = None):
        """
        Initialize the validation error.
        
        Args:
            field: Field name that failed validation
            value: Value that failed validation
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Validation failed for field '{field}' with value '{value}': {message}", details)
        self.field = field
        self.value = value


class PredictionError(AIOptimizerError):
    """Raised when ML model prediction fails."""
    
    def __init__(self, model_name: str, message: str, details: Optional[Any] = None):
        """
        Initialize the prediction error.
        
        Args:
            model_name: Name of the model making predictions
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Prediction failed for model '{model_name}': {message}", details)
        self.model_name = model_name 
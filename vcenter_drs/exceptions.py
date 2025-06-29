"""
Custom Exceptions for vCenter DRS

This module defines custom exception classes for the vCenter DRS system.
These exceptions provide more specific error handling and better debugging
capabilities than generic Python exceptions.
"""

from typing import Optional, Any


class VCenterDRSError(Exception):
    """Base exception class for vCenter DRS errors."""
    
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


class VCenterConnectionError(VCenterDRSError):
    """Raised when connection to vCenter Server fails."""
    
    def __init__(self, host: str, message: str, details: Optional[Any] = None):
        """
        Initialize the connection error.
        
        Args:
            host: vCenter Server hostname
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Failed to connect to vCenter {host}: {message}", details)
        self.host = host


class DatabaseConnectionError(VCenterDRSError):
    """Raised when database connection fails."""
    
    def __init__(self, host: str, database: str, message: str, details: Optional[Any] = None):
        """
        Initialize the database connection error.
        
        Args:
            host: Database server hostname
            database: Database name
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Failed to connect to database {database} on {host}: {message}", details)
        self.host = host
        self.database = database


class ConfigurationError(VCenterDRSError):
    """Raised when configuration is invalid or missing."""
    
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


class RuleEvaluationError(VCenterDRSError):
    """Raised when rule evaluation fails."""
    
    def __init__(self, rule_name: str, message: str, details: Optional[Any] = None):
        """
        Initialize the rule evaluation error.
        
        Args:
            rule_name: Name of the rule that failed
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Rule evaluation failed for '{rule_name}': {message}", details)
        self.rule_name = rule_name


class DataCollectionError(VCenterDRSError):
    """Raised when data collection from vCenter fails."""
    
    def __init__(self, object_type: str, object_name: str, message: str, details: Optional[Any] = None):
        """
        Initialize the data collection error.
        
        Args:
            object_type: Type of object being collected (VM, Host, etc.)
            object_name: Name of the object
            message: Error message
            details: Additional error details
        """
        super().__init__(f"Failed to collect data for {object_type} '{object_name}': {message}", details)
        self.object_type = object_type
        self.object_name = object_name


class ValidationError(VCenterDRSError):
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
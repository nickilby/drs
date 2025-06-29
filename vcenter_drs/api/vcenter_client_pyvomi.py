"""
vCenter API Client Module

This module provides a client for connecting to VMware vCenter Server using the pyVmomi library.
It handles authentication, connection management, and provides a clean interface for vCenter operations.

Example:
    client = VCenterPyVmomiClient()
    si = client.connect()
    print(si.CurrentTime())
    client.disconnect()
"""

import os
import json
import ssl
from typing import Optional, Dict, Any
from pyVim.connect import SmartConnect, Disconnect
from pyVim.connect import SmartConnectNoSSL, Disconnect
from pyVmomi import vim


class VCenterPyVmomiClient:
    """
    A client for connecting to VMware vCenter Server.
    
    This class handles authentication and connection management to vCenter Server.
    It supports both SSL and non-SSL connections and can load credentials from
    a JSON file or accept them as constructor parameters.
    
    Attributes:
        host (str): The vCenter Server hostname or IP address
        username (str): The username for vCenter authentication
        password (str): The password for vCenter authentication
        si (vim.ServiceInstance): The vCenter service instance connection
    """
    
    def __init__(
        self, 
        host: Optional[str] = None, 
        username: Optional[str] = None, 
        password: Optional[str] = None, 
        credentials_path: Optional[str] = None
    ) -> None:
        """
        Initialize the vCenter client.
        
        Args:
            host: vCenter Server hostname or IP address
            username: vCenter username
            password: vCenter password
            credentials_path: Path to JSON file containing credentials
            
        Note:
            If credentials are not provided directly, they will be loaded from
            the credentials file. The credentials file should contain:
            - host: vCenter Server address
            - username: vCenter username  
            - password: vCenter password
        """
        if not (host and username and password):
            if credentials_path is None:
                credentials_path = os.path.join(
                    os.path.dirname(__file__), '..', 'credentials.json'
                )
            
            with open(os.path.abspath(credentials_path), 'r') as f:
                creds: Dict[str, str] = json.load(f)
            
            self.host = creds['host']
            self.username = creds['username']
            self.password = creds['password']
        else:
            self.host = host
            self.username = username
            self.password = password
        
        self.si: Optional[vim.ServiceInstance] = None

    def connect(self) -> vim.ServiceInstance:
        """
        Connect to vCenter Server.
        
        Returns:
            vim.ServiceInstance: The connected service instance
            
        Raises:
            Exception: If connection fails
            
        Note:
            This method creates an unverified SSL context for development.
            In production, you should use proper SSL certificates.
        """
        try:
            # Create unverified SSL context for development
            context = ssl._create_unverified_context()
            
            self.si = SmartConnect(
                host=self.host,
                user=self.username,
                pwd=self.password,
                sslContext=context
            )
            
            return self.si
            
        except Exception as e:
            raise Exception(f"Failed to connect to vCenter {self.host}: {e}")

    def disconnect(self) -> None:
        """
        Disconnect from vCenter Server.
        
        This method safely disconnects the service instance if it exists.
        """
        if self.si:
            Disconnect(self.si)
            self.si = None

    def __enter__(self) -> 'VCenterPyVmomiClient':
        """Context manager entry point."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        self.disconnect()

# Example usage:
# client = VCenterPyVmomiClient()
# si = client.connect()
# print(si.CurrentTime())
# client.disconnect() 
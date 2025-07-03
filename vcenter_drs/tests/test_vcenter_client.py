"""
Unit tests for vCenter client functionality.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from vcenter_drs.api.vcenter_client_pyvomi import VCenterPyVmomiClient


class TestVCenterPyVmomiClient:
    """Test cases for VCenterPyVmomiClient."""

    def test_init_with_parameters(self):
        """Test client initialization with direct parameters."""
        client = VCenterPyVmomiClient(
            host="test-host",
            username="test-user",
            password="test-pass"
        )
        
        assert client.host == "test-host"
        assert client.username == "test-user"
        assert client.password == "test-pass"
        assert client.si is None

    def test_init_with_credentials_file(self):
        """Test client initialization with credentials file."""
        credentials = {
            "host": "file-host",
            "username": "file-user",
            "password": "file-pass"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(credentials, f)
            credentials_path = f.name
        
        try:
            client = VCenterPyVmomiClient(credentials_path=credentials_path)
            
            assert client.host == "file-host"
            assert client.username == "file-user"
            assert client.password == "file-pass"
        finally:
            os.unlink(credentials_path)

    def test_init_with_default_credentials_path(self):
        """Test client initialization with default credentials path."""
        credentials = {
            "host": "default-host",
            "username": "default-user",
            "password": "default-pass"
        }
        
        # Create credentials file in the expected location
        api_dir = os.path.dirname(os.path.abspath(__file__))
        credentials_path = os.path.join(api_dir, '..', 'credentials.json')
        
        # Backup existing credentials if they exist
        backup_path = None
        if os.path.exists(credentials_path):
            backup_path = credentials_path + '.backup'
            os.rename(credentials_path, backup_path)
        
        try:
            with open(credentials_path, 'w') as f:
                json.dump(credentials, f)
            
            client = VCenterPyVmomiClient()
            
            assert client.host == "default-host"
            assert client.username == "default-user"
            assert client.password == "default-pass"
        finally:
            # Clean up
            if os.path.exists(credentials_path):
                os.unlink(credentials_path)
            if backup_path and os.path.exists(backup_path):
                os.rename(backup_path, credentials_path)

    @patch('vcenter_drs.api.vcenter_client_pyvomi.SmartConnect')
    def test_connect_success(self, mock_smart_connect):
        """Test successful connection to vCenter."""
        # Setup mock
        mock_si = Mock()
        mock_smart_connect.return_value = mock_si
        
        client = VCenterPyVmomiClient(
            host="test-host",
            username="test-user",
            password="test-pass"
        )
        
        # Test connection
        result = client.connect()
        
        # Verify
        assert result == mock_si
        assert client.si == mock_si
        mock_smart_connect.assert_called_once()

    @patch('vcenter_drs.api.vcenter_client_pyvomi.SmartConnect')
    def test_connect_failure(self, mock_smart_connect):
        """Test connection failure."""
        # Setup mock to raise exception
        mock_smart_connect.side_effect = Exception("Connection failed")
        
        client = VCenterPyVmomiClient(
            host="test-host",
            username="test-user",
            password="test-pass"
        )
        
        # Test connection failure
        with pytest.raises(Exception, match="Failed to connect to vCenter test-host: Connection failed"):
            client.connect()

    @patch('vcenter_drs.api.vcenter_client_pyvomi.Disconnect')
    def test_disconnect_with_connection(self, mock_disconnect):
        """Test disconnection when connected."""
        client = VCenterPyVmomiClient(
            host="test-host",
            username="test-user",
            password="test-pass"
        )
        client.si = Mock()
        
        client.disconnect()
        
        mock_disconnect.assert_called_once_with(client.si)
        assert client.si is None

    def test_disconnect_without_connection(self):
        """Test disconnection when not connected."""
        client = VCenterPyVmomiClient(
            host="test-host",
            username="test-user",
            password="test-pass"
        )
        client.si = None
        
        # Should not raise any exception
        client.disconnect()
        assert client.si is None

    @patch('vcenter_drs.api.vcenter_client_pyvomi.SmartConnect')
    def test_context_manager(self, mock_smart_connect):
        """Test client as context manager."""
        mock_si = Mock()
        mock_smart_connect.return_value = mock_si
        
        client = VCenterPyVmomiClient(
            host="test-host",
            username="test-user",
            password="test-pass"
        )
        
        with client as ctx_client:
            assert ctx_client == client
            assert client.si == mock_si

    def test_context_manager_cleanup(self):
        """Test context manager cleanup on exit."""
        client = VCenterPyVmomiClient(
            host="test-host",
            username="test-user",
            password="test-pass"
        )
        client.si = Mock()
        
        # Simulate context manager exit
        client.__exit__(None, None, None)
        
        assert client.si is None 
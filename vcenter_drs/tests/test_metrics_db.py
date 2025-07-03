"""
Unit tests for database functionality.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from vcenter_drs.db.metrics_db import MetricsDB


class TestMetricsDB:
    """Test cases for MetricsDB."""

    def test_init_with_parameters(self):
        """Test database initialization with direct parameters."""
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        
        assert db.host == "test-host"
        assert db.user == "test-user"
        assert db.password == "test-pass"
        assert db.database == "test-db"
        assert db.conn is None

    def test_init_with_credentials_file(self):
        """Test database initialization with credentials file."""
        credentials = {
            "db_host": "file-host",
            "db_user": "file-user",
            "db_password": "file-pass",
            "db_database": "file-db"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(credentials, f)
            credentials_path = f.name
        
        try:
            db = MetricsDB(credentials_path=credentials_path)
            
            assert db.host == "file-host"
            assert db.user == "file-user"
            assert db.password == "file-pass"
            assert db.database == "file-db"
        finally:
            os.unlink(credentials_path)

    @patch('vcenter_drs.db.metrics_db.mysql.connector.connect')
    def test_connect_success(self, mock_connect):
        """Test successful database connection."""
        # Setup mock
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        
        # Test connection
        db.connect()
        
        # Verify
        assert db.conn == mock_conn
        mock_connect.assert_called_once_with(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )

    @patch('vcenter_drs.db.metrics_db.mysql.connector.connect')
    def test_connect_failure(self, mock_connect):
        """Test database connection failure."""
        from mysql.connector import Error
        
        # Setup mock to raise exception
        mock_connect.side_effect = Error("Connection failed")
        
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        
        # Test connection failure
        db.connect()
        
        # Should handle error gracefully
        assert db.conn is None

    @patch('vcenter_drs.db.metrics_db.mysql.connector.connect')
    def test_init_schema_success(self, mock_connect):
        """Test successful schema initialization."""
        # Setup mocks
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        db.connect()
        
        # Test schema initialization
        db.init_schema()
        
        # Verify
        assert mock_cursor.execute.call_count >= 6  # At least 6 table creation statements
        mock_conn.commit.assert_called_once()

    @patch('vcenter_drs.db.metrics_db.mysql.connector.connect')
    def test_init_schema_without_connection(self, mock_connect):
        """Test schema initialization without connection."""
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        # Don't connect
        
        # Test schema initialization
        db.init_schema()
        
        # Should handle gracefully without error
        mock_connect.assert_not_called()

    @patch('vcenter_drs.db.metrics_db.mysql.connector.connect')
    def test_close_with_connection(self, mock_connect):
        """Test closing database connection."""
        # Setup mocks
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        db.connect()
        
        # Test close
        db.close()
        
        # Verify
        mock_conn.close.assert_called_once()
        assert db.conn is None

    def test_close_without_connection(self):
        """Test closing without connection."""
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        db.conn = None
        
        # Should not raise any exception
        db.close()
        assert db.conn is None

    @patch('vcenter_drs.db.metrics_db.mysql.connector.connect')
    def test_context_manager(self, mock_connect):
        """Test database as context manager."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        
        with db as ctx_db:
            assert ctx_db == db
            assert db.conn == mock_conn

    def test_context_manager_cleanup(self):
        """Test context manager cleanup on exit."""
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        db.conn = Mock()
        
        # Simulate context manager exit
        db.__exit__(None, None, None)
        
        assert db.conn is None

    @patch('vcenter_drs.db.metrics_db.mysql.connector.connect')
    def test_add_exception(self, mock_connect):
        """Test adding exception to database."""
        # Setup mocks
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        db.connect()
        
        # Test data
        exception_data = {
            'rule_type': 'test-rule',
            'alias': 'test-alias',
            'affected_vms': ['vm1', 'vm2'],
            'cluster': 'test-cluster',
            'rule_hash': 'test-hash',
            'reason': 'test reason'
        }
        
        # Test adding exception
        db.add_exception(exception_data)
        
        # Verify
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch('vcenter_drs.db.metrics_db.mysql.connector.connect')
    def test_get_exceptions(self, mock_connect):
        """Test retrieving exceptions from database."""
        # Setup mocks
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock fetchall to return test data
        mock_cursor.fetchall.return_value = [
            (1, 'test-rule', 'test-alias', '["vm1", "vm2"]', 'test-cluster', 'test-hash', '2023-01-01', 'test reason')
        ]
        
        mock_connect.return_value = mock_conn
        
        db = MetricsDB(
            host="test-host",
            user="test-user",
            password="test-pass",
            database="test-db"
        )
        db.connect()
        
        # Test getting exceptions
        exceptions = db.get_exceptions()
        
        # Verify
        assert len(exceptions) == 1
        assert exceptions[0]['rule_type'] == 'test-rule'
        assert exceptions[0]['alias'] == 'test-alias'
        assert exceptions[0]['affected_vms'] == ['vm1', 'vm2'] 
"""
Pytest configuration and common fixtures for vCenter DRS tests.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch


@pytest.fixture
def temp_credentials_file():
    """Create a temporary credentials file for testing."""
    credentials = {
        "host": "test-vcenter.example.com",
        "username": "test_user",
        "password": "test_password",
        "db_host": "127.0.0.1",
        "db_user": "test_user",
        "db_password": "test_password",
        "db_database": "vcenter_drs_test"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(credentials, f)
        credentials_path = f.name
    
    yield credentials_path
    
    # Cleanup
    if os.path.exists(credentials_path):
        os.unlink(credentials_path)


@pytest.fixture
def mock_vcenter_connection():
    """Mock vCenter connection for testing."""
    with patch('vcenter_drs.api.vcenter_client_pyvomi.SmartConnect') as mock_connect:
        mock_si = Mock()
        mock_connect.return_value = mock_si
        yield mock_si


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    with patch('vcenter_drs.db.metrics_db.mysql.connector.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_conn, mock_cursor


@pytest.fixture
def sample_vm_data():
    """Sample VM data for testing."""
    return {
        "name": "test-vm-1",
        "host": "test-host-1",
        "cluster": "test-cluster",
        "dataset": "test-dataset",
        "cpu_usage": 25.5,
        "memory_usage": 60.2
    }


@pytest.fixture
def sample_host_data():
    """Sample host data for testing."""
    return {
        "name": "test-host-1",
        "cluster": "test-cluster",
        "cpu_usage": 45.2,
        "memory_usage": 70.1,
        "vm_count": 5
    }


@pytest.fixture
def sample_compliance_rules():
    """Sample compliance rules for testing."""
    return [
        {
            "type": "dataset-affinity",
            "name_pattern": "-dr-",
            "dataset_pattern": ["TRA"]
        },
        {
            "type": "anti-affinity",
            "level": "host",
            "role": "CACHE"
        },
        {
            "type": "affinity",
            "level": "cluster",
            "role": "WEB"
        }
    ]


@pytest.fixture
def sample_violations():
    """Sample compliance violations for testing."""
    return [
        {
            "type": "dataset-affinity",
            "alias": "test-alias-1",
            "cluster": "test-cluster",
            "affected_vms": ["vm1", "vm2"],
            "violation_text": "VMs should be on TRA dataset"
        },
        {
            "type": "anti-affinity",
            "alias": "test-alias-2",
            "cluster": "test-cluster",
            "affected_vms": ["vm3", "vm4"],
            "violation_text": "VMs should not be on same host"
        }
    ]


@pytest.fixture
def mock_prometheus_metrics():
    """Mock Prometheus metrics for testing."""
    with patch('vcenter_drs.streamlit_app.Gauge') as mock_gauge, \
         patch('vcenter_drs.streamlit_app.Counter') as mock_counter, \
         patch('vcenter_drs.streamlit_app.Histogram') as mock_histogram:
        
        # Create mock metric instances
        mock_service_up = Mock()
        mock_vm_count = Mock()
        mock_host_count = Mock()
        mock_rule_violations = Mock()
        mock_compliance_duration = Mock()
        
        # Configure mock gauge to return our mock instances
        mock_gauge.side_effect = lambda name, description, **kwargs: {
            'vcenter_drs_service_up': mock_service_up,
            'vcenter_drs_vm_count': mock_vm_count,
            'vcenter_drs_host_count': mock_host_count,
            'vcenter_drs_rule_violations_total': mock_rule_violations
        }.get(name, Mock())
        
        mock_histogram.return_value = mock_compliance_duration
        
        yield {
            'service_up': mock_service_up,
            'vm_count': mock_vm_count,
            'host_count': mock_host_count,
            'rule_violations': mock_rule_violations,
            'compliance_duration': mock_compliance_duration
        }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    
    # Create test directories if they don't exist
    test_dirs = ['vcenter_drs/tests/tmp', 'vcenter_drs/tests/data']
    for test_dir in test_dirs:
        os.makedirs(test_dir, exist_ok=True)
    
    yield
    
    # Cleanup after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


@pytest.fixture
def temp_test_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path) 
"""
Tests for AI Data Collector Module

This module tests the PrometheusDataCollector class and its methods.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import requests
from ai_optimizer.data_collector import PrometheusDataCollector
from ai_optimizer.config import AIConfig
from ai_optimizer.exceptions import PrometheusConnectionError


class TestPrometheusDataCollector:
    """Test cases for PrometheusDataCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AIConfig()
        self.collector = PrometheusDataCollector(self.config)
    
    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = AIConfig()
        collector = PrometheusDataCollector(config)
        assert collector.config == config
        assert collector.timeout == config.prometheus.timeout
    
    def test_init_without_config(self):
        """Test initialization without config (uses global)."""
        collector = PrometheusDataCollector()
        assert collector.config is not None
        assert isinstance(collector.config, AIConfig)
    
    def test_get_prometheus_url(self):
        """Test getting Prometheus URL from config."""
        url = self.collector.config.get_prometheus_url()
        assert 'prometheus.zengenti.com' in url
        assert ':9090' in url
    
    @patch('requests.Session.get')
    def test_query_prometheus_success(self, mock_get):
        """Test successful Prometheus query."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            'status': 'success',
            'data': {
                'result': [
                    {
                        'metric': {'vm_name': 'test-vm'},
                        'value': [1234567890, '50.5']
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.collector._query_prometheus('test_query')
        
        assert result['status'] == 'success'
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_query_prometheus_failure(self, mock_get):
        """Test Prometheus query failure."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        with pytest.raises(PrometheusConnectionError):
            self.collector._query_prometheus('test_query')
    
    @patch.object(PrometheusDataCollector, '_query_prometheus')
    def test_get_vm_cpu_trend(self, mock_query):
        """Test getting VM CPU trend."""
        # Mock response data
        mock_query.return_value = {
            'status': 'success',
            'data': {
                'result': [
                    {
                        'metric': {'vm_name': 'test-vm'},
                        'values': [
                            [1234567890, '50.5'],
                            [1234567950, '60.2'],
                            [1234568010, '45.8']
                        ]
                    }
                ]
            }
        }
        
        result = self.collector.get_vm_cpu_trend('test-vm', hours=1)
        
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], datetime) for item in result)
        assert all(isinstance(item[1], float) for item in result)
    
    @patch.object(PrometheusDataCollector, '_query_prometheus')
    def test_test_connection_success(self, mock_query):
        """Test successful connection test."""
        mock_query.return_value = {
            'status': 'success',
            'data': {'result': []}
        }
        
        result = self.collector.test_connection()
        
        assert result is True
    
    @patch.object(PrometheusDataCollector, '_query_prometheus')
    def test_test_connection_failure(self, mock_query):
        """Test failed connection test."""
        mock_query.side_effect = Exception("Connection failed")
        
        result = self.collector.test_connection()
        
        assert result is False 
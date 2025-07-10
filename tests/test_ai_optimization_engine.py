"""
Tests for AI Optimization Engine

This module tests the optimization engine functionality including
placement recommendations, host filtering, and business rules.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from ai_optimizer.optimization_engine import OptimizationEngine
from ai_optimizer.config import AIConfig
from ai_optimizer.data_collector import PrometheusDataCollector
from ai_optimizer.exceptions import PlacementRecommendationError


class TestOptimizationEngine:
    """Test the optimization engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AIConfig()
        self.data_collector = Mock(spec=PrometheusDataCollector)
        self.optimization_engine = OptimizationEngine(self.config, self.data_collector)
    
    def test_init(self):
        """Test optimization engine initialization."""
        assert self.optimization_engine.config == self.config
        assert self.optimization_engine.data_collector == self.data_collector
        assert self.optimization_engine.ml_engine is not None
    
    def test_get_vm_metrics(self):
        """Test VM metrics collection."""
        vm_name = "test-vm"
        
        # Mock trend data
        self.data_collector.get_vm_cpu_trend.return_value = [(1, 0.5), (2, 0.6)]
        self.data_collector.get_vm_ram_trend.return_value = [(1, 0.3), (2, 0.4)]
        self.data_collector.get_vm_ready_time_trend.return_value = [(1, 0.1), (2, 0.05)]
        self.data_collector.get_vm_io_trend.return_value = [(1, 0.2), (2, 0.3)]
        self.data_collector.get_storage_trends.return_value = [(1, 0.1), (2, 0.2)]
        
        metrics = self.optimization_engine.get_vm_metrics(vm_name)
        
        assert 'cpu_usage' in metrics
        assert 'ram_usage' in metrics
        assert 'ready_time' in metrics
        assert 'io_usage' in metrics
        assert 'storage_usage' in metrics
        assert 'cpu_trend' in metrics
        assert 'ram_trend' in metrics
        assert 'cpu_volatility' in metrics
        assert 'ram_volatility' in metrics
        
        # Check calculated values
        assert metrics['cpu_usage'] == 0.55  # Average of 0.5 and 0.6
        assert metrics['cpu_trend'] == 'increasing'  # 0.6 > 0.5
    
    def test_get_vm_metrics_fallback(self):
        """Test VM metrics collection with fallback values."""
        vm_name = "test-vm"
        
        # Mock empty trend data
        self.data_collector.get_vm_cpu_trend.return_value = []
        self.data_collector.get_vm_ram_trend.return_value = []
        self.data_collector.get_vm_ready_time_trend.return_value = []
        self.data_collector.get_vm_io_trend.return_value = []
        self.data_collector.get_storage_trends.return_value = []
        
        metrics = self.optimization_engine.get_vm_metrics(vm_name)
        
        assert metrics['cpu_usage'] == 0.0
        assert metrics['ram_usage'] == 0.0
        assert metrics['ready_time'] == 0.0
        assert metrics['io_usage'] == 0.0
        assert metrics['storage_usage'] == 0.0
        assert metrics['cpu_trend'] == 'stable'
        assert metrics['ram_trend'] == 'stable'
    
    def test_get_host_candidates(self):
        """Test host candidate collection."""
        # Mock all VMs metrics
        self.data_collector.get_all_vms_metrics.return_value = {
            'host-1': {'cluster': 'cluster-1'},
            'host-2': {'cluster': 'cluster-2'}
        }
        
        # Mock host performance metrics
        self.data_collector.get_host_performance_metrics.return_value = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3,
            'io_usage': 0.2,
            'ready_time': 0.02,
            'vm_count': 5,
            'utilization_score': 0.6
        }
        
        candidates = self.optimization_engine.get_host_candidates()
        
        assert len(candidates) == 2
        assert all('name' in candidate for candidate in candidates)
        assert all('cluster' in candidate for candidate in candidates)
        assert all('cpu_usage' in candidate for candidate in candidates)
        assert all('available_cpu' in candidate for candidate in candidates)
        assert all('available_ram' in candidate for candidate in candidates)
    
    def test_get_host_candidates_with_cluster_filter(self):
        """Test host candidate collection with cluster filter."""
        # Mock all VMs metrics
        self.data_collector.get_all_vms_metrics.return_value = {
            'host-1': {'cluster': 'cluster-1'},
            'host-2': {'cluster': 'cluster-2'}
        }
        
        # Mock host performance metrics
        self.data_collector.get_host_performance_metrics.return_value = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3,
            'io_usage': 0.2,
            'ready_time': 0.02,
            'vm_count': 5,
            'utilization_score': 0.6
        }
        
        candidates = self.optimization_engine.get_host_candidates('cluster-1')
        
        assert len(candidates) == 1
        assert candidates[0]['cluster'] == 'cluster-1'
    
    def test_filter_hosts_by_constraints(self):
        """Test host filtering by resource constraints."""
        hosts = [
            {
                'name': 'host-1',
                'cpu_usage': 0.3,
                'ram_usage': 0.2,
                'available_cpu': 0.7,
                'available_ram': 0.8
            },
            {
                'name': 'host-2',
                'cpu_usage': 0.8,
                'ram_usage': 0.9,
                'available_cpu': 0.2,
                'available_ram': 0.1
            }
        ]
        
        vm_metrics = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3
        }
        
        filtered_hosts = self.optimization_engine.filter_hosts_by_constraints(hosts, vm_metrics)
        
        # Only host-1 should pass (has enough resources and projected usage is in ideal range)
        assert len(filtered_hosts) == 1
        assert filtered_hosts[0]['name'] == 'host-1'
    
    def test_rank_hosts_by_ml_score(self):
        """Test host ranking using ML scores."""
        hosts = [
            {'name': 'host-1'},
            {'name': 'host-2'}
        ]
        
        vm_metrics = {
            'cpu_usage': 0.3,
            'ram_usage': 0.2,
            'ready_time': 0.05,
            'io_usage': 0.1
        }
        
        # Mock host metrics
        self.data_collector.get_host_performance_metrics.return_value = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3,
            'io_usage': 0.2,
            'ready_time': 0.02,
            'vm_count': 5,
            'utilization_score': 0.6
        }
        
        # Mock ML predictions
        self.optimization_engine.ml_engine.predict_placement_scores = Mock(return_value=[0.8, 0.6])
        
        host_scores = self.optimization_engine.rank_hosts_by_ml_score(hosts, vm_metrics)
        
        assert len(host_scores) == 2
        assert host_scores[0][1] == 0.8  # Highest score first
        assert host_scores[1][1] == 0.6
        assert host_scores[0][0]['name'] == 'host-1'
        assert host_scores[1][0]['name'] == 'host-2'
    
    def test_apply_business_rules(self):
        """Test business rules application."""
        host_scores = [
            ({'name': 'host-1', 'vm_count': 5, 'ready_time': 0.02, 'cpu_usage': 0.4, 'ram_usage': 0.4}, 0.8),
            ({'name': 'host-2', 'vm_count': 15, 'ready_time': 0.15, 'cpu_usage': 0.8, 'ram_usage': 0.3}, 0.7)
        ]
        
        vm_name = "test-vm"
        
        adjusted_scores = self.optimization_engine.apply_business_rules(host_scores, vm_name)
        
        assert len(adjusted_scores) == 2
        
        # host-1 should have higher adjusted score (better characteristics)
        assert adjusted_scores[0][1] > adjusted_scores[1][1]
        assert adjusted_scores[0][0]['name'] == 'host-1'
        assert adjusted_scores[1][0]['name'] == 'host-2'
    
    def test_generate_placement_recommendations_success(self):
        """Test successful placement recommendation generation."""
        vm_name = "test-vm"
        
        # Mock VM metrics
        self.optimization_engine.get_vm_metrics = Mock(return_value={
            'cpu_usage': 0.3,
            'ram_usage': 0.2,
            'ready_time': 0.05,
            'io_usage': 0.1,
            'storage_usage': 0.1
        })
        
        # Mock host candidates
        self.optimization_engine.get_host_candidates = Mock(return_value=[
            {
                'name': 'host-1',
                'cluster': 'cluster-1',
                'cpu_usage': 0.4,
                'ram_usage': 0.3,
                'io_usage': 0.2,
                'ready_time': 0.02,
                'vm_count': 5,
                'utilization_score': 0.6,
                'available_cpu': 0.6,
                'available_ram': 0.7
            }
        ])
        
        # Mock host filtering
        self.optimization_engine.filter_hosts_by_constraints = Mock(return_value=[
            {
                'name': 'host-1',
                'cluster': 'cluster-1',
                'cpu_usage': 0.4,
                'ram_usage': 0.3,
                'io_usage': 0.2,
                'ready_time': 0.02,
                'vm_count': 5,
                'utilization_score': 0.6,
                'available_cpu': 0.6,
                'available_ram': 0.7
            }
        ])
        
        # Mock ML ranking
        self.optimization_engine.rank_hosts_by_ml_score = Mock(return_value=[
            ({
                'name': 'host-1',
                'cluster': 'cluster-1',
                'cpu_usage': 0.4,
                'ram_usage': 0.3,
                'io_usage': 0.2,
                'ready_time': 0.02,
                'vm_count': 5,
                'utilization_score': 0.6,
                'available_cpu': 0.6,
                'available_ram': 0.7
            }, 0.8)
        ])
        
        # Mock business rules
        self.optimization_engine.apply_business_rules = Mock(return_value=[
            ({
                'name': 'host-1',
                'cluster': 'cluster-1',
                'cpu_usage': 0.4,
                'ram_usage': 0.3,
                'io_usage': 0.2,
                'ready_time': 0.02,
                'vm_count': 5,
                'utilization_score': 0.6,
                'available_cpu': 0.6,
                'available_ram': 0.7
            }, 0.8)
        ])
        
        recommendations = self.optimization_engine.generate_placement_recommendations(vm_name)
        
        assert len(recommendations) == 1
        recommendation = recommendations[0]
        
        assert recommendation['rank'] == 1
        assert recommendation['host_name'] == 'host-1'
        assert recommendation['cluster'] == 'cluster-1'
        assert recommendation['score'] == 0.8
        assert 'current_metrics' in recommendation
        assert 'projected_metrics' in recommendation
        assert 'vm_metrics' in recommendation
        assert 'reasoning' in recommendation
    
    def test_generate_placement_recommendations_no_candidates(self):
        """Test placement recommendation with no host candidates."""
        vm_name = "test-vm"
        
        # Mock empty host candidates
        self.optimization_engine.get_host_candidates = Mock(return_value=[])
        
        with pytest.raises(PlacementRecommendationError, match="No host candidates available"):
            self.optimization_engine.generate_placement_recommendations(vm_name)
    
    def test_generate_placement_recommendations_no_constraints_met(self):
        """Test placement recommendation with no hosts meeting constraints."""
        vm_name = "test-vm"
        
        # Mock host candidates
        self.optimization_engine.get_host_candidates = Mock(return_value=[
            {'name': 'host-1', 'available_cpu': 0.1, 'available_ram': 0.1}
        ])
        
        # Mock empty filtered hosts
        self.optimization_engine.filter_hosts_by_constraints = Mock(return_value=[])
        
        with pytest.raises(PlacementRecommendationError, match="No hosts meet resource constraints"):
            self.optimization_engine.generate_placement_recommendations(vm_name)
    
    def test_generate_reasoning(self):
        """Test reasoning generation for recommendations."""
        host = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3,
            'vm_count': 5,
            'ready_time': 0.02
        }
        
        vm_metrics = {
            'cpu_usage': 0.3,
            'ram_usage': 0.2,
            'ready_time': 0.05,
            'io_usage': 0.1
        }
        
        score = 0.8
        
        reasoning = self.optimization_engine._generate_reasoning(host, vm_metrics, score)
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert ';' in reasoning  # Multiple reasons separated by semicolons
    
    def test_train_models(self):
        """Test model training."""
        vms = [{'name': 'vm-1'}, {'name': 'vm-2'}]
        hosts = [{'name': 'host-1'}, {'name': 'host-2'}]
        
        # Mock ML engine training
        self.optimization_engine.ml_engine.train_models = Mock(return_value=True)
        
        result = self.optimization_engine.train_models(vms, hosts)
        
        assert result is True
        self.optimization_engine.ml_engine.train_models.assert_called_once_with(vms, hosts)
    
    def test_get_optimization_summary(self):
        """Test optimization summary generation."""
        summary = self.optimization_engine.get_optimization_summary()
        
        assert 'models_available' in summary
        assert 'configuration' in summary
        assert 'prometheus_connection' in summary
        assert 'last_training' in summary
        
        assert 'baseline_model' in summary['models_available']
        assert 'neural_network' in summary['models_available']
        assert 'ideal_host_usage_min' in summary['configuration']
        assert 'ideal_host_usage_max' in summary['configuration']
        assert 'max_recommendations' in summary['configuration']


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 
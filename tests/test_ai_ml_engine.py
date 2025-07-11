"""
Tests for AI ML Engine

This module tests the machine learning engine functionality including
model training, prediction, and data preprocessing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from ai_optimizer.ml_engine import MLModel, NeuralNetworkModel, MLEngine
from ai_optimizer.config import AIConfig
from ai_optimizer.data_collector import PrometheusDataCollector
from ai_optimizer.exceptions import ModelTrainingError, PredictionError


class TestMLModel:
    """Test the base ML model functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AIConfig()
        self.model = MLModel(self.config)
    
    def test_init(self):
        """Test model initialization."""
        assert self.model.config == self.config
        assert self.model.model is None
        assert not self.model.is_trained
        assert self.model.scaler is not None
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        data = [
            {
                'cpu_usage': 0.5,
                'ram_usage': 0.3,
                'ready_time': 0.1,
                'io_usage': 0.2,
                'host_cpu_usage': 0.6,
                'host_ram_usage': 0.4,
                'host_io_usage': 0.3,
                'host_ready_time': 0.05,
                'vm_count_on_host': 5,
                'host_utilization_score': 0.7,
                'placement_score': 0.8
            },
            {
                'cpu_usage': 0.7,
                'ram_usage': 0.5,
                'ready_time': 0.15,
                'io_usage': 0.4,
                'host_cpu_usage': 0.8,
                'host_ram_usage': 0.6,
                'host_io_usage': 0.5,
                'host_ready_time': 0.1,
                'vm_count_on_host': 8,
                'host_utilization_score': 0.9,
                'placement_score': 0.6
            }
        ]
        
        features, targets = self.model.preprocess_data(data)
        
        assert features.shape == (2, 10)
        assert targets.shape == (2,)
        assert np.allclose(targets, [0.8, 0.6])
    
    @patch('ai_optimizer.ml_engine.RandomForestRegressor')
    def test_train_success(self, mock_rf):
        """Test successful model training."""
        mock_model = Mock()
        mock_rf.return_value = mock_model
        
        data = [
            {
                'cpu_usage': 0.5,
                'ram_usage': 0.3,
                'ready_time': 0.1,
                'io_usage': 0.2,
                'host_cpu_usage': 0.6,
                'host_ram_usage': 0.4,
                'host_io_usage': 0.3,
                'host_ready_time': 0.05,
                'vm_count_on_host': 5,
                'host_utilization_score': 0.7,
                'placement_score': 0.8
            }
        ]
        
        result = self.model.train(data)
        
        assert result is True
        assert self.model.is_trained is True
        mock_model.fit.assert_called_once()
    
    def test_train_failure(self):
        """Test model training failure."""
        data = []  # Empty data should cause failure
        
        with pytest.raises(ModelTrainingError):
            self.model.train(data)
    
    def test_predict_not_trained(self):
        """Test prediction without trained model."""
        features = [{'cpu_usage': 0.5}]
        
        with pytest.raises(PredictionError, match="Model not trained"):
            self.model.predict(features)
    
    @patch('ai_optimizer.ml_engine.joblib')
    def test_save_load_model(self, mock_joblib):
        """Test model save and load functionality."""
        # Mock the model
        self.model.model = Mock()
        self.model.is_trained = True
        
        # Test save
        result = self.model.save_model('/tmp/test_model.pkl')
        assert result is True
        mock_joblib.dump.assert_called_once()
        
        # Test load
        mock_joblib.load.return_value = {
            'model': Mock(),
            'scaler': Mock(),
            'is_trained': True
        }
        
        result = self.model.load_model('/tmp/test_model.pkl')
        assert result is True
        mock_joblib.load.assert_called_once()


class TestNeuralNetworkModel:
    """Test the neural network model functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AIConfig()
        self.model = NeuralNetworkModel(self.config)
    
    def test_init(self):
        """Test neural network initialization."""
        assert self.model.config == self.config
        assert self.model.device is not None
        assert self.model.network is None
        assert self.model.optimizer is None
    
    def test_build_network(self):
        """Test network architecture building."""
        network = self.model._build_network(10)
        assert network is not None
        assert len(list(network.parameters())) > 0
    
    @patch('torch.FloatTensor')
    @patch('torch.optim.Adam')
    def test_train_success(self, mock_optimizer, mock_tensor):
        """Test successful neural network training."""
        mock_optimizer.return_value = Mock()
        mock_tensor.return_value = Mock()
        
        data = [
            {
                'cpu_usage': 0.5,
                'ram_usage': 0.3,
                'ready_time': 0.1,
                'io_usage': 0.2,
                'host_cpu_usage': 0.6,
                'host_ram_usage': 0.4,
                'host_io_usage': 0.3,
                'host_ready_time': 0.05,
                'vm_count_on_host': 5,
                'host_utilization_score': 0.7,
                'placement_score': 0.8
            }
        ]
        
        # Mock the training process
        with patch.object(self.model, '_build_network') as mock_build:
            mock_network = Mock()
            mock_build.return_value = mock_network
            
            result = self.model.train(data)
            
            assert result is True
            assert self.model.is_trained is True
            mock_build.assert_called_once()


class TestMLEngine:
    """Test the main ML engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AIConfig()
        self.data_collector = Mock(spec=PrometheusDataCollector)
        self.ml_engine = MLEngine(self.config, self.data_collector)
    
    def test_init(self):
        """Test ML engine initialization."""
        assert self.ml_engine.config == self.config
        assert self.ml_engine.data_collector == self.data_collector
        assert self.ml_engine.baseline_model is not None
        assert self.ml_engine.neural_network is not None
    
    def test_calculate_placement_score(self):
        """Test placement score calculation."""
        vm_metrics = {
            'cpu': 0.3,
            'ram': 0.2,
            'ready_time': 0.05,
            'io': 0.1
        }
        
        host_metrics = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3,
            'io_usage': 0.2,
            'ready_time': 0.02,
            'vm_count': 5,
            'utilization_score': 0.6
        }
        
        score = self.ml_engine._calculate_placement_score(vm_metrics, host_metrics)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_generate_training_data(self):
        """Test training data generation."""
        vms = [
            {'name': 'test-vm-1'},
            {'name': 'test-vm-2'}
        ]
        
        hosts = [
            {'name': 'host-1'},
            {'name': 'host-2'}
        ]
        
        # Mock data collector methods
        self.data_collector.get_vm_cpu_trend.return_value = [(1, 0.5)]
        self.data_collector.get_vm_ram_trend.return_value = [(1, 0.3)]
        self.data_collector.get_vm_ready_time_trend.return_value = [(1, 0.1)]
        self.data_collector.get_vm_io_trend.return_value = [(1, 0.2)]
        self.data_collector.get_host_performance_metrics.return_value = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3,
            'io_usage': 0.2,
            'ready_time': 0.02,
            'vm_count': 5,
            'utilization_score': 0.6
        }
        
        training_data = self.ml_engine.generate_training_data(vms, hosts)
        
        assert len(training_data) > 0
        assert all(isinstance(item, dict) for item in training_data)
        assert all('placement_score' in item for item in training_data)
    
    def test_predict_placement_scores(self):
        """Test placement score prediction."""
        vm_metrics = {
            'cpu_usage': 0.3,
            'ram_usage': 0.2,
            'ready_time': 0.05,
            'io_usage': 0.1
        }
        
        host_candidates = [
            {'name': 'host-1'},
            {'name': 'host-2'}
        ]
        
        # Mock host metrics
        self.data_collector.get_host_performance_metrics.return_value = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3,
            'io_usage': 0.2,
            'ready_time': 0.02,
            'vm_count': 5,
            'utilization_score': 0.6
        }
        
        # Mock model predictions
        self.ml_engine.baseline_model.is_trained = True
        self.ml_engine.baseline_model.predict = Mock(return_value=[0.7, 0.6])
        
        scores = self.ml_engine.predict_placement_scores(vm_metrics, host_candidates)
        
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        assert all(0.0 <= score <= 1.0 for score in scores)
    
    def test_predict_placement_scores_fallback(self):
        """Test placement score prediction with fallback."""
        vm_metrics = {'cpu_usage': 0.3}
        host_candidates = [{'name': 'host-1'}]
        
        # Mock host metrics
        self.data_collector.get_host_performance_metrics.return_value = {
            'cpu_usage': 0.4,
            'ram_usage': 0.3,
            'io_usage': 0.2,
            'ready_time': 0.02,
            'vm_count': 5,
            'utilization_score': 0.6
        }
        
        # No trained models
        self.ml_engine.baseline_model.is_trained = False
        self.ml_engine.neural_network.is_trained = False
        
        scores = self.ml_engine.predict_placement_scores(vm_metrics, host_candidates)
        
        assert len(scores) == 1
        assert scores[0] == 0.5  # Fallback score


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 
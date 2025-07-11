"""
Machine Learning Engine for AI Optimizer

This module provides the core ML functionality for VM placement optimization,
including model training, prediction, and evaluation using reinforcement learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .config import AIConfig
from .data_collector import PrometheusDataCollector
from .exceptions import ModelTrainingError, PredictionError


class MLModel:
    """Base class for ML models used in VM placement optimization."""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training.
        
        Args:
            data: List of dictionaries containing VM and host metrics
            
        Returns:
            Tuple of (features, targets)
        """
        features = []
        targets = []
        
        for item in data:
            # Extract features
            feature_vector = [
                item.get('cpu_usage', 0.0),
                item.get('ram_usage', 0.0),
                item.get('ready_time', 0.0),
                item.get('io_usage', 0.0),
                item.get('host_cpu_usage', 0.0),
                item.get('host_ram_usage', 0.0),
                item.get('host_io_usage', 0.0),
                item.get('host_ready_time', 0.0),
                item.get('vm_count_on_host', 0),
                item.get('host_utilization_score', 0.0)
            ]
            
            # Target: placement score (0-1, higher is better)
            target = item.get('placement_score', 0.5)
            
            features.append(feature_vector)
            targets.append(target)
        
        features_array = np.array(features)
        targets_array = np.array(targets)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_array)
        
        return features_scaled, targets_array
    
    def train(self, data: List[Dict[str, Any]]) -> bool:
        """
        Train the model on the provided data.
        
        Args:
            data: List of dictionaries containing features and targets
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            features, targets = self.preprocess_data(data)
            
            # Use Random Forest for baseline
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            if self.model is not None:
                self.model.fit(features, targets)
                self.is_trained = True
                
                self.logger.info(f"Model trained successfully on {len(data)} samples")
                return True
            else:
                self.logger.error("Failed to initialize model")
                return False
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise ModelTrainingError("MLModel", f"Failed to train model: {e}")
    
    def predict(self, features: List[Dict[str, Any]]) -> List[float]:
        """
        Make predictions on new data.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            List of predicted scores
        """
        if not self.is_trained or self.model is None:
            raise PredictionError("MLModel", "Model not trained")
        
        try:
            # Convert to feature vectors
            feature_vectors = []
            for item in features:
                feature_vector = [
                    item.get('cpu_usage', 0.0),
                    item.get('ram_usage', 0.0),
                    item.get('ready_time', 0.0),
                    item.get('io_usage', 0.0),
                    item.get('host_cpu_usage', 0.0),
                    item.get('host_ram_usage', 0.0),
                    item.get('host_io_usage', 0.0),
                    item.get('host_ready_time', 0.0),
                    item.get('vm_count_on_host', 0),
                    item.get('host_utilization_score', 0.0)
                ]
                feature_vectors.append(feature_vector)
            
            features_array = np.array(feature_vectors)
            features_scaled = self.scaler.transform(features_array)
            
            predictions = self.model.predict(features_scaled)
            return [float(p) for p in predictions.tolist()]
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise PredictionError("MLModel", f"Failed to make predictions: {e}")
    
    def save_model(self, path: str) -> bool:
        """Save the trained model to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, path)
            self.logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a trained model from disk."""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False


class NeuralNetworkModel(MLModel):
    """Neural network model for more complex VM placement optimization."""
    
    def __init__(self, config: AIConfig):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
    
    def _build_network(self, input_size: int) -> nn.Module:
        """Build the neural network architecture."""
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def train(self, data: List[Dict[str, Any]]) -> bool:
        """Train the neural network model."""
        try:
            features, targets = self.preprocess_data(data)
            
            # Convert to PyTorch tensors
            X = torch.FloatTensor(features).to(self.device)
            y = torch.FloatTensor(targets).to(self.device)
            
            # Build network
            input_size = features.shape[1]
            self.network = self._build_network(input_size).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.ml.learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            self.network.train()
            for epoch in range(self.config.ml.training_episodes):
                self.optimizer.zero_grad()
                outputs = self.network(X).squeeze()
                loss = criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                if epoch % 100 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            self.is_trained = True
            self.logger.info(f"Neural network trained successfully on {len(data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Neural network training failed: {e}")
            raise ModelTrainingError("NeuralNetworkModel", f"Failed to train neural network: {e}")
    
    def predict(self, features: List[Dict[str, Any]]) -> List[float]:
        """Make predictions using the neural network."""
        if not self.is_trained or self.network is None:
            raise PredictionError("NeuralNetworkModel", "Model not trained")
        
        try:
            # Convert to feature vectors
            feature_vectors = []
            for item in features:
                feature_vector = [
                    item.get('cpu_usage', 0.0),
                    item.get('ram_usage', 0.0),
                    item.get('ready_time', 0.0),
                    item.get('io_usage', 0.0),
                    item.get('host_cpu_usage', 0.0),
                    item.get('host_ram_usage', 0.0),
                    item.get('host_io_usage', 0.0),
                    item.get('host_ready_time', 0.0),
                    item.get('vm_count_on_host', 0),
                    item.get('host_utilization_score', 0.0)
                ]
                feature_vectors.append(feature_vector)
            
            features_array = np.array(feature_vectors)
            features_scaled = self.scaler.transform(features_array)
            
            # Convert to tensor
            X = torch.FloatTensor(features_scaled).to(self.device)
            
            # Make predictions
            self.network.eval()
            with torch.no_grad():
                predictions = self.network(X).squeeze().cpu().numpy()
            
            return [float(p) for p in predictions.tolist()]
            
        except Exception as e:
            self.logger.error(f"Neural network prediction failed: {e}")
            raise PredictionError("NeuralNetworkModel", f"Failed to make predictions: {e}")


class MLEngine:
    """
    Main ML engine that coordinates model training and prediction.
    
    This class manages multiple ML models and provides a unified interface
    for VM placement optimization.
    """
    
    def __init__(self, config: AIConfig, data_collector: PrometheusDataCollector):
        self.config = config
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.baseline_model = MLModel(config)
        self.neural_network = NeuralNetworkModel(config)
        
        # Model paths
        self.baseline_model_path = os.path.join(config.ml.model_save_path, 'baseline_model.pkl')
        self.nn_model_path = os.path.join(config.ml.model_save_path, 'neural_network.pkl')
    
    def generate_training_data(self, vms: List[Dict[str, Any]], hosts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate training data from VM and host metrics.
        
        Args:
            vms: List of VM dictionaries
            hosts: List of host dictionaries
            
        Returns:
            List of training data dictionaries
        """
        training_data = []
        
        for vm in vms:
            vm_name = vm.get('name', '')
            if not vm_name:
                continue
            
            # Get VM metrics
            cpu_trend = self.data_collector.get_vm_cpu_trend(vm_name, self.config.analysis.cpu_trend_hours)
            ram_trend = self.data_collector.get_vm_ram_trend(vm_name, self.config.analysis.ram_trend_hours)
            ready_trend = self.data_collector.get_vm_ready_time_trend(vm_name, self.config.analysis.ready_time_window)
            io_trend = self.data_collector.get_vm_io_trend(vm_name, self.config.analysis.io_trend_days)
            
            # Calculate average metrics
            cpu_usage = np.mean([v for _, v in cpu_trend]) if cpu_trend else 0.0
            ram_usage = np.mean([v for _, v in ram_trend]) if ram_trend else 0.0
            ready_time = np.mean([v for _, v in ready_trend]) if ready_trend else 0.0
            io_usage = np.mean([v for _, v in io_trend]) if io_trend else 0.0
            
            # Generate training samples for each host
            for host in hosts:
                host_name = host.get('name', '')
                if not host_name:
                    continue
                
                # Get host metrics
                host_metrics = self.data_collector.get_host_performance_metrics(host_name)
                
                # Calculate placement score based on optimization criteria
                placement_score = self._calculate_placement_score(
                    vm_metrics={'cpu': cpu_usage, 'ram': ram_usage, 'ready_time': ready_time, 'io': io_usage},
                    host_metrics=host_metrics
                )
                
                training_sample = {
                    'cpu_usage': cpu_usage,
                    'ram_usage': ram_usage,
                    'ready_time': ready_time,
                    'io_usage': io_usage,
                    'host_cpu_usage': host_metrics.get('cpu_usage', 0.0),
                    'host_ram_usage': host_metrics.get('ram_usage', 0.0),
                    'host_io_usage': host_metrics.get('io_usage', 0.0),
                    'host_ready_time': host_metrics.get('ready_time', 0.0),
                    'vm_count_on_host': host_metrics.get('vm_count', 0),
                    'host_utilization_score': host_metrics.get('utilization_score', 0.0),
                    'placement_score': placement_score
                }
                
                training_data.append(training_sample)
        
        return training_data
    
    def _calculate_placement_score(self, vm_metrics: Dict[str, float], host_metrics: Dict[str, float]) -> float:
        """
        Calculate a placement score based on optimization criteria.
        
        Args:
            vm_metrics: VM performance metrics
            host_metrics: Host performance metrics
            
        Returns:
            float: Placement score (0-1, higher is better)
        """
        # Ideal host usage range
        ideal_min = self.config.optimization.ideal_host_usage_min
        ideal_max = self.config.optimization.ideal_host_usage_max
        
        # Calculate host utilization after placement
        current_cpu = host_metrics.get('cpu_usage', 0.0)
        current_ram = host_metrics.get('ram_usage', 0.0)
        
        projected_cpu = min(1.0, current_cpu + vm_metrics['cpu'])
        projected_ram = min(1.0, current_ram + vm_metrics['ram'])
        
        # Score based on ideal utilization range
        cpu_score = 1.0 if ideal_min <= projected_cpu <= ideal_max else 0.5
        ram_score = 1.0 if ideal_min <= projected_ram <= ideal_max else 0.5
        
        # Penalty for high ready time
        ready_penalty = max(0, vm_metrics['ready_time'] - 0.1) * 2  # Penalty for >10% ready time
        
        # Weighted score
        weights = {
            'cpu': self.config.optimization.cpu_priority_weight,
            'ram': self.config.optimization.ram_priority_weight,
            'ready_time': self.config.optimization.ready_time_priority_weight,
            'io': self.config.optimization.io_priority_weight
        }
        
        total_score = (
            cpu_score * weights['cpu'] +
            ram_score * weights['ram'] +
            (1.0 - ready_penalty) * weights['ready_time'] +
            (1.0 - vm_metrics['io']) * weights['io']
        ) / sum(weights.values())
        
        return max(0.0, min(1.0, total_score))
    
    def train_models(self, vms: List[Dict[str, Any]], hosts: List[Dict[str, Any]]) -> bool:
        """
        Train both baseline and neural network models.
        
        Args:
            vms: List of VM dictionaries
            hosts: List of host dictionaries
            
        Returns:
            bool: True if training successful
        """
        try:
            # Generate training data
            training_data = self.generate_training_data(vms, hosts)
            
            if len(training_data) < 10:
                self.logger.warning("Insufficient training data")
                return False
            
            # Train baseline model
            self.logger.info("Training baseline model...")
            baseline_success = self.baseline_model.train(training_data)
            
            # Train neural network
            self.logger.info("Training neural network...")
            nn_success = self.neural_network.train(training_data)
            
            # Save models
            if baseline_success:
                self.baseline_model.save_model(self.baseline_model_path)
            
            if nn_success:
                self.neural_network.save_model(self.nn_model_path)
            
            return baseline_success and nn_success
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False
    
    def load_trained_models(self) -> bool:
        """Load previously trained models."""
        baseline_loaded = self.baseline_model.load_model(self.baseline_model_path)
        nn_loaded = self.neural_network.load_model(self.nn_model_path)
        
        return baseline_loaded or nn_loaded  # At least one model should be available
    
    def predict_placement_scores(self, vm_metrics: Dict[str, Any], host_candidates: List[Dict[str, Any]]) -> List[float]:
        """
        Predict placement scores for a VM on different host candidates.
        
        Args:
            vm_metrics: VM performance metrics
            host_candidates: List of candidate hosts
            
        Returns:
            List of predicted placement scores
        """
        try:
            # Prepare feature data for each host candidate
            features = []
            for host in host_candidates:
                host_metrics = self.data_collector.get_host_performance_metrics(host.get('name', ''))
                
                feature_dict = {
                    'cpu_usage': vm_metrics.get('cpu_usage', 0.0),
                    'ram_usage': vm_metrics.get('ram_usage', 0.0),
                    'ready_time': vm_metrics.get('ready_time', 0.0),
                    'io_usage': vm_metrics.get('io_usage', 0.0),
                    'host_cpu_usage': host_metrics.get('cpu_usage', 0.0),
                    'host_ram_usage': host_metrics.get('ram_usage', 0.0),
                    'host_io_usage': host_metrics.get('io_usage', 0.0),
                    'host_ready_time': host_metrics.get('ready_time', 0.0),
                    'vm_count_on_host': host_metrics.get('vm_count', 0),
                    'host_utilization_score': host_metrics.get('utilization_score', 0.0)
                }
                features.append(feature_dict)
            
            # Make predictions using available models
            predictions = []
            
            if self.baseline_model.is_trained:
                baseline_preds = self.baseline_model.predict(features)
                predictions.append(baseline_preds)
            
            if self.neural_network.is_trained:
                nn_preds = self.neural_network.predict(features)
                predictions.append(nn_preds)
            
            # Average predictions if multiple models available
            if predictions:
                avg_predictions = np.mean(predictions, axis=0)
                return avg_predictions.tolist()
            else:
                # Fallback to simple scoring
                return [0.5] * len(host_candidates)
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return [0.5] * len(host_candidates)  # Fallback scores 
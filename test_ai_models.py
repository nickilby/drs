#!/usr/bin/env python3
"""
AI Model Testing Script for vCenter DRS

This script tests the trained machine learning models with real VM and host data
to demonstrate how the AI models can predict optimal VM placements.
"""

import sys
import os
import joblib
import json
import numpy as np
from typing import Dict, List, Any
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_optimizer.config import AIConfig
from ai_optimizer.data_collector import PrometheusDataCollector
from ai_optimizer.optimization_engine import OptimizationEngine


class ModelTester:
    """Test trained AI models with real data"""
    
    def __init__(self):
        self.config = AIConfig()
        self.data_collector = PrometheusDataCollector(self.config)
        self.optimization_engine = OptimizationEngine(self.config, self.data_collector)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models_dir = "ai_optimizer/models"
        
        # Load trained models
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scalers"""
        try:
            # Load feature names
            feature_names_path = os.path.join(self.models_dir, "feature_names.json")
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            
            # Load models
            model_types = ['random_forest', 'gradient_boosting']
            
            for model_type in model_types:
                model_path = os.path.join(self.models_dir, f"{model_type}_model.pkl")
                scaler_path = os.path.join(self.models_dir, f"{model_type}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_type] = joblib.load(model_path)
                    self.scalers[model_type] = joblib.load(scaler_path)
                    self.logger.info(f"Loaded {model_type} model")
                else:
                    self.logger.warning(f"Model files not found for {model_type}")
            
            # Load training results
            results_path = os.path.join(self.models_dir, "training_results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.training_results = json.load(f)
                    self.logger.info("Loaded training results")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def prepare_features(self, vm_metrics: Dict[str, float], host_metrics: Dict[str, float]) -> np.ndarray:
        """Prepare features for model prediction"""
        # Calculate projected metrics
        projected_cpu_usage = min(1.0, host_metrics.get('cpu_usage', 0.0) + vm_metrics.get('cpu_usage', 0.0))
        projected_ram_usage = min(1.0, host_metrics.get('ram_usage', 0.0) + vm_metrics.get('ram_usage', 0.0))
        projected_io_usage = min(1.0, host_metrics.get('io_usage', 0.0) + vm_metrics.get('io_usage', 0.0))
        projected_vm_count = host_metrics.get('vm_count', 0) + 1
        
        # Create feature vector
        feature_vector = [
            vm_metrics.get('cpu_usage', 0.0),
            vm_metrics.get('ram_usage', 0.0),
            vm_metrics.get('ready_time', 0.0),
            vm_metrics.get('io_usage', 0.0),
            vm_metrics.get('cpu_mhz', 0.0),
            vm_metrics.get('ram_mb', 0.0),
            host_metrics.get('cpu_usage', 0.0),
            host_metrics.get('ram_usage', 0.0),
            host_metrics.get('io_usage', 0.0),
            host_metrics.get('ready_time', 0.0),
            host_metrics.get('vm_count', 0),
            host_metrics.get('cpu_max_mhz', 40000),
            host_metrics.get('ram_max_mb', 65536),
            projected_cpu_usage,
            projected_ram_usage,
            projected_io_usage,
            projected_vm_count
        ]
        
        return np.array([feature_vector])
    
    def predict_placement_score(self, vm_metrics: Dict[str, float], host_metrics: Dict[str, float]) -> Dict[str, float]:
        """Predict placement score using trained models"""
        try:
            # Prepare features
            features = self.prepare_features(vm_metrics, host_metrics)
            
            predictions = {}
            
            # Make predictions with each model
            for model_name, model in self.models.items():
                if model_name in self.scalers:
                    scaler = self.scalers[model_name]
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)[0]
                    predictions[model_name] = max(0.0, min(1.0, prediction))
            
            # Calculate ensemble prediction
            if len(predictions) > 1:
                ensemble_score = sum(predictions.values()) / len(predictions)
                predictions['ensemble'] = ensemble_score
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {}
    
    def test_with_real_data(self):
        """Test models with real VM and host data"""
        print("🧪 Testing AI Models with Real Data")
        print("="*50)
        
        # Test VMs
        test_vms = ['z-actegy-WEB1', 'z-actegy-APP1', 'z-actegy-DB1']
        test_hosts = ['lon-esxi8.zengenti.io', 'lon-esxi9.zengenti.io', 'lon-esxi10.zengenti.io']
        
        results = []
        
        for vm_name in test_vms:
            print(f"\n📊 Testing VM: {vm_name}")
            print("-" * 30)
            
            try:
                # Get VM metrics
                vm_metrics = self.optimization_engine.get_vm_metrics(vm_name)
                
                if not vm_metrics:
                    print(f"❌ Could not get metrics for VM {vm_name}")
                    continue
                
                print(f"VM Metrics:")
                print(f"  CPU Usage: {vm_metrics.get('cpu_usage', 0):.1%}")
                print(f"  RAM Usage: {vm_metrics.get('ram_usage', 0):.1%}")
                print(f"  Ready Time: {vm_metrics.get('ready_time', 0):.1%}")
                print(f"  I/O Usage: {vm_metrics.get('io_usage', 0):.1%}")
                
                # Test each host
                host_predictions = []
                
                for host_name in test_hosts:
                    try:
                        # Get host metrics
                        host_metrics = self.data_collector.get_host_performance_metrics(host_name)
                        
                        if not host_metrics:
                            print(f"❌ Could not get metrics for host {host_name}")
                            continue
                        
                        # Get AI predictions
                        predictions = self.predict_placement_score(vm_metrics, host_metrics)
                        
                        if predictions:
                            host_predictions.append({
                                'host_name': host_name,
                                'predictions': predictions,
                                'host_metrics': host_metrics
                            })
                            
                            print(f"\n🏠 Host: {host_name}")
                            print(f"  Current CPU: {host_metrics.get('cpu_usage', 0):.1%}")
                            print(f"  Current RAM: {host_metrics.get('ram_usage', 0):.1%}")
                            print(f"  VM Count: {host_metrics.get('vm_count', 0)}")
                            
                            for model_name, score in predictions.items():
                                print(f"  {model_name.upper()} Score: {score:.3f}")
                        
                    except Exception as e:
                        print(f"❌ Error testing host {host_name}: {e}")
                
                # Sort by best prediction
                if host_predictions:
                    best_model = 'ensemble' if 'ensemble' in host_predictions[0]['predictions'] else list(host_predictions[0]['predictions'].keys())[0]
                    host_predictions.sort(key=lambda x: x['predictions'].get(best_model, 0), reverse=True)
                    
                    print(f"\n🏆 Best Placement (using {best_model.upper()}):")
                    print(f"  Host: {host_predictions[0]['host_name']}")
                    print(f"  Score: {host_predictions[0]['predictions'].get(best_model, 0):.3f}")
                
                results.append({
                    'vm_name': vm_name,
                    'vm_metrics': vm_metrics,
                    'host_predictions': host_predictions
                })
                
            except Exception as e:
                print(f"❌ Error testing VM {vm_name}: {e}")
        
        return results
    
    def print_model_performance(self):
        """Print model performance from training"""
        if not self.training_results:
            print("❌ No training results available")
            return
        
        print("\n📈 Model Performance Summary")
        print("="*40)
        
        for model_name, metrics in self.training_results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  R² Score: {metrics['r2']:.4f}")
            print(f"  Mean Squared Error: {metrics['mse']:.4f}")
            print(f"  Mean Absolute Error: {metrics['mae']:.4f}")
        
        # Find best model
        best_model = max(self.training_results.items(), key=lambda x: x[1]['r2'])
        print(f"\n🏆 Best Model: {best_model[0].upper()}")
        print(f"R² Score: {best_model[1]['r2']:.4f}")


def main():
    """Main testing function"""
    print("🤖 AI Model Testing for vCenter DRS")
    print("="*50)
    
    tester = ModelTester()
    
    # Print model performance
    tester.print_model_performance()
    
    # Test with real data
    results = tester.test_with_real_data()
    
    if results:
        print(f"\n✅ Testing completed successfully!")
        print(f"Tested {len(results)} VMs with AI models")
    else:
        print(f"\n❌ Testing failed. Check the logs for details.")


if __name__ == "__main__":
    main() 
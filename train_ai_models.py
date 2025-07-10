#!/usr/bin/env python3
"""
AI Model Training Script for vCenter DRS

This script trains machine learning models for VM placement optimization using synthetic data
and real metrics from your vCenter environment. It trains multiple models and evaluates their performance.
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_optimizer.config import AIConfig
from ai_optimizer.data_collector import PrometheusDataCollector
from ai_optimizer.ml_engine import MLEngine
from ai_optimizer.optimization_engine import OptimizationEngine


class ModelTrainer:
    """Comprehensive model trainer for VM placement optimization"""
    
    def __init__(self):
        self.config = AIConfig()
        self.data_collector = PrometheusDataCollector(self.config)
        self.ml_engine = MLEngine(self.config, self.data_collector)
        self.optimization_engine = OptimizationEngine(self.config, self.data_collector)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models_dir = "ai_optimizer/models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Training results
        self.training_results = {}
    
    def generate_synthetic_training_data(self) -> List[Dict]:
        """Generate synthetic training data for demonstration"""
        self.logger.info("Generating synthetic training data...")
        
        training_samples = []
        
        # Generate realistic training scenarios
        for _ in range(1000):  # 1000 training samples
            
            # VM characteristics
            vm_cpu_usage = random.uniform(0.05, 0.8)  # 5-80% CPU usage
            vm_ram_usage = random.uniform(0.1, 0.9)   # 10-90% RAM usage
            vm_ready_time = random.uniform(0.01, 0.3)  # 1-30% ready time
            vm_io_usage = random.uniform(0.01, 0.5)    # 1-50% I/O usage
            vm_cpu_mhz = vm_cpu_usage * random.uniform(2000, 8000)  # CPU in MHz
            vm_ram_mb = vm_ram_usage * random.uniform(2048, 16384)  # RAM in MB
            
            # Host characteristics
            host_cpu_usage = random.uniform(0.1, 0.9)  # 10-90% CPU usage
            host_ram_usage = random.uniform(0.2, 0.95) # 20-95% RAM usage
            host_io_usage = random.uniform(0.01, 0.4)  # 1-40% I/O usage
            host_ready_time = random.uniform(0.01, 0.2) # 1-20% ready time
            host_vm_count = random.randint(1, 25)      # 1-25 VMs
            host_cpu_max_mhz = random.choice([40000, 60000, 80000, 100000])  # Max CPU MHz
            host_ram_max_mb = random.choice([32768, 65536, 131072, 262144])  # Max RAM MB
            
            # Calculate projected metrics
            projected_cpu_usage = min(1.0, host_cpu_usage + vm_cpu_usage)
            projected_ram_usage = min(1.0, host_ram_usage + vm_ram_usage)
            projected_io_usage = min(1.0, host_io_usage + vm_io_usage)
            projected_vm_count = host_vm_count + 1
            
            # Calculate placement score based on optimization criteria
            score = self._calculate_synthetic_score(
                vm_cpu_usage, vm_ram_usage, vm_ready_time, vm_io_usage,
                host_cpu_usage, host_ram_usage, host_io_usage, host_ready_time,
                host_vm_count, projected_cpu_usage, projected_ram_usage, projected_io_usage
            )
            
            sample = {
                # VM features
                'vm_cpu_usage': vm_cpu_usage,
                'vm_ram_usage': vm_ram_usage,
                'vm_ready_time': vm_ready_time,
                'vm_io_usage': vm_io_usage,
                'vm_cpu_mhz': vm_cpu_mhz,
                'vm_ram_mb': vm_ram_mb,
                
                # Host features
                'host_cpu_usage': host_cpu_usage,
                'host_ram_usage': host_ram_usage,
                'host_io_usage': host_io_usage,
                'host_ready_time': host_ready_time,
                'host_vm_count': host_vm_count,
                'host_cpu_max_mhz': host_cpu_max_mhz,
                'host_ram_max_mb': host_ram_max_mb,
                
                # Projected features
                'projected_cpu_usage': projected_cpu_usage,
                'projected_ram_usage': projected_ram_usage,
                'projected_io_usage': projected_io_usage,
                'projected_vm_count': projected_vm_count,
                
                # Target
                'placement_score': score
            }
            
            training_samples.append(sample)
        
        self.logger.info(f"Generated {len(training_samples)} synthetic training samples")
        return training_samples
    
    def _calculate_synthetic_score(self, vm_cpu, vm_ram, vm_ready, vm_io,
                                 host_cpu, host_ram, host_io, host_ready,
                                 host_vm_count, proj_cpu, proj_ram, proj_io) -> float:
        """Calculate synthetic placement score based on optimization criteria"""
        score = 0.0
        
        # Ideal host usage score (30-70% is ideal)
        if 0.3 <= proj_cpu <= 0.7:
            score += 0.25  # Ideal range
        elif proj_cpu <= 0.5:  # Low usage
            score += 0.15
        elif proj_cpu <= 0.8:  # Acceptable usage
            score += 0.1
        elif proj_cpu <= 0.9:  # High usage
            score += 0.05
        else:
            score -= 0.2  # Overloaded
        
        # RAM usage score
        if 0.3 <= proj_ram <= 0.7:
            score += 0.25  # Ideal range
        elif proj_ram <= 0.5:  # Low usage
            score += 0.15
        elif proj_ram <= 0.8:  # Acceptable usage
            score += 0.1
        elif proj_ram <= 0.9:  # High usage
            score += 0.05
        else:
            score -= 0.2  # Overloaded
        
        # VM count score
        if host_vm_count <= 3:
            score += 0.15  # Very low - excellent
        elif host_vm_count <= 5:
            score += 0.1   # Low - good
        elif host_vm_count <= 10:
            score += 0.05  # Moderate - acceptable
        elif host_vm_count <= 20:
            score += 0.0   # High - neutral
        elif host_vm_count <= 30:
            score -= 0.05  # Very high - concerning
        else:
            score -= 0.1   # Extremely high - poor
        
        # VM ready time improvement score
        if vm_ready <= 0.1:  # Excellent ready time
            score += 0.15
        elif vm_ready <= 0.3:  # Good ready time
            score += 0.1
        elif vm_ready <= 0.5:  # Acceptable ready time
            score += 0.05
        elif vm_ready <= 0.8:  # Poor ready time
            score += 0.0
        else:
            score -= 0.1  # Very poor ready time
        
        # Resource efficiency score
        if proj_cpu <= 0.6 and proj_ram <= 0.6:
            score += 0.15  # Excellent resource availability
        elif proj_cpu <= 0.8 and proj_ram <= 0.8:
            score += 0.1   # Good resource availability
        elif proj_cpu <= 0.9 and proj_ram <= 0.9:
            score += 0.05  # Acceptable resource availability
        else:
            score -= 0.1   # Poor resource availability
        
        # Current host load consideration
        if host_cpu <= 0.3 and host_ram <= 0.3:
            score += 0.1   # Very low current load
        elif host_cpu <= 0.5 and host_ram <= 0.5:
            score += 0.05  # Low current load
        elif host_cpu >= 0.8 or host_ram >= 0.8:
            score -= 0.05  # High current load
        
        # Normalize score to 0-1 range
        score = max(0.0, min(1.0, score))
        
        # Add small randomization for differentiation
        score += random.uniform(-0.01, 0.01)
        score = max(0.0, min(1.0, score))
        
        return score
    
    def prepare_features(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        features = []
        targets = []
        
        for sample in samples:
            # Feature vector (excluding target)
            feature_vector = [
                sample['vm_cpu_usage'],
                sample['vm_ram_usage'],
                sample['vm_ready_time'],
                sample['vm_io_usage'],
                sample['vm_cpu_mhz'],
                sample['vm_ram_mb'],
                sample['host_cpu_usage'],
                sample['host_ram_usage'],
                sample['host_io_usage'],
                sample['host_ready_time'],
                sample['host_vm_count'],
                sample['host_cpu_max_mhz'],
                sample['host_ram_max_mb'],
                sample['projected_cpu_usage'],
                sample['projected_ram_usage'],
                sample['projected_io_usage'],
                sample['projected_vm_count']
            ]
            
            features.append(feature_vector)
            targets.append(sample['placement_score'])
        
        return np.array(features), np.array(targets)
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train multiple ML models and evaluate performance"""
        self.logger.info("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        results = {}
        
        # 1. Random Forest
        self.logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        models['random_forest'] = {
            'model': rf_model,
            'scaler': scaler,
            'feature_importance': rf_model.feature_importances_
        }
        
        results['random_forest'] = {
            'mse': mean_squared_error(y_test, rf_pred),
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred)
        }
        
        # 2. Gradient Boosting
        self.logger.info("Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        models['gradient_boosting'] = {
            'model': gb_model,
            'scaler': scaler,
            'feature_importance': gb_model.feature_importances_
        }
        
        results['gradient_boosting'] = {
            'mse': mean_squared_error(y_test, gb_pred),
            'mae': mean_absolute_error(y_test, gb_pred),
            'r2': r2_score(y_test, gb_pred)
        }
        
        # 3. Ensemble (Average of both models)
        ensemble_pred = (rf_pred + gb_pred) / 2
        results['ensemble'] = {
            'mse': mean_squared_error(y_test, ensemble_pred),
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'r2': r2_score(y_test, ensemble_pred)
        }
        
        self.logger.info("Model training completed!")
        return models, results
    
    def save_models(self, models: Dict[str, Any], results: Dict[str, Any]):
        """Save trained models and results"""
        self.logger.info("Saving models and results...")
        
        # Save models
        for name, model_data in models.items():
            model_path = os.path.join(self.models_dir, f"{name}_model.pkl")
            scaler_path = os.path.join(self.models_dir, f"{name}_scaler.pkl")
            
            joblib.dump(model_data['model'], model_path)
            joblib.dump(model_data['scaler'], scaler_path)
            
            self.logger.info(f"Saved {name} model to {model_path}")
        
        # Save results
        results_path = os.path.join(self.models_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save feature names for reference
        feature_names = [
            'vm_cpu_usage', 'vm_ram_usage', 'vm_ready_time', 'vm_io_usage',
            'vm_cpu_mhz', 'vm_ram_mb', 'host_cpu_usage', 'host_ram_usage',
            'host_io_usage', 'host_ready_time', 'host_vm_count', 'host_cpu_max_mhz',
            'host_ram_max_mb', 'projected_cpu_usage', 'projected_ram_usage',
            'projected_io_usage', 'projected_vm_count'
        ]
        
        features_path = os.path.join(self.models_dir, "feature_names.json")
        with open(features_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        self.logger.info(f"Saved training results to {results_path}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print training results"""
        print("\n" + "="*60)
        print("AI MODEL TRAINING RESULTS")
        print("="*60)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Mean Squared Error: {metrics['mse']:.4f}")
            print(f"  Mean Absolute Error: {metrics['mae']:.4f}")
            print(f"  R² Score: {metrics['r2']:.4f}")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"Best performing model: {best_model[0].upper()}")
        print(f"R² Score: {best_model[1]['r2']:.4f}")
        
        if best_model[1]['r2'] > 0.7:
            print("✅ Excellent model performance!")
        elif best_model[1]['r2'] > 0.5:
            print("✅ Good model performance")
        elif best_model[1]['r2'] > 0.3:
            print("⚠️  Moderate model performance - consider more training data")
        else:
            print("❌ Poor model performance - need more training data")
    
    def run_training(self):
        """Run the complete training pipeline"""
        try:
            self.logger.info("Starting AI model training pipeline...")
            
            # Step 1: Generate synthetic training data
            samples = self.generate_synthetic_training_data()
            
            if len(samples) < 10:
                self.logger.error("Insufficient training samples")
                return False
            
            # Step 2: Prepare features
            X, y = self.prepare_features(samples)
            
            # Step 3: Train models
            models, results = self.train_models(X, y)
            
            # Step 4: Save models
            self.save_models(models, results)
            
            # Step 5: Print results
            self.print_results(results)
            
            self.logger.info("Training pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return False


def main():
    """Main training function"""
    print("🤖 AI Model Training for vCenter DRS")
    print("="*50)
    
    trainer = ModelTrainer()
    success = trainer.run_training()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("Models saved to: ai_optimizer/models/")
        print("You can now use the trained models in the Streamlit app.")
    else:
        print("\n❌ Training failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 
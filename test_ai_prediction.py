#!/usr/bin/env python3
"""
Test AI prediction functionality
"""

import sys
import os
import joblib
import json
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_optimizer.config import AIConfig
from ai_optimizer.data_collector import PrometheusDataCollector
from ai_optimizer.optimization_engine import OptimizationEngine

def test_ai_prediction_workflow():
    """Test the complete AI prediction workflow"""
    print("🧪 Testing AI Prediction Workflow")
    print("="*50)
    
    try:
        # Initialize components
        ai_config = AIConfig()
        data_collector = PrometheusDataCollector(ai_config)
        optimization_engine = OptimizationEngine(ai_config, data_collector)
        
        print("✅ Components initialized")
        
        # Test VM metrics collection
        vm_name = "z-actegy-WEB1"
        print(f"\n📊 Testing VM metrics collection for: {vm_name}")
        
        vm_metrics = optimization_engine.get_vm_metrics(vm_name)
        print(f"✅ VM metrics collected: {len(vm_metrics)} metrics")
        print(f"  CPU Usage: {vm_metrics.get('cpu_usage', 0):.1%}")
        print(f"  RAM Usage: {vm_metrics.get('ram_usage', 0):.1%}")
        print(f"  Ready Time: {vm_metrics.get('ready_time', 0):.1%}")
        print(f"  I/O Usage: {vm_metrics.get('io_usage', 0):.1%}")
        
        # Test host metrics collection
        host_name = "lon-esxi8.zengenti.io"
        print(f"\n🏠 Testing host metrics collection for: {host_name}")
        
        host_metrics = data_collector.get_host_performance_metrics(host_name)
        if host_metrics:
            print(f"✅ Host metrics collected: {len(host_metrics)} metrics")
            print(f"  CPU Usage: {host_metrics.get('cpu_usage', 0):.1%}")
            print(f"  RAM Usage: {host_metrics.get('ram_usage', 0):.1%}")
            print(f"  I/O Usage: {host_metrics.get('io_usage', 0):.1%}")
            print(f"  VM Count: {host_metrics.get('vm_count', 0)}")
        else:
            print("⚠️ No host metrics available")
        
        # Test AI model loading
        print(f"\n🤖 Testing AI model loading")
        models_dir = "ai_optimizer/models"
        trained_models = {}
        model_scalers = {}
        
        model_types = ['random_forest', 'gradient_boosting']
        for model_type in model_types:
            model_path = os.path.join(models_dir, f"{model_type}_model.pkl")
            scaler_path = os.path.join(models_dir, f"{model_type}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                trained_models[model_type] = joblib.load(model_path)
                model_scalers[model_type] = joblib.load(scaler_path)
                print(f"✅ Loaded {model_type} model")
            else:
                print(f"❌ Model files not found for {model_type}")
        
        if trained_models:
            print(f"✅ Successfully loaded {len(trained_models)} models")
            
            # Test AI prediction
            if host_metrics:
                print(f"\n🎯 Testing AI prediction")
                
                # Prepare features for AI models
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
                
                # Get predictions from each model
                predictions = {}
                for model_name, model in trained_models.items():
                    if model_name in model_scalers:
                        scaler = model_scalers[model_name]
                        features_scaled = scaler.transform([feature_vector])
                        prediction = model.predict(features_scaled)[0]
                        predictions[model_name] = max(0.0, min(1.0, prediction))
                        print(f"  {model_name.upper()}: {predictions[model_name]:.3f}")
                
                # Calculate ensemble prediction
                if len(predictions) > 1:
                    ensemble_score = sum(predictions.values()) / len(predictions)
                    predictions['ensemble'] = ensemble_score
                    print(f"  ENSEMBLE: {ensemble_score:.3f}")
                
                print("✅ AI prediction test completed successfully")
            else:
                print("⚠️ Skipping AI prediction test - no host metrics available")
        else:
            print("❌ No trained models available for prediction test")
        
        print("\n✅ AI prediction workflow test completed successfully!")
        
    except Exception as e:
        print(f"❌ AI prediction workflow test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_prediction_workflow() 
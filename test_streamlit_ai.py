#!/usr/bin/env python3
"""
Test script to verify AI integration in Streamlit app
"""

import sys
import os
import joblib
import json
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ai_model_loading():
    """Test if AI models can be loaded correctly"""
    print("🧪 Testing AI Model Loading")
    print("="*40)
    
    models_dir = "ai_optimizer/models"
    trained_models = {}
    model_scalers = {}
    model_performance = {}
    
    # Test model performance loading
    try:
        results_path = os.path.join(models_dir, "training_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                model_performance = json.load(f)
            print("✅ Model performance data loaded successfully")
            
            for model_name, metrics in model_performance.items():
                print(f"  {model_name.upper()}: R² = {metrics['r2']:.4f}")
        else:
            print("❌ No training results found")
    except Exception as e:
        print(f"❌ Failed to load model performance: {e}")
    
    # Test model loading
    try:
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
        else:
            print("❌ No models loaded")
            
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
    
    return trained_models, model_scalers, model_performance

def test_ai_prediction():
    """Test AI prediction functionality"""
    print("\n🧪 Testing AI Prediction")
    print("="*40)
    
    trained_models, model_scalers, model_performance = test_ai_model_loading()
    
    if not trained_models:
        print("❌ Cannot test predictions without trained models")
        return
    
    # Create sample feature vector
    sample_features = [
        0.3,  # vm_cpu_usage
        0.4,  # vm_ram_usage
        0.05, # vm_ready_time
        0.2,  # vm_io_usage
        2400, # vm_cpu_mhz
        4096, # vm_ram_mb
        0.5,  # host_cpu_usage
        0.6,  # host_ram_usage
        0.1,  # host_io_usage
        0.02, # host_ready_time
        8,    # host_vm_count
        40000, # host_cpu_max_mhz
        65536, # host_ram_max_mb
        0.8,  # projected_cpu_usage
        1.0,  # projected_ram_usage
        0.3,  # projected_io_usage
        9     # projected_vm_count
    ]
    
    try:
        predictions = {}
        
        for model_name, model in trained_models.items():
            if model_name in model_scalers:
                scaler = model_scalers[model_name]
                features_scaled = scaler.transform([sample_features])
                prediction = model.predict(features_scaled)[0]
                predictions[model_name] = max(0.0, min(1.0, prediction))
                print(f"  {model_name.upper()}: {predictions[model_name]:.3f}")
        
        # Calculate ensemble
        if len(predictions) > 1:
            ensemble_score = sum(predictions.values()) / len(predictions)
            predictions['ensemble'] = ensemble_score
            print(f"  ENSEMBLE: {ensemble_score:.3f}")
        
        print("✅ AI prediction test completed successfully")
        
    except Exception as e:
        print(f"❌ AI prediction test failed: {e}")

def test_streamlit_integration():
    """Test Streamlit app integration"""
    print("\n🧪 Testing Streamlit Integration")
    print("="*40)
    
    try:
        # Test if we can import the necessary modules
        from ai_optimizer.config import AIConfig
        from ai_optimizer.data_collector import PrometheusDataCollector
        from ai_optimizer.optimization_engine import OptimizationEngine
        
        print("✅ AI modules imported successfully")
        
        # Test configuration
        ai_config = AIConfig()
        print("✅ AI configuration loaded")
        
        # Test data collector
        data_collector = PrometheusDataCollector(ai_config)
        print("✅ Data collector initialized")
        
        # Test optimization engine
        optimization_engine = OptimizationEngine(ai_config, data_collector)
        print("✅ Optimization engine initialized")
        
        print("✅ Streamlit integration test passed")
        
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")

def main():
    """Main test function"""
    print("🤖 AI Integration Test for vCenter DRS")
    print("="*50)
    
    # Run all tests
    test_ai_model_loading()
    test_ai_prediction()
    test_streamlit_integration()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 
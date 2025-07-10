#!/usr/bin/env python3
"""
Test script to verify the get_vm_metrics method fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'vcenter_drs'))

from vcenter_drs.ai_optimizer.optimization_engine import OptimizationEngine
from vcenter_drs.ai_optimizer.data_collector import PrometheusDataCollector
from vcenter_drs.ai_optimizer.config import AIConfig

def test_vm_metrics():
    """Test the get_vm_metrics method"""
    print("Testing get_vm_metrics method...")
    
    try:
        # Create config and data collector
        config = AIConfig()
        data_collector = PrometheusDataCollector(config)
        
        # Create optimization engine
        engine = OptimizationEngine(config, data_collector)
        
        # Test get_vm_metrics method
        vm_name = "test-vm"
        metrics = engine.get_vm_metrics(vm_name)
        
        print(f"✅ get_vm_metrics method works!")
        print(f"VM metrics: {metrics}")
        
        # Check if all required fields are present
        required_fields = ['cpu_usage', 'ram_usage', 'io_usage', 'ready_time', 'cpu_mhz', 'ram_mb']
        missing_fields = [field for field in required_fields if field not in metrics]
        
        if missing_fields:
            print(f"⚠️ Missing fields: {missing_fields}")
        else:
            print("✅ All required fields present")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_vm_metrics()
    if success:
        print("\n✅ VM metrics fix test passed!")
    else:
        print("\n❌ VM metrics fix test failed!") 
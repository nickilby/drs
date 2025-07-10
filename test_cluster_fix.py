#!/usr/bin/env python3
"""
Test script to verify cluster information is properly stored and retrieved
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'vcenter_drs'))

from vcenter_drs.rules.rules_engine import get_db_state

def test_cluster_information():
    """Test that cluster information is properly stored and retrieved"""
    print("Testing cluster information retrieval...")
    
    try:
        # Get database state
        clusters, hosts, vms = get_db_state()
        
        print(f"Found {len(clusters)} clusters")
        print(f"Found {len(hosts)} hosts")
        print(f"Found {len(vms)} VMs")
        
        # Check if VMs have cluster information
        vms_with_cluster = 0
        vms_without_cluster = 0
        
        for vm_id, vm_data in vms.items():
            if vm_data.get('cluster'):
                vms_with_cluster += 1
                print(f"✅ VM {vm_data['name']} has cluster: {vm_data['cluster']}")
            else:
                vms_without_cluster += 1
                print(f"⚠️ VM {vm_data['name']} has no cluster info")
        
        print(f"\n📊 Summary:")
        print(f"  VMs with cluster info: {vms_with_cluster}")
        print(f"  VMs without cluster info: {vms_without_cluster}")
        
        if vms_with_cluster > 0:
            print("✅ Cluster information is working!")
            return True
        else:
            print("❌ No VMs have cluster information")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_cluster_information()
    if success:
        print("\n✅ Cluster fix test passed!")
    else:
        print("\n❌ Cluster fix test failed!") 
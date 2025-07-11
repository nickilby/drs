#!/usr/bin/env python3
"""
Script to check database state for specific VMs
"""

import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

try:
    import sys
    sys.path.append('vcenter_drs')
    from rules.rules_engine import get_db_state
    from api.vcenter_client_pyvomi import VCenterPyVmomiClient
    import json
    
    def check_vm_datasets():
        """Check database and vCenter for specific VMs"""
        print("=== Checking VM Dataset Information ===\n")
        
        # Get database state
        print("1. Database State:")
        print("-" * 40)
        clusters, hosts, vms = get_db_state()
        
        # Find the specific VMs
        target_vms = ['z-content-reviews-SQL', 'z-content-reviews-WEB2']
        
        for vm_name in target_vms:
            vm_found = False
            for vm_id, vm_data in vms.items():
                if vm_data['name'] == vm_name:
                    vm_found = True
                    print(f"\nVM: {vm_name}")
                    print(f"  Host ID: {vm_data['host_id']}")
                    print(f"  Cluster ID: {vm_data['cluster_id']}")
                    print(f"  Dataset ID: {vm_data['dataset_id']}")
                    print(f"  Dataset Name: {vm_data.get('dataset_name', 'None')}")
                    print(f"  Power Status: {vm_data['power_status']}")
                    
                    # Get host name
                    host_name = hosts.get(vm_data['host_id'], {}).get('name', 'Unknown')
                    print(f"  Host Name: {host_name}")
                    
                    # Get cluster name
                    cluster_name = clusters.get(vm_data['cluster_id'], 'Unknown')
                    print(f"  Cluster Name: {cluster_name}")
                    break
            
            if not vm_found:
                print(f"\nVM: {vm_name} - NOT FOUND IN DATABASE")
        
        # Check vCenter state
        print("\n\n2. vCenter State:")
        print("-" * 40)
        
        try:
            client = VCenterPyVmomiClient()
            si = client.connect()
            content = si.RetrieveContent()
            
            for vm_name in target_vms:
                vm_found = False
                for dc in content.rootFolder.childEntity:
                    if hasattr(dc, 'hostFolder'):
                        for cluster in dc.hostFolder.childEntity:
                            if hasattr(cluster, 'host'):
                                for host in cluster.host:
                                    for vm in host.vm:
                                        if vm.name == vm_name:
                                            vm_found = True
                                            print(f"\nVM: {vm_name}")
                                            print(f"  Host: {host.name}")
                                            print(f"  Cluster: {cluster.name}")
                                            
                                            # Get datastore information
                                            datastore_name = None
                                            if hasattr(vm, 'datastore') and vm.datastore:
                                                datastore_name = vm.datastore[0].name
                                            print(f"  Datastore: {datastore_name}")
                                            print(f"  Power Status: {vm.runtime.powerState}")
                                            break
                                    if vm_found:
                                        break
                            if vm_found:
                                break
                        if vm_found:
                            break
                    if vm_found:
                        break
                
                if not vm_found:
                    print(f"\nVM: {vm_name} - NOT FOUND IN VCENTER")
            
            client.disconnect()
            
        except Exception as e:
            print(f"Error connecting to vCenter: {e}")
        
        # Check rules configuration
        print("\n\n3. Rules Configuration:")
        print("-" * 40)
        
        try:
            from rules.rules_engine import load_rules
            rules = load_rules()
            
            dataset_affinity_rules = [rule for rule in rules if rule.get('type') == 'dataset-affinity']
            print(f"Found {len(dataset_affinity_rules)} dataset-affinity rules:")
            
            for i, rule in enumerate(dataset_affinity_rules, 1):
                print(f"\nRule {i}:")
                print(f"  Type: {rule.get('type')}")
                print(f"  Role: {rule.get('role')}")
                print(f"  Dataset Pattern: {rule.get('dataset_pattern')}")
                print(f"  Name Pattern: {rule.get('name_pattern')}")
                print(f"  Level: {rule.get('level')}")
                
        except Exception as e:
            print(f"Error loading rules: {e}")
        
        print("\n\n4. Analysis:")
        print("-" * 40)
        print("This will help identify why the dataset-affinity violation is being triggered.")
        print("Check if the VMs are on datasets that match the required patterns.")
    
    if __name__ == "__main__":
        check_vm_datasets()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project directory")
    sys.exit(1) 
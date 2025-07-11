#!/usr/bin/env python3
"""
Test script to verify database and vCenter connections
"""

import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

try:
    from vcenter_drs.db.metrics_db import MetricsDB
    from vcenter_drs.api.vcenter_client_pyvomi import VCenterPyVmomiClient
    import json
    
    def test_database_connection():
        """Test database connection and basic operations"""
        print("Testing database connection...")
        
        # Load credentials
        with open('vcenter_drs/credentials.json', 'r') as f:
            creds = json.load(f)
        
        # Test database connection
        db = MetricsDB(
            host=creds['db_host'],
            user=creds['db_user'],
            password=creds['db_password'],
            database=creds['db_database']
        )
        
        try:
            db.connect()
            if db.conn:
                print("✓ Database connection successful")
                
                # Test basic query
                cursor = db.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM vms")
                vm_count = cursor.fetchone()[0]
                print(f"✓ Database query successful - {vm_count} VMs found")
                
                # Test specific VM query
                cursor.execute("""
                    SELECT v.name, d.name as dataset_name, v.power_status 
                    FROM vms v 
                    LEFT JOIN datasets d ON v.dataset_id = d.id 
                    WHERE v.name LIKE '%east-ayrshire%' 
                    ORDER BY v.name
                """)
                vms = cursor.fetchall()
                print(f"✓ Found {len(vms)} east-ayrshire VMs:")
                for vm in vms:
                    print(f"  - {vm[0]} -> Dataset: {vm[1]}, Status: {vm[2]}")
                
                cursor.close()
            else:
                print("✗ Database connection failed")
                return False
                
        except Exception as e:
            print(f"✗ Database error: {e}")
            return False
        finally:
            db.close()
        
        return True
    
    def test_vcenter_connection():
        """Test vCenter connection and basic operations"""
        print("\nTesting vCenter connection...")
        
        try:
            client = VCenterPyVmomiClient()
            si = client.connect()
            
            if si:
                print("✓ vCenter connection successful")
                
                # Get content
                content = si.RetrieveContent()
                print("✓ vCenter content retrieved")
                
                # Count VMs
                vm_count = 0
                for dc in content.rootFolder.childEntity:
                    if hasattr(dc, 'hostFolder'):
                        for cluster in dc.hostFolder.childEntity:
                            if hasattr(cluster, 'host'):
                                for host in cluster.host:
                                    vm_count += len(host.vm)
                
                print(f"✓ Found {vm_count} VMs in vCenter")
                
                # Check for specific VMs
                east_ayrshire_vms = []
                for dc in content.rootFolder.childEntity:
                    if hasattr(dc, 'hostFolder'):
                        for cluster in dc.hostFolder.childEntity:
                            if hasattr(cluster, 'host'):
                                for host in cluster.host:
                                    for vm in host.vm:
                                        if 'east-ayrshire' in vm.name:
                                            datastore_name = None
                                            if hasattr(vm, 'datastore') and vm.datastore:
                                                datastore_name = vm.datastore[0].name
                                            east_ayrshire_vms.append({
                                                'name': vm.name,
                                                'host': host.name,
                                                'cluster': cluster.name,
                                                'datastore': datastore_name,
                                                'power_status': vm.runtime.powerState
                                            })
                
                print(f"✓ Found {len(east_ayrshire_vms)} east-ayrshire VMs in vCenter:")
                for vm in east_ayrshire_vms:
                    print(f"  - {vm['name']} -> Host: {vm['host']}, Cluster: {vm['cluster']}, Datastore: {vm['datastore']}, Status: {vm['power_status']}")
                
                client.disconnect()
                return True
            else:
                print("✗ vCenter connection failed")
                return False
                
        except Exception as e:
            print(f"✗ vCenter error: {e}")
            return False
    
    def main():
        """Main test function"""
        print("=== Connection Tests ===\n")
        
        db_ok = test_database_connection()
        vcenter_ok = test_vcenter_connection()
        
        print(f"\n=== Summary ===")
        print(f"Database: {'✓ OK' if db_ok else '✗ FAILED'}")
        print(f"vCenter: {'✓ OK' if vcenter_ok else '✗ FAILED'}")
        
        if db_ok and vcenter_ok:
            print("\nBoth connections are working. The issue might be in the data collection logic.")
        else:
            print("\nConnection issues detected. Please check credentials and network connectivity.")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project directory")
    sys.exit(1) 
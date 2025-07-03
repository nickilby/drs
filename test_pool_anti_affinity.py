#!/usr/bin/env python3
"""
Test script for Pool Anti-Affinity Rules

This script demonstrates how the new pool anti-affinity rules work
by creating sample data and running compliance checks.
"""

import os
import sys
import json
from vcenter_drs.db.metrics_db import MetricsDB
from vcenter_drs.rules.rules_engine import extract_pool_from_dataset, evaluate_rules

def test_pool_extraction():
    """Test the pool name extraction function"""
    print("Testing Pool Name Extraction:")
    print("=" * 40)
    
    test_datasets = [
        "HQS1WEB1",
        "HQS1DAT1", 
        "HQS2WEB1",
        "HQS2DAT1",
        "L1WEB1",
        "M1DAT1",
        "POOL5WEB1",
        "STORAGE10DAT1",
        "INVALID_DATASET",
        ""
    ]
    
    for dataset in test_datasets:
        pool = extract_pool_from_dataset(dataset)
        print(f"Dataset: {dataset:15} -> Pool: {pool}")
    
    print()

def create_sample_data():
    """Create sample data for testing pool anti-affinity rules"""
    print("Creating Sample Data:")
    print("=" * 40)
    
    db = MetricsDB()
    db.connect()
    cursor = db.conn.cursor()
    
    try:
        # Clear existing data
        cursor.execute("DELETE FROM vms")
        cursor.execute("DELETE FROM datasets")
        cursor.execute("DELETE FROM hosts")
        cursor.execute("DELETE FROM clusters")
        
        # Create sample cluster
        cursor.execute("INSERT INTO clusters (name) VALUES ('TestCluster')")
        cluster_id = cursor.lastrowid
        
        # Create sample host
        cursor.execute("INSERT INTO hosts (name, cluster_id) VALUES ('esxi-test-01', %s)", (cluster_id,))
        host_id = cursor.lastrowid
        
        # Create sample datasets with different pools
        datasets = [
            ("HQS1WEB1", "hqs1"),
            ("HQS1WEB2", "hqs1"),
            ("HQS2WEB1", "hqs2"),
            ("HQS2WEB2", "hqs2"),
            ("L1WEB1", "l1"),
            ("M1WEB1", "m1"),
            ("POOL5WEB1", "pool5"),
            ("STORAGE10WEB1", "storage10")
        ]
        
        dataset_ids = {}
        for dataset_name, pool_name in datasets:
            cursor.execute(
                "INSERT INTO datasets (name, pool_name) VALUES (%s, %s)",
                (dataset_name, pool_name)
            )
            dataset_ids[dataset_name] = cursor.lastrowid
        
        # Create sample VMs with different roles
        vms = [
            ("z-app1-web1", "HQS1WEB1"),
            ("z-app1-web2", "HQS1WEB2"),  # Same pool as web1 - VIOLATION
            ("z-app1-web3", "HQS2WEB1"),  # Different pool - OK
            ("z-app1-lb1", "HQS1WEB1"),
            ("z-app1-lb2", "HQS1WEB2"),   # Same pool as lb1 - VIOLATION
            ("z-app1-cache1", "L1WEB1"),
            ("z-app1-cache2", "M1WEB1"),  # Different pool - OK
            ("z-app1-web4", "POOL5WEB1"), # Different pool - OK
            ("z-app1-web5", "STORAGE10WEB1") # Different pool - OK
        ]
        
        for vm_name, dataset_name in vms:
            dataset_id = dataset_ids[dataset_name]
            cursor.execute(
                "INSERT INTO vms (name, host_id, dataset_id) VALUES (%s, %s, %s)",
                (vm_name, host_id, dataset_id)
            )
        
        db.conn.commit()
        print("Sample data created successfully!")
        print(f"- 1 cluster, 1 host, {len(datasets)} datasets, {len(vms)} VMs")
        print()
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        db.conn.rollback()
    finally:
        cursor.close()
        db.close()

def test_pool_anti_affinity_rules():
    """Test the pool anti-affinity rules"""
    print("Testing Pool Anti-Affinity Rules:")
    print("=" * 40)
    
    # Create sample rules
    sample_rules = [
        {
            "type": "pool-anti-affinity",
            "role": ["WEB"],
            "pool_pattern": ["HQ", "L", "M"]
        },
        {
            "type": "pool-anti-affinity",
            "role": ["LB"],
            "pool_pattern": ["HQ", "L", "M"]
        },
        {
            "type": "pool-anti-affinity",
            "role": ["CACHE"],
            "pool_pattern": ["HQ", "L", "M"]
        }
    ]
    
    # Save rules to temporary file
    rules_file = "temp_rules.json"
    with open(rules_file, 'w') as f:
        json.dump(sample_rules, f, indent=2)
    
    try:
        # Run compliance check
        print("Running compliance check...")
        violations = evaluate_rules(return_structured=True)
        
        if violations:
            print(f"Found {len(violations)} violations:")
            print()
            for i, violation in enumerate(violations, 1):
                print(f"Violation {i}:")
                print(violation['violation_text'])
                print()
        else:
            print("No violations found!")
            
    except Exception as e:
        print(f"Error running compliance check: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(rules_file):
            os.remove(rules_file)

def main():
    """Main test function"""
    print("Pool Anti-Affinity Rules Test")
    print("=" * 50)
    print()
    
    # Test pool extraction
    test_pool_extraction()
    
    # Create sample data
    create_sample_data()
    
    # Test rules
    test_pool_anti_affinity_rules()
    
    print("Test completed!")

if __name__ == "__main__":
    main() 
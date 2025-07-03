#!/usr/bin/env python3
"""
Test case sensitivity for pool anti-affinity rules
"""

from vcenter_drs.rules.rules_engine import extract_pool_from_dataset

def test_pool_extraction():
    """Test pool name extraction"""
    print("Testing Pool Name Extraction:")
    print("=" * 40)
    
    test_datasets = [
        "HQS1WEB1",
        "HQS2DAT1", 
        "L1WEB1",
        "M1DAT1"
    ]
    
    for dataset in test_datasets:
        pool = extract_pool_from_dataset(dataset)
        print(f"Dataset: {dataset:15} -> Pool: {pool}")
    
    print()

def test_pattern_matching():
    """Test pattern matching with case sensitivity"""
    print("Testing Pattern Matching:")
    print("=" * 40)
    
    patterns = ["HQ", "L", "M"]
    test_pools = ["hqs1", "hqs2", "l1", "m1", "hqs3", "l2"]
    
    print(f"Patterns: {patterns}")
    print()
    
    for pool in test_pools:
        # Old way (case-sensitive)
        old_match = any(pat in pool for pat in patterns)
        
        # New way (case-insensitive)
        new_match = any(pat.lower() in pool.lower() for pat in patterns)
        
        print(f"Pool: {pool:8} | Old match: {old_match:5} | New match: {new_match:5}")
    
    print()

def test_rule_logic():
    """Test the actual rule logic"""
    print("Testing Rule Logic:")
    print("=" * 40)
    
    # Simulate your rule
    rule = {
        "type": "pool-anti-affinity",
        "role": ["WEB", "LB", "CACHE"],
        "pool_pattern": ["HQ", "L", "M"]
    }
    
    # Simulate VMs that should trigger violations
    test_vms = [
        {"name": "z-app1-web1", "role": "WEB", "dataset": "HQS1WEB1"},
        {"name": "z-app1-web2", "role": "WEB", "dataset": "HQS1WEB2"},  # Same pool as web1
        {"name": "z-app1-lb1", "role": "LB", "dataset": "HQS1WEB1"},    # Same pool as web1
        {"name": "z-app1-web3", "role": "WEB", "dataset": "HQS2WEB1"},  # Different pool
    ]
    
    patterns = rule["pool_pattern"]
    roles = rule["role"]
    
    print(f"Rule patterns: {patterns}")
    print(f"Rule roles: {roles}")
    print()
    
    # Group VMs by pool
    pool_groups = {}
    
    for vm in test_vms:
        pool_name = extract_pool_from_dataset(vm["dataset"])
        if pool_name and vm["role"] in roles:
            # Check if pool matches patterns (case-insensitive)
            if any(pat.lower() in pool_name.lower() for pat in patterns):
                if pool_name not in pool_groups:
                    pool_groups[pool_name] = []
                pool_groups[pool_name].append(vm)
    
    print("VMs grouped by pool:")
    for pool, vms in pool_groups.items():
        print(f"  Pool {pool}: {[vm['name'] for vm in vms]}")
        if len(vms) > 1:
            print(f"    ⚠️  VIOLATION: {len(vms)} VMs on same pool!")
        else:
            print(f"    ✅ OK: Only {len(vms)} VM on this pool")
        print()

if __name__ == "__main__":
    test_pool_extraction()
    test_pattern_matching()
    test_rule_logic() 
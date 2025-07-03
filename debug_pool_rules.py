#!/usr/bin/env python3
"""
Debug script to trace pool anti-affinity rules for specific example
"""

from vcenter_drs.rules.rules_engine import extract_pool_from_dataset, parse_alias_and_role

def debug_specific_example():
    """Debug the specific example: z-bcc-dev-WEB1 and z-bcc-dev-WEB2 on HQS5WEB1 and HQS5TRA1"""
    
    print("Debugging Pool Anti-Affinity Rules")
    print("=" * 50)
    
    # Your specific example
    vm1 = {"name": "z-bcc-dev-WEB1", "dataset": "HQS5WEB1"}
    vm2 = {"name": "z-bcc-dev-WEB2", "dataset": "HQS5TRA1"}
    
    print(f"VM1: {vm1['name']} on dataset {vm1['dataset']}")
    print(f"VM2: {vm2['name']} on dataset {vm2['dataset']}")
    print()
    
    # Step 1: Extract pool names
    pool1 = extract_pool_from_dataset(vm1['dataset'])
    pool2 = extract_pool_from_dataset(vm2['dataset'])
    
    print("Step 1: Pool Extraction")
    print(f"  {vm1['dataset']} -> Pool: {pool1}")
    print(f"  {vm2['dataset']} -> Pool: {pool2}")
    print()
    
    # Step 2: Parse alias and role
    alias1, role1 = parse_alias_and_role(vm1['name'])
    alias2, role2 = parse_alias_and_role(vm2['name'])
    
    print("Step 2: Alias and Role Parsing")
    print(f"  {vm1['name']} -> Alias: {alias1}, Role: {role1}")
    print(f"  {vm2['name']} -> Alias: {alias2}, Role: {role2}")
    print()
    
    # Step 3: Check rule patterns
    rule_patterns = ["HQ", "L", "M"]
    
    print("Step 3: Pattern Matching")
    print(f"  Rule patterns: {rule_patterns}")
    
    match1 = any(pat.lower() in pool1.lower() for pat in rule_patterns) if pool1 else False
    match2 = any(pat.lower() in pool2.lower() for pat in rule_patterns) if pool2 else False
    
    print(f"  Pool {pool1} matches patterns: {match1}")
    print(f"  Pool {pool2} matches patterns: {match2}")
    print()
    
    # Step 4: Check role matching
    rule_roles = ["WEB", "LB", "CACHE", "v14-ES", "controller-", "keeper", "etcd-", "request-handler-lb", "utility", "redis", "broker"]
    
    print("Step 4: Role Matching")
    print(f"  Rule roles: {rule_roles}")
    
    role_match1 = role1 in rule_roles if role1 else False
    role_match2 = role2 in rule_roles if role2 else False
    
    print(f"  Role {role1} in rule roles: {role_match1}")
    print(f"  Role {role2} in rule roles: {role_match2}")
    print()
    
    # Step 5: Check if VMs would be grouped together
    print("Step 5: VM Grouping Logic")
    
    if pool1 == pool2:
        print(f"  ✅ Same pool detected: {pool1}")
        if role1 == role2:
            print(f"  ✅ Same role detected: {role1}")
            print(f"  ⚠️  VIOLATION: Both VMs have same role on same pool!")
        else:
            print(f"  ❌ Different roles: {role1} vs {role2}")
            print(f"  ✅ No violation (different roles)")
    else:
        print(f"  ❌ Different pools: {pool1} vs {pool2}")
        print(f"  ✅ No violation (different pools)")
    
    print()
    
    # Step 6: Check the actual rule logic
    print("Step 6: Rule Logic Check")
    
    # Simulate the exact logic from rules_engine.py
    pool_groups = {}
    
    for vm in [vm1, vm2]:
        pool_name = extract_pool_from_dataset(vm['dataset'])
        alias, vm_role = parse_alias_and_role(vm['name'])
        
        print(f"  Processing {vm['name']}:")
        print(f"    Pool: {pool_name}")
        print(f"    Role: {vm_role}")
        print(f"    Role in rule_roles: {vm_role in rule_roles if vm_role else False}")
        print(f"    Pool matches patterns: {any(pat.lower() in pool_name.lower() for pat in rule_patterns) if pool_name else False}")
        
        if vm_role in rule_roles and pool_name and any(pat.lower() in pool_name.lower() for pat in rule_patterns):
            if pool_name not in pool_groups:
                pool_groups[pool_name] = []
            pool_groups[pool_name].append(vm)
            print(f"    ✅ Added to pool group {pool_name}")
        else:
            print(f"    ❌ Not added to any pool group")
        print()
    
    print("Final Pool Groups:")
    for pool, vms in pool_groups.items():
        print(f"  Pool {pool}: {[vm['name'] for vm in vms]}")
        if len(vms) > 1:
            print(f"    ⚠️  VIOLATION: {len(vms)} VMs on same pool!")
        else:
            print(f"    ✅ OK: Only {len(vms)} VM on this pool")

if __name__ == "__main__":
    debug_specific_example() 
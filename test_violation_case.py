#!/usr/bin/env python3
"""
Test script to verify violations are still detected when they should be
"""

from vcenter_drs.rules.rules_engine import extract_pool_from_dataset, parse_alias_and_role

def test_violation_case():
    """Test a case that SHOULD trigger a violation: same alias, same role, same pool"""
    
    print("Testing Violation Case")
    print("=" * 50)
    
    # This SHOULD trigger a violation
    vm1 = {"name": "z-staffscc-WEB1", "dataset": "M1S5WEB1"}
    vm2 = {"name": "z-staffscc-WEB2", "dataset": "M1S5WEB2"}
    
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
    
    # Step 3: Check the updated rule logic
    print("Step 3: Updated Rule Logic Check")
    
    rule_patterns = ["HQ", "L", "M"]
    rule_roles = ["WEB", "LB", "CACHE", "v14-ES", "controller-", "keeper", "etcd-", "request-handler-lb", "utility", "redis", "broker"]
    
    # Simulate the exact logic from rules_engine.py with the fix
    pool_alias_role_groups = {}
    
    for vm in [vm1, vm2]:
        pool_name = extract_pool_from_dataset(vm['dataset'])
        alias, vm_role = parse_alias_and_role(vm['name'])
        
        print(f"  Processing {vm['name']}:")
        print(f"    Pool: {pool_name}")
        print(f"    Role: {vm_role}")
        
        if vm_role in rule_roles and pool_name and any(pat.lower() in pool_name.lower() for pat in rule_patterns):
            group_key = (pool_name, alias, vm_role)
            if group_key not in pool_alias_role_groups:
                pool_alias_role_groups[group_key] = []
            pool_alias_role_groups[group_key].append(vm)
            print(f"    ✅ Added to pool group {group_key}")
        else:
            print(f"    ❌ Not added to any pool group")
        print()
    
    print("Final Pool Groups:")
    for (pool, alias, role), vms in pool_alias_role_groups.items():
        print(f"  Pool {pool}, Alias {alias}, Role {role}: {[vm['name'] for vm in vms]}")
        if len(vms) > 1:
            print(f"    ⚠️  VIOLATION: {len(vms)} VMs on same pool with same alias and role!")
        else:
            print(f"    ✅ OK: Only {len(vms)} VM on this pool with this alias and role")
    
    print()
    print("Summary:")
    if pool1 == pool2 and alias1 == alias2:
        if role1 == role2:
            print("  ⚠️  VIOLATION: Same pool, same alias, same role")
        else:
            print("  ✅ ALLOWED: Same pool, same alias, different roles")
    else:
        print("  ✅ ALLOWED: Different pools or different aliases")

if __name__ == "__main__":
    test_violation_case() 
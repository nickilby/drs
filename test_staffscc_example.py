#!/usr/bin/env python3
"""
Test script to verify the staffscc example works correctly
"""

from vcenter_drs.rules.rules_engine import extract_pool_from_dataset, parse_alias_and_role

def test_staffscc_example():
    """Test the staffscc example: z-staffscc-LB1 and z-staffscc-WEB1 on same pool"""
    
    print("Testing StaffSCC Example")
    print("=" * 50)
    
    # Your specific example
    vm1 = {"name": "z-staffscc-LB1", "dataset": "M1S5TRA1"}
    vm2 = {"name": "z-staffscc-WEB1", "dataset": "M1S5WEB1"}
    
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
    
    # Step 5: Check the updated rule logic
    print("Step 5: Updated Rule Logic Check")
    
    # Simulate the exact logic from rules_engine.py with the fix
    pool_alias_role_groups = {}
    
    for vm in [vm1, vm2]:
        pool_name = extract_pool_from_dataset(vm['dataset'])
        alias, vm_role = parse_alias_and_role(vm['name'])
        
        print(f"  Processing {vm['name']}:")
        print(f"    Pool: {pool_name}")
        print(f"    Role: {vm_role}")
        print(f"    Role in rule_roles: {vm_role in rule_roles if vm_role else False}")
        print(f"    Pool matches patterns: {any(pat.lower() in pool_name.lower() for pat in rule_patterns) if pool_name else False}")
        
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
    test_staffscc_example() 
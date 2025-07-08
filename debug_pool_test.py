#!/usr/bin/env python3

import re
from collections import defaultdict

def parse_alias_and_role(vm_name):
    # Remove 'z-' prefix if present
    if vm_name.startswith('z-'):
        name = vm_name[2:]
    else:
        name = vm_name

    # Remove trailing number (if any)
    match = re.match(r'(.+)-(\D+)(\d+)$', name)
    if match:
        alias = match.group(1)
        role = match.group(2)
        return alias.lower(), role.upper()
    # If no trailing number, split from right on dash
    if '-' in name:
        alias, role = name.rsplit('-', 1)
        return alias.lower(), role.upper()
    return None, None

def extract_pool_from_dataset(dataset_name):
    """
    Extract ZFS pool name from dataset name.
    """
    if not dataset_name:
        return None
    
    # Enhanced pattern: Extract letters followed by numbers, optionally followed by letters and numbers
    match = re.match(r'^([A-Z]+)([0-9]+)([A-Z]+[0-9]+)?', dataset_name, re.IGNORECASE)
    if match:
        letters = match.group(1).lower()
        numbers = match.group(2)
        suffix = match.group(3).lower() if match.group(3) else ""
        return f"{letters}{numbers}{suffix}"
    
    return None

# Test the specific case from the failing test
print("=== Testing Pool Anti-Affinity Logic ===")

# Test data from the failing test
rules = [{"type": "pool-anti-affinity", "level": "storage", "role": ["WEB"], "pool_pattern": ["HQS"]}]
vms = {
    1: {"name": "z-alias-WEB1", "host_id": 1, "dataset_id": 1, "dataset_name": "HQS1WEB1", "power_status": "poweredon"},
    2: {"name": "z-alias-WEB2", "host_id": 1, "dataset_id": 2, "dataset_name": "HQS1WEB2", "power_status": "poweredon"},
}

print(f"Rules: {rules}")
print(f"VMs: {vms}")

# Parse alias and role for each VM
vm_alias_role = {}
for vm_id, vm in vms.items():
    alias, role = parse_alias_and_role(vm['name'])
    vm_alias_role[vm_id] = (alias, role)
    print(f"VM {vm_id} ({vm['name']}): alias='{alias}', role='{role}'")

# Test pool extraction
for vm_id, vm in vms.items():
    pool_name = extract_pool_from_dataset(vm['dataset_name'])
    print(f"VM {vm_id} dataset '{vm['dataset_name']}' -> pool '{pool_name}'")

# Test the pool anti-affinity logic
patterns = rules[0]['pool_pattern']
role = rules[0]['role']

print(f"\nChecking rule: role={role}, patterns={patterns}")

pool_alias_role_groups = defaultdict(list)
for vm_id, vm in vms.items():
    alias, vm_role = vm_alias_role[vm_id]
    dataset_name = vm.get('dataset_name') or ''
    pool_name = extract_pool_from_dataset(dataset_name)
    
    print(f"VM {vm_id}: alias='{alias}', role='{vm_role}', pool='{pool_name}'")
    
    # Check if VM matches rule criteria
    role_matches = (isinstance(role, str) and vm_role == role) or (isinstance(role, list) and vm_role in role)
    pattern_matches = pool_name and any(pat.lower() in pool_name.lower() for pat in patterns)
    
    print(f"  Role matches: {role_matches}")
    print(f"  Pattern matches: {pattern_matches}")
    
    if role_matches and pattern_matches:
        pool_alias_role_groups[(pool_name, alias, vm_role)].append((vm_id, vm))
        print(f"  -> Added to group ({pool_name}, {alias}, {vm_role})")

print(f"\nGroups found: {dict(pool_alias_role_groups)}")

for (pool, alias, vm_role), group in pool_alias_role_groups.items():
    print(f"Group ({pool}, {alias}, {vm_role}): {len(group)} VMs")
    if len(group) > 1:
        print("  -> VIOLATION DETECTED!")
    else:
        print("  -> No violation (only 1 VM in group)") 
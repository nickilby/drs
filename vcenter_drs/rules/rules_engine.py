# Placeholder for rules engine logic 

import os
import json
from db.metrics_db import MetricsDB
from collections import defaultdict

def load_rules(rules_path=None):
    if rules_path is None:
        rules_path = os.path.join(os.path.dirname(__file__), 'rules.json')
    with open(rules_path, 'r') as f:
        return json.load(f)

def parse_alias_and_role(vm_name):
    # Assumes VM name format: z-<alias>-<role><number>
    parts = vm_name.split('-')
    if len(parts) >= 3:
        alias = parts[1]
        role_num = parts[2]
        role = ''.join([c for c in role_num if not c.isdigit()])
        return alias.lower(), role.upper()
    return None, None

def get_db_state():
    db = MetricsDB()
    db.connect()
    cursor = db.conn.cursor(dictionary=True)
    # Get clusters
    cursor.execute('SELECT * FROM clusters')
    clusters = {row['id']: row['name'] for row in cursor.fetchall()}
    # Get hosts
    cursor.execute('SELECT * FROM hosts')
    hosts = {row['id']: {'name': row['name'], 'cluster_id': row['cluster_id']} for row in cursor.fetchall()}
    # Get VMs
    cursor.execute('SELECT * FROM vms')
    vms = {row['id']: {'name': row['name'], 'host_id': row['host_id']} for row in cursor.fetchall()}
    cursor.close()
    db.close()
    return clusters, hosts, vms

def print_violation_header(title):
    print("\n" + "=" * 30)
    print(title)
    print("-" * 30)

def evaluate_rules():
    rules = load_rules()
    clusters, hosts, vms = get_db_state()
    vm_name_to_id = {vm['name']: vm_id for vm_id, vm in vms.items()}
    # Build VM alias/role map
    vm_alias_role = {}
    for vm_id, vm in vms.items():
        alias, role = parse_alias_and_role(vm['name'])
        vm_alias_role[vm_id] = (alias, role)
    # Group VMs by alias
    alias_to_vm_ids = defaultdict(list)
    for vm_id, (alias, role) in vm_alias_role.items():
        if alias:
            alias_to_vm_ids[alias].append(vm_id)

    for rule in rules:
        if 'role' not in rule:
            continue
        roles = [rule['role']] if isinstance(rule['role'], str) else rule['role']
        for alias, vm_ids in alias_to_vm_ids.items():
            # Find VMs in this alias group with the specified roles
            group_vm_ids = [vid for vid in vm_ids if vm_alias_role[vid][1] in roles]
            if len(group_vm_ids) < 2:
                continue  # Need at least two VMs to check
            host_ids = set(vms[vid]['host_id'] for vid in group_vm_ids)
            if rule['type'] == 'affinity':
                if rule['level'] == 'host' and len(host_ids) > 1:
                    print_violation_header("Affinity Violation")
                    print(f"Rule: {', '.join(roles)} VMs for alias '{alias}' should be on the same host")
                    print("VMs found on different hosts:")
                    for vid in group_vm_ids:
                        print(f"  - {vms[vid]['name']}")
                    print("Suggestions:")
                    print(f"  - Place all {', '.join(roles)} VMs for alias '{alias}' on the same host")
                elif rule['level'] == 'cluster':
                    cluster_ids = set(hosts[vms[vid]['host_id']]['cluster_id'] for vid in group_vm_ids)
                    if len(cluster_ids) > 1:
                        print_violation_header("Affinity Violation")
                        print(f"Rule: {', '.join(roles)} VMs for alias '{alias}' should be in the same cluster")
                        print("VMs found in different clusters:")
                        for vid in group_vm_ids:
                            print(f"  - {vms[vid]['name']}")
                        print("Suggestions:")
                        print(f"  - Place all {', '.join(roles)} VMs for alias '{alias}' in the same cluster")
            elif rule['type'] == 'anti-affinity':
                if rule['level'] == 'host' and len(host_ids) == 1:
                    print_violation_header("Anti-Affinity Violation")
                    print(f"Rule: {', '.join(roles)} VMs for alias '{alias}' must not be on the same host")
                    print(f"Host: {hosts[list(host_ids)[0]]['name']}")
                    print("VMs on this host:")
                    for vid in group_vm_ids:
                        print(f"  - {vms[vid]['name']}")
                    print("Suggestions:")
                    for vid in group_vm_ids[1:]:
                        print(f"  - Move VM {vms[vid]['name']} to a different host")
    # VM-specific anti-affinity (unchanged)
    if rule['type'] == 'anti-affinity' and 'vms' in rule:
        vm_ids = [vm_name_to_id.get(name) for name in rule['vms'] if vm_name_to_id.get(name)]
        host_groups = defaultdict(list)
        for vm_id in vm_ids:
            host_id = vms[vm_id]['host_id']
            host_groups[host_id].append(vm_id)
        for host_id, group in host_groups.items():
            if len(group) > 1:
                print_violation_header("Anti-Affinity Violation")
                print(f"Rule: VMs {rule['vms']} must not be on the same host")
                print(f"Host: {hosts[host_id]['name']}")
                print("VMs on this host:")
                for vid in group:
                    print(f"  - {vms[vid]['name']}")
                print("Suggestions:")
                for vid in group[1:]:
                    print(f"  - Move VM {vms[vid]['name']} to a different host")

if __name__ == "__main__":
    evaluate_rules() 
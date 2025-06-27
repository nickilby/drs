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

def evaluate_rules(cluster_filter=None):
    rules = load_rules()
    clusters, hosts, vms = get_db_state()
    
    # Filter clusters if a filter is provided
    if cluster_filter and cluster_filter != "All Clusters":
        if isinstance(cluster_filter, str):
            cluster_names = [cluster_filter]
        else:
            cluster_names = cluster_filter
        cluster_ids = [cid for cid, name in clusters.items() if name in cluster_names]
        hosts = {hid: h for hid, h in hosts.items() if h['cluster_id'] in cluster_ids}
        vms = {vid: v for vid, v in vms.items() if v['host_id'] in hosts}
    
    vm_name_to_id = {vm['name']: vm_id for vm_id, vm in vms.items()}
    vm_alias_role = {}
    for vm_id, vm in vms.items():
        alias, role = parse_alias_and_role(vm['name'])
        vm_alias_role[vm_id] = (alias, role)
    alias_to_vm_ids = defaultdict(list)
    for vm_id, (alias, role) in vm_alias_role.items():
        if alias:
            alias_to_vm_ids[alias].append(vm_id)

    # Group violations by cluster
    cluster_violations = defaultdict(list)

    for rule in rules:
        if 'role' not in rule:
            continue
        roles = [rule['role']] if isinstance(rule['role'], str) else rule['role']
        for alias, vm_ids in alias_to_vm_ids.items():
            group_vm_ids = [vid for vid in vm_ids if vm_alias_role[vid][1] in roles]
            if len(group_vm_ids) < 2:
                continue
            host_ids = set(vms[vid]['host_id'] for vid in group_vm_ids)
            # Find the clusters for these VMs
            cluster_ids = set(hosts[vms[vid]['host_id']]['cluster_id'] for vid in group_vm_ids)
            for cluster_id in cluster_ids:
                cluster_name = clusters[cluster_id]
                if rule['type'] == 'affinity':
                    if rule['level'] == 'host' and len(host_ids) > 1:
                        msg = [
                            "Affinity Violation",
                            f"Rule: {', '.join(roles)} VMs for alias '{alias}' should be on the same host",
                            "VMs found on different hosts:",
                        ]
                        msg += [f"  - {vms[vid]['name']}" for vid in group_vm_ids]
                        msg.append("Suggestions:")
                        msg.append(f"  - Place all {', '.join(roles)} VMs for alias '{alias}' on the same host")
                        cluster_violations[cluster_name].append("\n".join(msg))
                    elif rule['level'] == 'cluster' and len(cluster_ids) > 1:
                        msg = [
                            "Affinity Violation",
                            f"Rule: {', '.join(roles)} VMs for alias '{alias}' should be in the same cluster",
                            "VMs found in different clusters:",
                        ]
                        msg += [f"  - {vms[vid]['name']}" for vid in group_vm_ids]
                        msg.append("Suggestions:")
                        msg.append(f"  - Place all {', '.join(roles)} VMs for alias '{alias}' in the same cluster")
                        cluster_violations[cluster_name].append("\n".join(msg))
                elif rule['type'] == 'anti-affinity':
                    if rule['level'] == 'host' and len(host_ids) == 1:
                        host_id = list(host_ids)[0]
                        msg = [
                            "Anti-Affinity Violation",
                            f"Rule: {', '.join(roles)} VMs for alias '{alias}' must not be on the same host",
                            f"Host: {hosts[host_id]['name']}",
                            "VMs on this host:",
                        ]
                        msg += [f"  - {vms[vid]['name']}" for vid in group_vm_ids]
                        msg.append("Suggestions:")
                        msg += [f"  - Move VM {vms[vid]['name']} to a different host" for vid in group_vm_ids[1:]]
                        cluster_violations[cluster_name].append("\n".join(msg))
    # VM-specific anti-affinity (unchanged, but grouped by cluster)
    for rule in rules:
        if rule['type'] == 'anti-affinity' and 'vms' in rule:
            vm_ids = [vm_name_to_id.get(name) for name in rule['vms'] if vm_name_to_id.get(name)]
            host_groups = defaultdict(list)
            for vm_id in vm_ids:
                host_id = vms[vm_id]['host_id']
                host_cluster_id = hosts[host_id]['cluster_id']
                host_groups[(host_id, host_cluster_id)].append(vm_id)
            for (host_id, cluster_id), group in host_groups.items():
                if len(group) > 1:
                    cluster_name = clusters[cluster_id]
                    msg = [
                        "Anti-Affinity Violation",
                        f"Rule: VMs {rule['vms']} must not be on the same host",
                        f"Host: {hosts[host_id]['name']}",
                        "VMs on this host:",
                    ]
                    msg += [f"  - {vms[vid]['name']}" for vid in group]
                    msg.append("Suggestions:")
                    msg += [f"  - Move VM {vms[vid]['name']} to a different host" for vid in group[1:]]
                    cluster_violations[cluster_name].append("\n".join(msg))

    # Print violations grouped by cluster
    for cluster_name, violations in cluster_violations.items():
        print("\n" + "#" * 40)
        print(f"Cluster: {cluster_name}")
        print("#" * 40)
        for v in violations:
            print(v)
            print("\n" + "-" * 30)

if __name__ == "__main__":
    evaluate_rules() 
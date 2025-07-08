# Placeholder for rules engine logic 

import os
import json
from vcenter_drs.db.metrics_db import MetricsDB
from collections import defaultdict
import re

def load_rules(rules_path=None):
    if rules_path is None:
        rules_path = os.path.join(os.path.dirname(__file__), 'rules.json')
    with open(rules_path, 'r') as f:
        return json.load(f)

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
    
    Generic pattern that handles various pool naming conventions:
    - HQS1DAT1 -> hqs1
    - HQS2WEB1 -> hqs2
    - L1DAT1 -> l1
    - M1S2TRA1 -> m1s2
    - M1S4WEB1 -> m1s4
    - POOL5DAT1 -> pool5
    - STORAGE10WEB1 -> storage10
    
    Args:
        dataset_name (str): Dataset name like 'HQS1DAT1'
        
    Returns:
        str: Pool name or None if cannot be extracted
    """
    if not dataset_name:
        return None
    
    # Pattern: Extract letters followed by numbers (the pool name)
    # This handles: HQS1, L1, M1S2, M1S4, POOL5, STORAGE10, etc.
    # For HQS1DAT1, we want to extract just HQS1, not HQS1DAT1
    match = re.match(r'^([A-Z]+[0-9]+)', dataset_name, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    return None

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
    # Get VMs (with dataset info and power status)
    cursor.execute('''
        SELECT v.id, v.name, v.host_id, v.dataset_id, d.name as dataset_name, v.power_status
        FROM vms v
        LEFT JOIN datasets d ON v.dataset_id = d.id
    ''')
    vms = {row['id']: {'name': row['name'], 'host_id': row['host_id'], 'dataset_id': row['dataset_id'], 'dataset_name': row['dataset_name'], 'power_status': row['power_status']} for row in cursor.fetchall()}
    cursor.close()
    db.close()
    return clusters, hosts, vms

def print_violation_header(title):
    print("\n" + "=" * 30)
    print(title)
    print("-" * 30)

def evaluate_rules(cluster_filter=None, return_structured=False):
    rules = load_rules()
    clusters, hosts, vms = get_db_state()
    db = MetricsDB()
    db.connect()
    
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
    structured_violations = []

    def violation_is_exception(violation):
        return db.is_exception(violation)

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
                        violation_text = "\n".join(msg)
                        violation_obj = {
                            "type": rule['type'],
                            "rule": rule,
                            "alias": alias,
                            "affected_vms": [vms[vid]['name'] for vid in group_vm_ids],
                            "cluster": cluster_name,
                            "violation_text": violation_text,
                            "level": rule.get('level', 'host')
                        }
                        if violation_is_exception(violation_obj):
                            continue
                        cluster_violations[cluster_name].append(violation_text)
                        structured_violations.append(violation_obj)
                    elif rule['level'] == 'cluster' and len(cluster_ids) > 1:
                        msg = [
                            "Affinity Violation",
                            f"Rule: {', '.join(roles)} VMs for alias '{alias}' should be in the same cluster",
                            "VMs found in different clusters:",
                        ]
                        msg += [f"  - {vms[vid]['name']}" for vid in group_vm_ids]
                        msg.append("Suggestions:")
                        msg.append(f"  - Place all {', '.join(roles)} VMs for alias '{alias}' in the same cluster")
                        violation_text = "\n".join(msg)
                        violation_obj = {
                            "type": rule['type'],
                            "rule": rule,
                            "alias": alias,
                            "affected_vms": [vms[vid]['name'] for vid in group_vm_ids],
                            "cluster": cluster_name,
                            "violation_text": violation_text,
                            "level": rule.get('level', 'host')
                        }
                        if violation_is_exception(violation_obj):
                            continue
                        cluster_violations[cluster_name].append(violation_text)
                        structured_violations.append(violation_obj)
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
                        violation_text = "\n".join(msg)
                        violation_obj = {
                            "type": rule['type'],
                            "rule": rule,
                            "alias": alias,
                            "affected_vms": [vms[vid]['name'] for vid in group_vm_ids],
                            "cluster": cluster_name,
                            "violation_text": violation_text,
                            "level": rule.get('level', 'host')
                        }
                        if violation_is_exception(violation_obj):
                            continue
                        cluster_violations[cluster_name].append(violation_text)
                        structured_violations.append(violation_obj)
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
                    violation_text = "\n".join(msg)
                    violation_obj = {
                        "type": rule['type'],
                        "rule": rule,
                        "alias": None,
                        "affected_vms": [vms[vid]['name'] for vid in group],
                        "cluster": cluster_name,
                        "violation_text": violation_text,
                        "level": rule.get('level', 'host')
                    }
                    if violation_is_exception(violation_obj):
                        continue
                    cluster_violations[cluster_name].append(violation_text)
                    structured_violations.append(violation_obj)

    # Dataset affinity rules
    processed_vms = set()
    for rule in rules:
        if rule.get('type') == 'dataset-affinity' and 'dataset_pattern' in rule:
            patterns = rule['dataset_pattern']
            # Role-based
            if 'role' in rule:
                role = rule['role'].upper() if isinstance(rule['role'], str) else [r.upper() for r in rule['role']]
                for vm_id, vm in vms.items():
                    if vm_id in processed_vms:
                        continue
                    alias, vm_role = vm_alias_role[vm_id]
                    if (isinstance(role, str) and vm_role == role) or (isinstance(role, list) and vm_role in role):
                        dataset_name = vm.get('dataset_name') or ''
                        if not any(pat in dataset_name for pat in patterns):
                            host = hosts.get(vm['host_id'])
                            if not host:
                                continue
                            cluster_id = host['cluster_id']
                            cluster_name = clusters[cluster_id]
                            msg = [
                                "Dataset Affinity Violation",
                                f"Rule: {vm['name']} (role: {vm_role}) must be on a dataset containing {patterns}",
                                f"Current dataset: {dataset_name if dataset_name else 'None'}",
                                "Suggestions:",
                                f"  - Move VM {vm['name']} to a datastore containing one of: {patterns}"
                            ]
                            violation_text = "\n".join(msg)
                            violation_obj = {
                                "type": rule['type'],
                                "rule": rule,
                                "alias": alias,
                                "affected_vms": [vm['name']],
                                "cluster": cluster_name,
                                "violation_text": violation_text,
                                "level": rule.get('level', 'host')
                            }
                            if violation_is_exception(violation_obj):
                                continue
                            cluster_violations[cluster_name].append(violation_text)
                            structured_violations.append(violation_obj)
                        processed_vms.add(vm_id)
            # Name-pattern-based
            if 'name_pattern' in rule:
                name_pattern = rule['name_pattern']
                for vm_id, vm in vms.items():
                    if vm_id in processed_vms:
                        continue
                    if name_pattern in vm['name']:
                        alias, vm_role = vm_alias_role[vm_id]
                        dataset_name = vm.get('dataset_name') or ''
                        if not any(pat in dataset_name for pat in patterns):
                            host = hosts.get(vm['host_id'])
                            if not host:
                                continue
                            cluster_id = host['cluster_id']
                            cluster_name = clusters[cluster_id]
                            msg = [
                                "Dataset Affinity Violation",
                                f"Rule: {vm['name']} (name contains: {name_pattern}) must be on a dataset containing {patterns}",
                                f"Current dataset: {dataset_name if dataset_name else 'None'}",
                                "Suggestions:",
                                f"  - Move VM {vm['name']} to a datastore containing one of: {patterns}"
                            ]
                            violation_text = "\n".join(msg)
                            violation_obj = {
                                "type": rule['type'],
                                "rule": rule,
                                "alias": alias,
                                "affected_vms": [vm['name']],
                                "cluster": cluster_name,
                                "violation_text": violation_text,
                                "level": rule.get('level', 'host')
                            }
                            if violation_is_exception(violation_obj):
                                continue
                            cluster_violations[cluster_name].append(violation_text)
                            structured_violations.append(violation_obj)
                        processed_vms.add(vm_id)

    # Pool anti-affinity rules (new rule type for ZFS pool distribution)
    for rule in rules:
        if rule.get('type') == 'pool-anti-affinity' and 'pool_pattern' in rule:
            patterns = rule['pool_pattern']
            # Role-based pool anti-affinity
            if 'role' in rule:
                role = rule['role'].upper() if isinstance(rule['role'], str) else [r.upper() for r in rule['role']]
                # Group VMs by pool, alias, and role (allow different roles on same pool)
                pool_alias_role_groups = defaultdict(list)
                for vm_id, vm in vms.items():
                    alias, vm_role = vm_alias_role[vm_id]
                    dataset_name = vm.get('dataset_name') or ''
                    # Extract pool name from dataset (assuming format like HQS5WEB1, HQS5DAT1 where HQS5 is the pool)
                    pool_name = extract_pool_from_dataset(dataset_name)
                    if ((isinstance(role, str) and vm_role == role) or (isinstance(role, list) and vm_role in role)) and pool_name and any(pat.lower() in pool_name.lower() for pat in patterns):
                        # Group by pool, alias, and role combination (allow different roles on same pool)
                        pool_alias_role_groups[(pool_name, alias, vm_role)].append((vm_id, vm))
                for (pool, alias, vm_role), group in pool_alias_role_groups.items():
                    if len(group) > 1:
                        # Violation: more than one matching VM on the same pool with same alias and role
                        cluster_ids = set(hosts[vm['host_id']]['cluster_id'] for vm_id, vm in group)
                        for cluster_id in cluster_ids:
                            cluster_name = clusters[cluster_id]
                            vms_on_pool = [vm['name'] for vm_id, vm in group if hosts[vm['host_id']]['cluster_id'] == cluster_id]
                            msg = [
                                "Pool Anti-Affinity Violation",
                                f"Rule: VMs with alias '{alias}' and role '{vm_role}' must NOT be on the same ZFS pool matching {patterns}",
                                f"Pool: {pool}",
                                f"Alias: {alias}",
                                f"Role: {vm_role}",
                                "VMs on this pool:",
                            ]
                            msg += [f"  - {name}" for name in vms_on_pool]
                            msg.append("Suggestions:")
                            msg += [f"  - Move VM {name} to a different ZFS pool" for name in vms_on_pool[1:]]
                            violation_text = "\n".join(msg)
                            violation_obj = {
                                "type": rule['type'],
                                "rule": rule,
                                "alias": alias,
                                "affected_vms": vms_on_pool,
                                "cluster": cluster_name,
                                "violation_text": violation_text,
                                "level": rule.get('level', 'host')
                            }
                            if violation_is_exception(violation_obj):
                                continue
                            cluster_violations[cluster_name].append(violation_text)
                            structured_violations.append(violation_obj)
            # Name-pattern-based pool anti-affinity
            if 'name_pattern' in rule:
                name_pattern = rule['name_pattern']
                pool_groups = defaultdict(list)
                for vm_id, vm in vms.items():
                    if name_pattern in vm['name']:
                        alias, vm_role = vm_alias_role[vm_id]
                        dataset_name = vm.get('dataset_name') or ''
                        pool_name = extract_pool_from_dataset(dataset_name)
                        if pool_name and any(pat.lower() in pool_name.lower() for pat in patterns):
                            pool_groups[pool_name].append((vm_id, vm))
                for pool, group in pool_groups.items():
                    if len(group) > 1:
                        cluster_ids = set(hosts[vm['host_id']]['cluster_id'] for vm_id, vm in group)
                        for cluster_id in cluster_ids:
                            cluster_name = clusters[cluster_id]
                            vms_on_pool = [vm['name'] for vm_id, vm in group if hosts[vm['host_id']]['cluster_id'] == cluster_id]
                            msg = [
                                "Pool Anti-Affinity Violation",
                                f"Rule: VMs with name containing '{name_pattern}' must NOT be on the same ZFS pool matching {patterns}",
                                f"Pool: {pool}",
                                "VMs on this pool:",
                            ]
                            msg += [f"  - {name}" for name in vms_on_pool]
                            msg.append("Suggestions:")
                            msg += [f"  - Move VM {name} to a different ZFS pool" for name in vms_on_pool[1:]]
                            violation_text = "\n".join(msg)
                            violation_obj = {
                                "type": rule['type'],
                                "rule": rule,
                                "alias": None,
                                "affected_vms": vms_on_pool,
                                "cluster": cluster_name,
                                "violation_text": violation_text,
                                "level": rule.get('level', 'host')
                            }
                            if violation_is_exception(violation_obj):
                                continue
                            cluster_violations[cluster_name].append(violation_text)
                            structured_violations.append(violation_obj)

    # Dataset anti-affinity rules
    for rule in rules:
        if rule.get('type') == 'dataset-anti-affinity' and 'dataset_pattern' in rule:
            patterns = rule['dataset_pattern']
            # Role-based
            if 'role' in rule:
                role = rule['role'].upper() if isinstance(rule['role'], str) else [r.upper() for r in rule['role']]
                # Group VMs by dataset
                dataset_groups = defaultdict(list)
                for vm_id, vm in vms.items():
                    alias, vm_role = vm_alias_role[vm_id]
                    dataset_name = vm.get('dataset_name') or ''
                    if ((isinstance(role, str) and vm_role == role) or (isinstance(role, list) and vm_role in role)) and any(pat in dataset_name for pat in patterns):
                        dataset_groups[dataset_name].append((vm_id, vm))
                for dataset, group in dataset_groups.items():
                    if len(group) > 1:
                        # Violation: more than one matching VM on the same dataset
                        cluster_ids = set(hosts[vm['host_id']]['cluster_id'] for vm_id, vm in group)
                        for cluster_id in cluster_ids:
                            cluster_name = clusters[cluster_id]
                            vms_on_dataset = [vm['name'] for vm_id, vm in group if hosts[vm['host_id']]['cluster_id'] == cluster_id]
                            msg = [
                                "Dataset Anti-Affinity Violation",
                                f"Rule: VMs with role(s) {role} must NOT be on the same dataset matching {patterns}",
                                f"Dataset: {dataset}",
                                "VMs on this dataset:",
                            ]
                            msg += [f"  - {name}" for name in vms_on_dataset]
                            msg.append("Suggestions:")
                            msg += [f"  - Move VM {name} to a different dataset" for name in vms_on_dataset[1:]]
                            violation_text = "\n".join(msg)
                            violation_obj = {
                                "type": rule['type'],
                                "rule": rule,
                                "alias": None,
                                "affected_vms": vms_on_dataset,
                                "cluster": cluster_name,
                                "violation_text": violation_text,
                                "level": rule.get('level', 'host')
                            }
                            if violation_is_exception(violation_obj):
                                continue
                            cluster_violations[cluster_name].append(violation_text)
                            structured_violations.append(violation_obj)
            # Name-pattern-based
            if 'name_pattern' in rule:
                name_pattern = rule['name_pattern']
                dataset_groups = defaultdict(list)
                for vm_id, vm in vms.items():
                    if name_pattern in vm['name']:
                        alias, vm_role = vm_alias_role[vm_id]
                        dataset_name = vm.get('dataset_name') or ''
                        if any(pat in dataset_name for pat in patterns):
                            dataset_groups[dataset_name].append((vm_id, vm))
                for dataset, group in dataset_groups.items():
                    if len(group) > 1:
                        cluster_ids = set(hosts[vm['host_id']]['cluster_id'] for vm_id, vm in group)
                        for cluster_id in cluster_ids:
                            cluster_name = clusters[cluster_id]
                            vms_on_dataset = [vm['name'] for vm_id, vm in group if hosts[vm['host_id']]['cluster_id'] == cluster_id]
                            msg = [
                                "Dataset Anti-Affinity Violation",
                                f"Rule: VMs with name containing '{name_pattern}' must NOT be on the same dataset matching {patterns}",
                                f"Dataset: {dataset}",
                                "VMs on this dataset:",
                            ]
                            msg += [f"  - {name}" for name in vms_on_dataset]
                            msg.append("Suggestions:")
                            msg += [f"  - Move VM {name} to a different dataset" for name in vms_on_dataset[1:]]
                            violation_text = "\n".join(msg)
                            violation_obj = {
                                "type": rule['type'],
                                "rule": rule,
                                "alias": None,
                                "affected_vms": vms_on_dataset,
                                "cluster": cluster_name,
                                "violation_text": violation_text,
                                "level": rule.get('level', 'host')
                            }
                            if violation_is_exception(violation_obj):
                                continue
                            cluster_violations[cluster_name].append(violation_text)
                            structured_violations.append(violation_obj)

    # Print violations grouped by cluster
    for cluster_name, violations in cluster_violations.items():
        print("\n" + "#" * 40)
        print(f"Cluster: {cluster_name}")
        print("#" * 40)
        for v in violations:
            print(v)
            print("\n" + "-" * 30)

    if return_structured:
        db.close()
        return structured_violations

if __name__ == "__main__":
    evaluate_rules() 
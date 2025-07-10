import streamlit as st
import sys
import os
from typing import Dict, Any
from vcenter_drs.api.collect_and_store_metrics import main as collect_and_store_metrics_main
from vcenter_drs.rules.rules_engine import evaluate_rules, get_db_state, load_rules, parse_alias_and_role
import time
import threading
from collections import defaultdict
from vcenter_drs.db.metrics_db import MetricsDB
import json
from prometheus_client import start_http_server, Gauge, Counter, Histogram, REGISTRY
import atexit
import requests

API_BASE_URL = "https://pap.zengenti.com/"

def bold_unicode(text):
    bold_map = {c: chr(ord(c) + 0x1D400 - ord('A')) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    bold_map.update({c: chr(ord(c) + 0x1D41A - ord('a')) for c in 'abcdefghijklmnopqrstuvwxyz'})
    bold_map.update({c: chr(ord(c) + 0x1D7CE - ord('0')) for c in '0123456789'})
    return ''.join([str(bold_map.get(ch, ch)) for ch in text])

def trigger_remediation_api(alias, affected_vms, token, playbook_name="e-vmotion-server", priority="normal"):
    endpoint = API_BASE_URL.rstrip('/') + '/execute_playbook/'
    payload = {
        "alias": alias,
        "playbook_name": playbook_name,
        "priority": priority,
        "options": {
            "limit": affected_vms
        }
    }
    headers = {
        "Content-Type": "application/json",
        "X-Security-Token": token
    }
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            resp_json = response.json()
            if resp_json.get('success'):
                task_id = resp_json.get('new_task_id')
                url = f"https://dashboard.zengenti.com/env/{alias}/history/{task_id}" if task_id else None
                msg = f"Remediation triggered! Task ID: {task_id} (deduplicated: {resp_json.get('deduplicated')})"
                if url:
                    msg += f"\n[View Task History]({url})"
                return True, msg
            else:
                return False, f"Remediation API error: {resp_json}"
        else:
            return False, f"API call failed: {response.status_code} {response.text}"
    except Exception as e:
        return False, f"API call error: {e}"

st.set_page_config(page_title="vCenter DRS Compliance Dashboard", layout="wide")

# Prometheus Metrics Setup
# Start metrics server in background thread
def start_metrics_server():
    try:
        start_http_server(8081)
        print("Prometheus metrics server started on port 8081")
    except Exception as e:
        print(f"Failed to start metrics server: {e}")

# Initialize metrics only once
def get_or_create_metric(metric_type, name, description, *args, **kwargs):
    """Get existing metric or create new one to avoid duplicates"""
    try:
        # Try to get existing metric
        return REGISTRY.get_sample_value(name)
    except:
        # Create new metric if it doesn't exist
        return metric_type(name, description, *args, **kwargs)

# Start metrics server if not already running
if not hasattr(st.session_state, 'metrics_server_started'):
    metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
    metrics_thread.start()
    st.session_state.metrics_server_started = True

# Define Prometheus metrics (only create if they don't exist)
try:
    SERVICE_UP = Gauge('vcenter_drs_service_up', 'Service status (1=up, 0=down)')
    RULE_VIOLATIONS = Gauge('vcenter_drs_rule_violations_total', 'Current rule violations by type', ['rule_type'])
    VM_COUNT = Gauge('vcenter_drs_vm_count', 'Total number of VMs monitored')
    HOST_COUNT = Gauge('vcenter_drs_host_count', 'Total number of hosts monitored')
    LAST_COLLECTION_TIME = Gauge('vcenter_drs_last_collection_timestamp', 'Timestamp of last metrics collection')
    COMPLIANCE_CHECK_DURATION = Histogram('vcenter_drs_compliance_check_duration_seconds', 'Duration of compliance checks')
    UPTIME = Gauge('vcenter_drs_uptime_seconds', 'Service uptime in seconds')
except ValueError:
    # Metrics already exist, get them from registry
    SERVICE_UP = REGISTRY._names_to_collectors['vcenter_drs_service_up']  # type: ignore
    RULE_VIOLATIONS = REGISTRY._names_to_collectors['vcenter_drs_rule_violations_total']  # type: ignore
    VM_COUNT = REGISTRY._names_to_collectors['vcenter_drs_vm_count']  # type: ignore
    HOST_COUNT = REGISTRY._names_to_collectors['vcenter_drs_host_count']  # type: ignore
    LAST_COLLECTION_TIME = REGISTRY._names_to_collectors['vcenter_drs_last_collection_timestamp']  # type: ignore
    COMPLIANCE_CHECK_DURATION = REGISTRY._names_to_collectors['vcenter_drs_compliance_check_duration_seconds']  # type: ignore
    UPTIME = REGISTRY._names_to_collectors['vcenter_drs_uptime_seconds']  # type: ignore

# Set service as up
SERVICE_UP.set(1)
start_time = time.time()

# Load VM power status from DB
_, _, vms_db = get_db_state()
vm_power_status = {vm['name']: str(vm.get('power_status') or '').lower() for vm in vms_db.values()}

def update_violation_metrics(violations=None):
    """Update Prometheus metrics based on current violations"""
    # Load VM power status from DB
    _, _, vms_db = get_db_state()
    vm_power_status = {vm['name']: str(vm.get('power_status') or '').lower() for vm in vms_db.values()}
    
    # Count violations by rule type (only if at least one affected VM is powered on)
    violation_counts: Dict[str, int] = defaultdict(int)
    
    # Use provided violations or fall back to session state
    violations_to_check = violations if violations is not None else st.session_state.get('violations', [])
    
    if violations_to_check:
        for violation in violations_to_check:
            affected_vms = violation.get('affected_vms', [])
            any_powered_on = any(vm_power_status.get(vm_name, '') == 'poweredon' for vm_name in affected_vms)
            if any_powered_on:
                rule_type = violation.get('type', 'unknown')
                violation_counts[rule_type] += 1

    # Always ensure metrics exist by setting all known rule types to 0 first
    known_rule_types = ['anti-affinity', 'dataset-affinity', 'dataset-anti-affinity', 'affinity', 'pool-anti-affinity']
    for rule_type in known_rule_types:
        RULE_VIOLATIONS.labels(rule_type=rule_type).set(0)

    # Then set the actual counts
    for rule_type, count in violation_counts.items():
        RULE_VIOLATIONS.labels(rule_type=rule_type).set(count)

# Initial metrics update
update_violation_metrics()

# Update uptime periodically
def update_uptime():
    while True:
        UPTIME.set(time.time() - start_time)
        # Also update violation metrics periodically
        update_violation_metrics()
        time.sleep(60)  # Update every minute

uptime_thread = threading.Thread(target=update_uptime, daemon=True)
uptime_thread.start()

# Cleanup function for graceful shutdown
def cleanup():
    try:
        SERVICE_UP.set(0)
    except:
        pass
    print("vCenter DRS service shutting down...")

atexit.register(cleanup)

st.title("vCenter DRS Compliance Dashboard")

# Sidebar: API Authentication
with st.sidebar:
    st.header("API Authentication")
    username = st.text_input("Username", value=st.session_state.get('auth_username', ''))
    password = st.text_input("Password", type="password", value=st.session_state.get('auth_password', ''))
    if st.button("Authenticate"):
        auth_url = API_BASE_URL.rstrip('/') + '/users/authenticate'
        try:
            resp = requests.post(auth_url, json={"username": username, "password": password}, timeout=10)
            if resp.status_code == 200:
                resp_json = resp.json()
                token = resp_json.get("token")
                if token:
                    st.session_state['remediation_token'] = token
                    st.session_state['auth_username'] = username
                    st.session_state['auth_password'] = password
                    st.success("Successfully authenticated. Token stored for session.")
                else:
                    st.error(f"Authentication failed: No token in response. {resp_json}")
            else:
                st.error(f"Authentication failed: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Authentication error: {e}")
    if st.session_state.get('remediation_token'):
        st.info("Token is set for this session.")
        show_token = st.checkbox("Show token", value=False)
        if show_token:
            st.code(st.session_state['remediation_token'], language=None)

# Load clusters from the database
def get_clusters():
    clusters, hosts, vms = get_db_state()
    return list(clusters.values())

clusters = get_clusters()
selected_cluster = st.selectbox("Select a cluster to check compliance:", ["All Clusters"] + clusters)

st.write(f"### Compliance Results for: {selected_cluster}")

# Patch: temporarily redirect stdout to capture rules engine output
def run_rules_engine_for_cluster(selected_cluster, return_grouped=False):
    import io
    import contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        evaluate_rules(selected_cluster if selected_cluster != "All Clusters" else None)
    result = output.getvalue()
    # Parse output into grouped violations by cluster
    grouped = {}
    current_cluster = None
    current_violation = []
    for line in result.splitlines():
        if line.startswith("########################################"):  # Cluster header
            continue
        elif line.startswith("Cluster: "):
            current_cluster = line.replace("Cluster: ", "").strip()
            grouped[current_cluster] = []
        elif line.startswith("------------------------------") or line.startswith("-------------------------------"):
            continue
        elif line.strip() == "":
            if current_violation and current_cluster:
                grouped[current_cluster].append("\n".join(current_violation))
                current_violation = []
        else:
            current_violation.append(line)
    if current_violation and current_cluster:
        grouped[current_cluster].append("\n".join(current_violation))
    if return_grouped:
        return grouped
    return result

def get_last_collection_time():
    try:
        with open("last_collection_time.txt", "r") as f:
            return float(f.read())
    except Exception:
        return 10  # Default to 10 seconds if not available

def timed_data_collection():
    import time
    start = time.time()
    collect_and_store_metrics_main()
    end = time.time()
    duration = end - start
    
    # Update Prometheus metrics
    LAST_COLLECTION_TIME.set(end)
    
    # Update VM and host counts
    try:
        clusters, hosts, vms = get_db_state()
        VM_COUNT.set(len(vms))
        HOST_COUNT.set(len(hosts))
    except Exception as e:
        print(f"Failed to update VM/Host counts: {e}")
    
    with open("last_collection_time.txt", "w") as f:
        f.write(str(duration))
    return duration

def run_collection_in_background():
    timed_data_collection()

if st.button("Refresh Data from vCenter"):
    last_duration = get_last_collection_time()  # Read from .txt
    progress = st.progress(0)
    # Start data collection in a background thread
    thread = threading.Thread(target=run_collection_in_background)
    thread.start()
    start = time.time()
    while thread.is_alive():
        elapsed = time.time() - start
        percent = min(1.0, elapsed / last_duration)
        progress.progress(int(percent * 100))
        time.sleep(0.1)
    progress.progress(100)
    st.success("Data refreshed from vCenter!")

# Add sidebar navigation
page = st.sidebar.radio("Navigation", ["Compliance Dashboard", "Exception Management", "Rule Management", "VM Rule Validator", "AI Config", "AI Optimizer"])

if page == "Compliance Dashboard":
    if 'violations' not in st.session_state:
        st.session_state['violations'] = None

    # Add UI filter for powered-on VMs
    show_only_powered_on = st.checkbox("Show only powered-on VMs (hide violations for powered-off VMs)", value=True)

    if st.button("Run Compliance Check"):
        # Use the new structured output
        structured_violations = evaluate_rules(selected_cluster if selected_cluster != "All Clusters" else None, return_structured=True)
        st.session_state['violations'] = structured_violations
        
        # Update Prometheus metrics with the new violations
        update_violation_metrics(structured_violations)
        
        if not structured_violations:
            st.success("✅ All VMs in this cluster are compliant! No violations found.")
            st.balloons()

    def display_single_violation(violation, cluster_name, idx, show_only_powered_on):
        """Display a single violation"""
        # Filter: Only show if at least one affected VM is powered on (if filter is enabled)
        if show_only_powered_on:
            # Get power status for affected VMs
            _, _, vms = get_db_state()
            powered_on = False
            for vm_name in violation['affected_vms']:
                for vm in vms.values():
                    if vm['name'] == vm_name and (vm.get('power_status') or '').lower() == 'poweredon':
                        powered_on = True
                        break
                if powered_on:
                    break
            if not powered_on:
                return
        
        expander_title = f"Alias: {violation['alias']} | Rule: {violation['type']}"
        with st.expander(expander_title, expanded=True):
            st.code(violation["violation_text"])
            st.write(f"**Rule Type:** {violation['type']}")
            # Extract alias for display
            if violation['alias'] is not None:
                display_alias = violation['alias']
            elif violation['affected_vms']:
                # Extract alias from first VM name for CockroachDB pattern
                first_vm = violation['affected_vms'][0]
                if first_vm.startswith('z-cockroach-'):
                    # Extract cluster name from z-cockroach-{cluster}-{node}
                    parts = first_vm.split('-')
                    if len(parts) >= 3:
                        display_alias = f"cockroach-{parts[2]}"  # cockroach-{cluster}
                    else:
                        display_alias = first_vm
                else:
                    display_alias = first_vm
            else:
                display_alias = 'unknown'
            st.write(f"**Alias:** {display_alias}")
            st.write(f"**Affected VMs:** {', '.join(violation['affected_vms'])}")

            # Create a unique key for this single violation
            import hashlib
            unique_key_data = f"{cluster_name}_{display_alias}_{','.join(violation['affected_vms'])}"
            unique_key_hash = hashlib.md5(unique_key_data.encode()).hexdigest()[:8]

            if st.button("Add Exception", key=f"add_exc_{unique_key_hash}"):
                import hashlib
                # Normalize fields for robust matching - extract alias for name-pattern rules
                if violation.get('alias') is not None:
                    alias = violation.get('alias').strip().lower()
                elif violation.get('affected_vms'):
                    # Extract alias from first VM name for CockroachDB pattern
                    first_vm = violation['affected_vms'][0]
                    if first_vm.startswith('z-cockroach-'):
                        # Extract cluster name from z-cockroach-{cluster}-{node}
                        parts = first_vm.split('-')
                        if len(parts) >= 3:
                            alias = f"cockroach-{parts[2]}".lower()  # cockroach-{cluster}
                        else:
                            alias = first_vm.lower()
                    else:
                        alias = first_vm.lower()
                else:
                    alias = 'unknown'
                cluster = (violation.get('cluster') or '').strip().lower()
                affected_vms = [vm.strip().lower() for vm in violation.get('affected_vms', [])]
                relevant = {
                    'rule_type': violation.get('type'),
                    'alias': alias,
                    'affected_vms': sorted(affected_vms),
                    'cluster': cluster
                }
                rule_hash = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
                exception_data = {
                    'rule_type': violation.get('type'),
                    'alias': alias,
                    'affected_vms': affected_vms,
                    'cluster': cluster,
                    'rule_hash': rule_hash,
                    'reason': None
                }
                try:
                    db = MetricsDB()
                    db.connect()
                    db.add_exception(exception_data)
                    db.close()
                    st.success(f"Exception added for Alias: {alias} | Rule: {violation.get('type')}. This violation will be ignored in future checks.")
                except Exception as e:
                    st.error(f"[ERROR] Failed to add exception: {e}")
            # For name-pattern-based rules, extract alias from VM name
            if violation['alias'] is not None:
                alias_display = violation['alias']
            elif violation['affected_vms']:
                # Extract alias from first VM name for CockroachDB pattern
                first_vm = violation['affected_vms'][0]
                if first_vm.startswith('z-cockroach-'):
                    # Extract cluster name from z-cockroach-{cluster}-{node}
                    parts = first_vm.split('-')
                    if len(parts) >= 3:
                        alias_display = f"cockroach-{parts[2]}"  # cockroach-{cluster}
                    else:
                        alias_display = first_vm
                else:
                    alias_display = first_vm
            else:
                alias_display = 'unknown'
            if st.button(f"Remediate/Fix for alias {alias_display}", key=f"remediate_fix_{unique_key_hash}"):
                token = st.session_state.get('remediation_token')
                if not token:
                    st.error("You must authenticate first. Please log in via the sidebar to obtain a valid token before attempting remediation.")
                else:
                    # Select playbook based on rule level
                    level = violation.get('level', 'host')
                    if level == 'storage':
                        playbook_name = 'e-vmotion-storage'
                    else:
                        playbook_name = 'e-vmotion-server'
                    # For name-pattern-based rules, extract alias from VM name
                    if violation['alias'] is not None:
                        alias = violation['alias']
                    elif violation['affected_vms']:
                        # Extract alias from first VM name for CockroachDB pattern
                        first_vm = violation['affected_vms'][0]
                        if first_vm.startswith('z-cockroach-'):
                            # Extract cluster name from z-cockroach-{cluster}-{node}
                            parts = first_vm.split('-')
                            if len(parts) >= 3:
                                alias = f"cockroach-{parts[2]}"  # cockroach-{cluster}
                            else:
                                alias = first_vm
                        else:
                            alias = first_vm
                    else:
                        alias = 'unknown'
                    success, msg = trigger_remediation_api(alias, violation['affected_vms'], token, playbook_name=playbook_name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

    def display_grouped_violations(grouped_violations, cluster_name, show_only_powered_on):
        """Display multiple violations for the same alias/rule as a grouped violation"""
        # Check if any VMs are powered on (if filter is enabled)
        if show_only_powered_on:
            _, _, vms = get_db_state()
            powered_on = False
            for violation in grouped_violations:
                for vm_name in violation['affected_vms']:
                    for vm in vms.values():
                        if vm['name'] == vm_name and (vm.get('power_status') or '').lower() == 'poweredon':
                            powered_on = True
                            break
                    if powered_on:
                        break
                if powered_on:
                    break
            if not powered_on:
                return
        
        # Extract alias for display
        first_violation = grouped_violations[0]
        if first_violation['alias'] is not None:
            display_alias = first_violation['alias']
        elif first_violation['affected_vms']:
            # Extract alias from first VM name for CockroachDB pattern
            first_vm = first_violation['affected_vms'][0]
            if first_vm.startswith('z-cockroach-'):
                # Extract cluster name from z-cockroach-{cluster}-{node}
                parts = first_vm.split('-')
                if len(parts) >= 3:
                    display_alias = f"cockroach-{parts[2]}"  # cockroach-{cluster}
                else:
                    display_alias = first_vm
            else:
                display_alias = first_vm
        else:
            display_alias = 'unknown'
        
        # Combine all affected VMs and create grouped violation text
        all_affected_vms = []
        all_pools = set()
        combined_violation_text = []
        
        for violation in grouped_violations:
            all_affected_vms.extend(violation['affected_vms'])
            # Extract pool information from violation text
            violation_lines = violation['violation_text'].split('\n')
            for line in violation_lines:
                if line.startswith('Pool: '):
                    pool_name = line.replace('Pool: ', '').strip()
                    all_pools.add(pool_name)
                    break
        
        # Create grouped violation text based on actual rule type
        rule_type = first_violation['type']
        if rule_type == 'dataset-affinity':
            combined_violation_text.append(f"Dataset Affinity Violation (Grouped)")
            combined_violation_text.append(f"Rule: VMs must be on datasets matching specific patterns")
        elif rule_type == 'dataset-anti-affinity':
            combined_violation_text.append(f"Dataset Anti-Affinity Violation (Grouped)")
            combined_violation_text.append(f"Rule: VMs must NOT be on the same dataset matching specific patterns")
        elif rule_type == 'pool-anti-affinity':
            combined_violation_text.append(f"Pool Anti-Affinity Violation (Grouped)")
            combined_violation_text.append(f"Rule: VMs must NOT be on the same ZFS pool matching specific patterns")
        elif rule_type == 'affinity':
            combined_violation_text.append(f"Host Affinity Violation (Grouped)")
            combined_violation_text.append(f"Rule: VMs must be on the same host")
        elif rule_type == 'anti-affinity':
            combined_violation_text.append(f"Host Anti-Affinity Violation (Grouped)")
            combined_violation_text.append(f"Rule: VMs must NOT be on the same host")
        else:
            combined_violation_text.append(f"{rule_type.capitalize()} Violation (Grouped)")
            combined_violation_text.append(f"Rule: {rule_type} rule violation")
        
        combined_violation_text.append(f"Alias: {display_alias}")
        combined_violation_text.append(f"Affected VMs:")
        for vm in sorted(set(all_affected_vms)):
            combined_violation_text.append(f"  - {vm}")
        
        # Add rule-specific information
        if rule_type in ['dataset-affinity', 'dataset-anti-affinity']:
            # Extract dataset information from violation text
            all_datasets = set()
            for violation in grouped_violations:
                violation_lines = violation['violation_text'].split('\n')
                for line in violation_lines:
                    if line.startswith('Dataset: '):
                        dataset_name = line.replace('Dataset: ', '').strip()
                        all_datasets.add(dataset_name)
                        break
            if all_datasets:
                combined_violation_text.append(f"Affected Datasets: {', '.join(sorted(all_datasets))}")
        
        if rule_type == 'pool-anti-affinity':
            combined_violation_text.append(f"Affected Pools: {', '.join(sorted(all_pools))}")
        
        combined_violation_text.append("Suggestions:")
        if rule_type in ['dataset-affinity', 'dataset-anti-affinity']:
            combined_violation_text.append("  - Move VMs to appropriate datasets")
        elif rule_type == 'pool-anti-affinity':
            combined_violation_text.append("  - Move VMs to distribute them across different pools")
        elif rule_type in ['affinity', 'anti-affinity']:
            combined_violation_text.append("  - Move VMs to appropriate hosts")
        else:
            combined_violation_text.append("  - Review and fix rule violation")
        
        # Create a unique key for this grouped violation set
        import hashlib
        unique_key_data = f"{cluster_name}_{display_alias}_{','.join(sorted(set(all_affected_vms)))}"
        unique_key_hash = hashlib.md5(unique_key_data.encode()).hexdigest()[:8]
        
        expander_title = f"Alias: {display_alias} | Rule: {first_violation['type']} (Grouped - {len(grouped_violations)} violations)"
        with st.expander(expander_title, expanded=True):
            st.code('\n'.join(combined_violation_text))
            st.write(f"**Rule Type:** {first_violation['type']}")
            st.write(f"**Alias:** {display_alias}")
            st.write(f"**Affected VMs:** {', '.join(sorted(set(all_affected_vms)))}")
            st.write(f"**Affected Pools:** {', '.join(sorted(all_pools))}")
            st.write(f"**Number of Violations:** {len(grouped_violations)}")

            if st.button("Add Exception", key=f"add_exc_grouped_{unique_key_hash}"):
                import hashlib
                # Use the same alias extraction logic
                if first_violation.get('alias') is not None:
                    alias = first_violation.get('alias').strip().lower()
                elif first_violation.get('affected_vms'):
                    first_vm = first_violation['affected_vms'][0]
                    if first_vm.startswith('z-cockroach-'):
                        parts = first_vm.split('-')
                        if len(parts) >= 3:
                            alias = f"cockroach-{parts[2]}".lower()
                        else:
                            alias = first_vm.lower()
                    else:
                        alias = first_vm.lower()
                else:
                    alias = 'unknown'
                
                cluster = (first_violation.get('cluster') or '').strip().lower()
                affected_vms = [vm.strip().lower() for vm in set(all_affected_vms)]
                relevant = {
                    'rule_type': first_violation.get('type'),
                    'alias': alias,
                    'affected_vms': sorted(affected_vms),
                    'cluster': cluster
                }
                rule_hash = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
                exception_data = {
                    'rule_type': first_violation.get('type'),
                    'alias': alias,
                    'affected_vms': affected_vms,
                    'cluster': cluster,
                    'rule_hash': rule_hash,
                    'reason': None
                }
                try:
                    db = MetricsDB()
                    db.connect()
                    db.add_exception(exception_data)
                    db.close()
                    st.success(f"Exception added for Alias: {alias} | Rule: {first_violation.get('type')}. This violation will be ignored in future checks.")
                except Exception as e:
                    st.error(f"[ERROR] Failed to add exception: {e}")
            
            if st.button(f"Remediate/Fix for alias {display_alias} (Grouped)", key=f"remediate_fix_grouped_{unique_key_hash}"):
                token = st.session_state.get('remediation_token')
                if not token:
                    st.error("You must authenticate first. Please log in via the sidebar to obtain a valid token before attempting remediation.")
                else:
                    # Select playbook based on rule level
                    level = first_violation.get('level', 'host')
                    if level == 'storage':
                        playbook_name = 'e-vmotion-storage'
                    else:
                        playbook_name = 'e-vmotion-server'
                    
                    # Use the same alias extraction logic
                    if first_violation['alias'] is not None:
                        alias = first_violation['alias']
                    elif first_violation['affected_vms']:
                        first_vm = first_violation['affected_vms'][0]
                        if first_vm.startswith('z-cockroach-'):
                            parts = first_vm.split('-')
                            if len(parts) >= 3:
                                alias = f"cockroach-{parts[2]}"
                            else:
                                alias = first_vm
                        else:
                            alias = first_vm
                    else:
                        alias = 'unknown'
                    
                    # Pass all affected VMs for grouped remediation
                    success, msg = trigger_remediation_api(alias, list(set(all_affected_vms)), token, playbook_name=playbook_name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

    # Only display violations if they exist in session state
    if st.session_state['violations']:
        # Update Prometheus metrics with current violations
        update_violation_metrics(st.session_state['violations'])
        
        from collections import defaultdict
        
        # Group violations by cluster, then by (alias, rule_type) to combine related violations
        cluster_grouped = defaultdict(list)
        for v in st.session_state['violations']:
            cluster_grouped[v["cluster"]].append(v)
        
        for cluster_name, violations in cluster_grouped.items():
            st.markdown(f"### Cluster: {cluster_name}")
            
            # Group violations by alias and rule type
            alias_rule_grouped = defaultdict(list)
            for violation in violations:
                # Extract alias for grouping
                if violation['alias'] is not None:
                    alias = violation['alias']
                elif violation['affected_vms']:
                    # Extract alias from first VM name for CockroachDB pattern
                    first_vm = violation['affected_vms'][0]
                    if first_vm.startswith('z-cockroach-'):
                        # Extract cluster name from z-cockroach-{cluster}-{node}
                        parts = first_vm.split('-')
                        if len(parts) >= 3:
                            alias = f"cockroach-{parts[2]}"  # cockroach-{cluster}
                        else:
                            alias = first_vm
                    else:
                        alias = first_vm
                else:
                    alias = 'unknown'
                
                # Group by (alias, rule_type)
                group_key = (alias, violation['type'])
                alias_rule_grouped[group_key].append(violation)
            
            # Display grouped violations
            for (alias, rule_type), grouped_violations in alias_rule_grouped.items():
                if len(grouped_violations) == 1:
                    # Single violation - display normally
                    violation = grouped_violations[0]
                    idx = violations.index(violation)
                    display_single_violation(violation, cluster_name, idx, show_only_powered_on)
                else:
                    # Multiple violations for same alias/rule - display as grouped
                    display_grouped_violations(grouped_violations, cluster_name, show_only_powered_on)
    else:
        st.info("Click 'Run Compliance Check' to evaluate compliance for the selected cluster.")

elif page == "Exception Management":
    st.title("Exception Management")
    db = MetricsDB()
    db.connect()
    exceptions = db.get_exceptions()
    db.close()
    if not exceptions:
        st.info("No exceptions found.")
    else:
        import pandas as pd
        df = pd.DataFrame(exceptions)
        # Show a table with a remove button for each exception
        for i, exc in df.iterrows():
            with st.expander(f"Alias: {bold_unicode(exc['alias'])} | Rule: {exc['rule_type']} | Cluster: {exc['cluster']}"):
                st.write(f"**Affected VMs:** {', '.join(exc['affected_vms']) if isinstance(exc['affected_vms'], list) else exc['affected_vms']}")
                st.write(f"**Created At:** {exc['created_at']}")
                st.write(f"**Reason:** {exc['reason']}")
                if st.button(f"Remove Exception {exc['id']}", key=f"remove_exc_{exc['id']}"):
                    db = MetricsDB()
                    db.connect()
                    if db.conn:
                        db.conn.cursor().execute('DELETE FROM exceptions WHERE id = %s', (exc['id'],))
                        db.conn.commit()
                    db.close()
                    st.success("Exception removed. Please refresh the page.")

elif page == "Rule Management":
    st.title("Rule Management")
    import json
    import os
    rules_path = os.path.join(os.path.dirname(__file__), 'rules', 'rules.json')
    # Load rules
    with open(rules_path, 'r') as f:
        rules = json.load(f)
    # Show rules in expandable list
    for idx, rule in enumerate(rules):
        with st.expander(f"Rule {idx+1}: {rule.get('type', '').capitalize()} | {rule.get('role', rule.get('name_pattern', ''))}"):
            st.json(rule)
            if st.button(f"Delete Rule {idx+1}", key=f"delete_rule_{idx}"):
                rules.pop(idx)
                with open(rules_path, 'w') as f:
                    json.dump(rules, f, indent=2)
                st.success("Rule deleted. Please refresh the page.")
                st.stop()
    st.markdown("---")
    st.header("Add New Rule")
    # Simple form for adding a new rule
    with st.form("add_rule_form"):
        rule_type = st.selectbox("Type", ["affinity", "anti-affinity", "dataset-affinity", "dataset-anti-affinity", "pool-anti-affinity"])
        level = st.selectbox("Level", ["host", "storage", "cluster", "(none)"])
        role = st.text_input("Role (comma-separated for multiple, leave blank if using name_pattern)")
        name_pattern = st.text_input("Name Pattern (optional)")
        dataset_pattern = st.text_input("Dataset Pattern (comma-separated, for dataset-affinity/dataset-anti-affinity only)")
        pool_pattern = st.text_input("Pool Pattern (comma-separated, for pool-anti-affinity only)")
        submitted = st.form_submit_button("Add Rule")
        if submitted:
            new_rule: Dict[str, Any] = {"type": rule_type}
            if level != "(none)":
                new_rule["level"] = level
            if role:
                roles = [r.strip() for r in role.split(",") if r.strip()]
                new_rule["role"] = roles if len(roles) > 1 else roles[0]
            if name_pattern:
                new_rule["name_pattern"] = name_pattern
            if dataset_pattern:
                patterns = [p.strip() for p in dataset_pattern.split(",") if p.strip()]
                new_rule["dataset_pattern"] = patterns
            if pool_pattern:
                patterns = [p.strip() for p in pool_pattern.split(",") if p.strip()]
                new_rule["pool_pattern"] = patterns
            rules.append(new_rule)
            with open(rules_path, 'w') as f:
                json.dump(rules, f, indent=2)
            st.success("New rule added. Please refresh the page.")
            st.stop()

elif page == "VM Rule Validator":
    st.title("VM Rule Validator")
    st.write("Enter a VM name, select a host and dataset, and see which rules would apply and if the placement would be compliant.")
    rules = load_rules()
    clusters, hosts, vms = get_db_state()
    host_names = [h['name'] for h in hosts.values()]
    dataset_name_set = set(v['dataset_name'] for v in vms.values() if v['dataset_name'])
    dataset_names = sorted(dataset_name_set)

    with st.form("vm_rule_validator_form"):
        vm_name = st.text_input("VM Name", "z-example-alias-LB1")
        host = st.selectbox("Host", host_names)
        dataset = st.selectbox("Dataset", dataset_names)
        submitted = st.form_submit_button("Validate")

    if submitted:
        alias, role = parse_alias_and_role(vm_name)
        st.write(f"**Alias:** {alias}")
        st.write(f"**Role:** {role}")
        st.write(f"**Host:** {host}")
        st.write(f"**Dataset:** {dataset}")
        # Simulate VM dict
        vm = {'name': vm_name, 'host_id': None, 'dataset_name': dataset}
        # Find host_id and cluster
        host_id = None
        cluster_id = None
        for hid, h in hosts.items():
            if h['name'] == host:
                host_id = hid
                cluster_id = h['cluster_id']
                break
        if not host_id:
            st.error("Host not found in DB.")
        else:
            cluster_name = clusters[cluster_id]
            st.write(f"**Cluster:** {cluster_name}")
            # Check which rules would apply
            applicable_rules = []
            violations = []
            for rule in rules:
                applies = False
                violation = None
                # Affinity/anti-affinity
                if rule['type'] in ['affinity', 'anti-affinity']:
                    if 'role' in rule:
                        rule_roles = [rule['role']] if isinstance(rule['role'], str) else rule['role']
                        if role in [r.upper() for r in rule_roles]:
                            applies = True
                            # Simulate: for anti-affinity, if another VM with same alias+role is on the same host, would violate
                            if rule['type'] == 'anti-affinity' and rule.get('level') == 'host':
                                for v in vms.values():
                                    a2, r2 = parse_alias_and_role(v['name'])
                                    if a2 == alias and r2 == role and v['host_id'] == host_id:
                                        violation = f"Would violate anti-affinity: another {role} VM with alias {alias} is already on host {host}."
                            if rule['type'] == 'affinity' and rule.get('level') == 'host':
                                # If other VMs with same alias+role are on different hosts, would violate
                                other_hosts = set(v['host_id'] for v in vms.values() if parse_alias_and_role(v['name']) == (alias, role) and v['host_id'] is not None)
                                if other_hosts and host_id not in other_hosts:
                                    violation = f"Would violate affinity: other {role} VMs with alias {alias} are on different hosts."
                # Dataset affinity/anti-affinity
                if rule['type'] in ['dataset-affinity', 'dataset-anti-affinity']:
                    patterns = rule.get('dataset_pattern', [])
                    # Role-based
                    if 'role' in rule:
                        rule_roles = [rule['role']] if isinstance(rule['role'], str) else rule['role']
                        if role in [r.upper() for r in rule_roles]:
                            applies = True
                            if rule['type'] == 'dataset-affinity':
                                if not any(pat in dataset for pat in patterns):
                                    violation = f"Would violate dataset-affinity: dataset {dataset} does not match {patterns}."
                            if rule['type'] == 'dataset-anti-affinity':
                                # If another matching VM is on the same dataset, would violate
                                for v in vms.values():
                                    a2, r2 = parse_alias_and_role(v['name'])
                                    if a2 == alias and r2 == role and v['dataset_name'] == dataset:
                                        violation = f"Would violate dataset-anti-affinity: another {role} VM with alias {alias} is already on dataset {dataset}."
                    # Name-pattern-based
                    if 'name_pattern' in rule:
                        if rule['name_pattern'] in vm_name:
                            applies = True
                            if rule['type'] == 'dataset-affinity':
                                if not any(pat in dataset for pat in patterns):
                                    violation = f"Would violate dataset-affinity: dataset {dataset} does not match {patterns}."
                            if rule['type'] == 'dataset-anti-affinity':
                                for v in vms.values():
                                    if rule['name_pattern'] in v['name'] and v['dataset_name'] == dataset:
                                        violation = f"Would violate dataset-anti-affinity: another VM with name pattern {rule['name_pattern']} is already on dataset {dataset}."
                
                # Pool anti-affinity
                if rule['type'] == 'pool-anti-affinity':
                    from vcenter_drs.rules.rules_engine import extract_pool_from_dataset
                    pool_patterns = rule.get('pool_pattern', [])
                    current_pool = extract_pool_from_dataset(dataset)
                    
                    # Role-based
                    if 'role' in rule:
                        rule_roles = [rule['role']] if isinstance(rule['role'], str) else rule['role']
                        if role in [r.upper() for r in rule_roles]:
                            applies = True
                            if current_pool and any(pat.lower() in current_pool.lower() for pat in pool_patterns):
                                # Check if another matching VM is on the same pool
                                for v in vms.values():
                                    a2, r2 = parse_alias_and_role(v['name'])
                                    v_pool = extract_pool_from_dataset(v['dataset_name'])
                                    if a2 == alias and r2 == role and v_pool == current_pool:
                                        violation = f"Would violate pool-anti-affinity: another {role} VM with alias {alias} is already on pool {current_pool}."
                    # Name-pattern-based
                    if 'name_pattern' in rule:
                        if rule['name_pattern'] in vm_name:
                            applies = True
                            if current_pool and any(pat.lower() in current_pool.lower() for pat in pool_patterns):
                                # Check if another matching VM is on the same pool
                                for v in vms.values():
                                    v_pool = extract_pool_from_dataset(v['dataset_name'])
                                    if rule['name_pattern'] in v['name'] and v_pool == current_pool:
                                        violation = f"Would violate pool-anti-affinity: another VM with name pattern {rule['name_pattern']} is already on pool {current_pool}."
                if applies:
                    applicable_rules.append(rule)
                    if violation:
                        violations.append((rule, violation))
            st.markdown("---")
            st.subheader("Applicable Rules:")
            if not applicable_rules:
                st.info("No rules would apply to this VM.")
            else:
                for rule in applicable_rules:
                    st.json(rule)
            st.markdown("---")
            st.subheader("Potential Violations:")
            if not violations:
                st.success("This VM would be compliant with all applicable rules for the selected placement.")
            else:
                for rule, vtext in violations:
                    st.error(vtext) 

elif page == "AI Config":
    st.title("AI Optimizer Configuration")
    st.write("Configure AI optimization parameters and model settings.")
    
    # Import AI modules
    try:
        from ai_optimizer.config import AIConfig
        from ai_optimizer.data_collector import PrometheusDataCollector
        from ai_optimizer.optimization_engine import OptimizationEngine
        
        # Initialize AI components
        ai_config = AIConfig()
        data_collector = PrometheusDataCollector(ai_config)
        optimization_engine = OptimizationEngine(ai_config, data_collector)
        
        st.success("✅ AI Optimizer modules loaded successfully")
        
        # Configuration sections
        st.header("Prometheus Configuration")
        col1, col2 = st.columns(2)
        with col1:
            prometheus_url = st.text_input("Prometheus URL", value=ai_config.prometheus.url)
            prometheus_port = st.number_input("Prometheus Port", value=ai_config.prometheus.port, min_value=1, max_value=65535)
        with col2:
            timeout = st.number_input("Timeout (seconds)", value=ai_config.prometheus.timeout, min_value=5, max_value=300)
            retry_attempts = st.number_input("Retry Attempts", value=ai_config.prometheus.retry_attempts, min_value=1, max_value=10)
        
        st.header("Analysis Windows")
        col1, col2, col3 = st.columns(3)
        with col1:
            cpu_trend_hours = st.number_input("CPU Trend (hours)", value=ai_config.analysis.cpu_trend_hours, min_value=1, max_value=24)
            storage_trend_days = st.number_input("Storage Trend (days)", value=ai_config.analysis.storage_trend_days, min_value=1, max_value=7)
        with col2:
            ram_trend_hours = st.number_input("RAM Trend (hours)", value=ai_config.analysis.ram_trend_hours, min_value=1, max_value=24)
            io_trend_days = st.number_input("I/O Trend (days)", value=ai_config.analysis.io_trend_days, min_value=1, max_value=7)
        with col3:
            ready_time_window = st.number_input("Ready Time Window (hours)", value=ai_config.analysis.ready_time_window, min_value=1, max_value=24)
        
        st.header("Optimization Parameters")
        col1, col2 = st.columns(2)
        with col1:
            ideal_host_min = st.slider("Ideal Host Usage Min (%)", value=int(ai_config.optimization.ideal_host_usage_min * 100), min_value=10, max_value=90)
            ideal_host_max = st.slider("Ideal Host Usage Max (%)", value=int(ai_config.optimization.ideal_host_usage_max * 100), min_value=10, max_value=90)
            max_recommendations = st.number_input("Max Recommendations", value=ai_config.optimization.max_recommendations, min_value=1, max_value=20)
        with col2:
            cpu_weight = st.slider("CPU Priority Weight", value=float(ai_config.optimization.cpu_priority_weight), min_value=0.1, max_value=2.0, step=0.1)
            ram_weight = st.slider("RAM Priority Weight", value=float(ai_config.optimization.ram_priority_weight), min_value=0.1, max_value=2.0, step=0.1)
            ready_time_weight = st.slider("Ready Time Priority Weight", value=float(ai_config.optimization.ready_time_priority_weight), min_value=0.1, max_value=2.0, step=0.1)
            io_weight = st.slider("I/O Priority Weight", value=float(ai_config.optimization.io_priority_weight), min_value=0.1, max_value=2.0, step=0.1)
        
        st.header("ML Model Settings")
        col1, col2 = st.columns(2)
        with col1:
            training_episodes = st.number_input("Training Episodes", value=ai_config.ml.training_episodes, min_value=100, max_value=10000)
            learning_rate = st.number_input("Learning Rate", value=ai_config.ml.learning_rate, min_value=0.0001, max_value=0.1, format="%.4f")
        with col2:
            batch_size = st.number_input("Batch Size", value=ai_config.ml.batch_size, min_value=8, max_value=128)
            exploration_rate = st.slider("Exploration Rate", value=ai_config.ml.exploration_rate, min_value=0.01, max_value=0.5, step=0.01)
        
        # Save configuration button
        if st.button("Save Configuration"):
            # Update configuration (this would need to be implemented to persist changes)
            st.success("Configuration saved! (Note: This is a placeholder - actual persistence not yet implemented)")
        
        # System status
        st.header("System Status")
        col1, col2 = st.columns(2)
        with col1:
            # Test Prometheus connection
            if st.button("Test Prometheus Connection"):
                with st.spinner("Testing connection..."):
                    time.sleep(1)  # Simulate connection test
                    if data_collector.test_connection():
                        st.success("✅ Prometheus connection successful")
                        st.info(f"Connected to: {data_collector.current_url}")
                    else:
                        st.error("❌ Both Prometheus servers failed")
                        st.warning("Using simulated data for recommendations")
        
        with col2:
            # Show optimization summary
            if st.button("Show Optimization Summary"):
                with st.spinner("Generating summary..."):
                    time.sleep(0.5)
                    summary = optimization_engine.get_optimization_summary()
                    st.json(summary)
                    
                    # Show additional status info
                    st.info(f"Models trained: {'✅ Yes' if summary['models_trained'] else '❌ No'}")
                    st.info(f"Prometheus connected: {'✅ Yes' if summary['prometheus_connected'] else '❌ No'}")
                    st.info(f"Current server: {summary['current_prometheus_server']}")
        
    except ImportError as e:
        st.error(f"❌ Failed to import AI Optimizer modules: {e}")
        st.info("Make sure the ai_optimizer module is properly installed and accessible.")

elif page == "AI Optimizer":
    st.title("AI VM Placement Optimizer")
    st.write("Get AI-powered VM placement recommendations based on performance metrics and optimization criteria.")
    
    try:
        from ai_optimizer.config import AIConfig
        from ai_optimizer.data_collector import PrometheusDataCollector
        from ai_optimizer.optimization_engine import OptimizationEngine
        
        # Initialize AI components
        ai_config = AIConfig()
        data_collector = PrometheusDataCollector(ai_config)
        optimization_engine = OptimizationEngine(ai_config, data_collector)
        
        st.success("✅ AI Optimizer loaded successfully")
        
        # Load trained AI models
        import joblib
        import json
        import numpy as np
        
        models_dir = "ai_optimizer/models"
        trained_models = {}
        model_scalers = {}
        model_performance = {}
        
        # Load model performance data
        try:
            results_path = os.path.join(models_dir, "training_results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    model_performance = json.load(f)
                st.success("✅ Trained AI models loaded successfully")
                
                # Display model performance
                with st.expander("📊 AI Model Performance", expanded=False):
                    for model_name, metrics in model_performance.items():
                        st.write(f"**{model_name.upper()}:**")
                        st.write(f"  - R² Score: {metrics['r2']:.4f}")
                        st.write(f"  - Mean Squared Error: {metrics['mse']:.4f}")
                        st.write(f"  - Mean Absolute Error: {metrics['mae']:.4f}")
            else:
                st.warning("⚠️ No trained models found. Models will be trained on first use.")
        except Exception as e:
            st.warning(f"⚠️ Could not load model performance data: {e}")
        
        # Load trained models
        try:
            model_types = ['random_forest', 'gradient_boosting']
            for model_type in model_types:
                model_path = os.path.join(models_dir, f"{model_type}_model.pkl")
                scaler_path = os.path.join(models_dir, f"{model_type}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    trained_models[model_type] = joblib.load(model_path)
                    model_scalers[model_type] = joblib.load(scaler_path)
            
            if trained_models:
                st.success(f"✅ Loaded {len(trained_models)} trained AI models")
            else:
                st.warning("⚠️ No trained models available")
        except Exception as e:
            st.warning(f"⚠️ Could not load trained models: {e}")
        
        # Get available VMs from database and real hosts from Prometheus
        clusters, hosts, vms = get_db_state()
        vm_names = [vm['name'] for vm in vms.values()]
        cluster_names = list(clusters.values())
        
        # Get real host names from Prometheus data
        try:
            # Query Prometheus for available hosts
            import requests
            response = requests.get("http://10.65.32.4:9090/api/v1/query?query=vmware_host_cpu_usage_average", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    # Extract unique host names from the results
                    real_hosts = list(set([result['metric']['host_name'] for result in data['data']['result']]))
                    st.info(f"📊 Found {len(real_hosts)} real hosts in Prometheus data")
                else:
                    real_hosts = []
            else:
                real_hosts = []
        except Exception as e:
            st.warning(f"⚠️ Could not fetch real hosts from Prometheus: {e}")
            real_hosts = []
        
        st.header("VM Placement Recommendations")
        
        # VM selection
        col1, col2 = st.columns(2)
        with col1:
            selected_vm = st.selectbox("Select VM for placement", vm_names)
        with col2:
            selected_cluster = st.selectbox("Cluster Filter (Optional)", ["All Clusters"] + cluster_names)
        
        # Number of recommendations
        num_recommendations = st.slider("Number of Recommendations", value=5, min_value=1, max_value=10)
        
        # Generate recommendations button
        if st.button("Generate Placement Recommendations"):
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Stage 1: Collecting VM metrics
                status_text.text("Stage 1/4: Collecting VM performance metrics...")
                progress_bar.progress(25)
                time.sleep(0.5)  # Simulate processing time
                
                # Stage 2: Analyzing available hosts
                status_text.text("Stage 2/4: Analyzing available hosts and clusters...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                # Stage 3: Running optimization algorithms
                status_text.text("Stage 3/4: Running AI optimization algorithms...")
                progress_bar.progress(75)
                time.sleep(0.5)
                
                # Stage 4: Generating recommendations
                status_text.text("Stage 4/4: Generating placement recommendations...")
                progress_bar.progress(90)
                time.sleep(0.5)
                
                # Use real hosts if available, otherwise fall back to dummy hosts
                if real_hosts:
                    # Use a subset of real hosts for recommendations
                    host_subset = real_hosts[:min(5, len(real_hosts))]  # Limit to 5 hosts
                    cluster_filter = host_subset
                    st.info(f"🎯 Using {len(host_subset)} real hosts from Prometheus data")
                else:
                    # Fall back to dummy hosts
                    cluster_filter = ["host-01.zengenti.com", "host-02.zengenti.com"]
                    st.warning("⚠️ Using dummy host data (real hosts unavailable)")
                
                recommendations = optimization_engine.generate_placement_recommendations(
                    selected_vm, 
                    cluster_filter, 
                    num_recommendations
                )
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                if recommendations:
                    st.success(f"✅ Generated {len(recommendations)} placement recommendations for {selected_vm}")
                    
                    # Show analysis summary
                    st.info(f"📊 Analysis Summary:")
                    st.info(f"  • Analyzed {len(recommendations)} potential host placements")
                    st.info(f"  • Best score: {max(r['score'] for r in recommendations):.3f}")
                    st.info(f"  • Average score: {sum(r['score'] for r in recommendations) / len(recommendations):.3f}")
                    
                    # Display recommendations
                    for i, rec in enumerate(recommendations):
                        with st.expander(f"#{rec['rank']} - {rec['host_name']} (Score: {rec['score']:.3f})", expanded=(i==0)):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Current Host Metrics")
                                st.metric("CPU Usage", f"{rec['current_metrics']['cpu_usage']:.1%}")
                                st.metric("RAM Usage", f"{rec['current_metrics']['ram_usage']:.1%}")
                                st.metric("Disk I/O Usage", f"{rec['current_metrics']['io_usage']:.1%}")
                                st.metric("VM Count", rec['current_metrics']['vm_count'])
                            
                            with col2:
                                st.subheader("Projected Metrics (After Placement)")
                                st.metric("CPU Usage", f"{rec['projected_metrics']['cpu_usage']:.1%}")
                                st.metric("RAM Usage", f"{rec['projected_metrics']['ram_usage']:.1%}")
                                st.metric("Disk I/O Usage", f"{rec['projected_metrics'].get('io_usage', 0):.1%}")
                                st.metric("VM Count", rec['projected_metrics']['vm_count'])
                            
                            st.subheader("VM Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("CPU Usage", f"{rec['vm_metrics']['cpu_usage']:.1%}")
                            with col2:
                                st.metric("RAM Usage", f"{rec['vm_metrics']['ram_usage']:.1%}")
                            with col3:
                                # Display VM ready time as a performance indicator (lower is better)
                                ready_time = rec['vm_metrics']['ready_time']
                                if ready_time <= 0.1:
                                    st.metric("Ready Time", "Excellent", delta="Optimal")
                                elif ready_time <= 0.5:
                                    st.metric("Ready Time", "Good", delta="Acceptable")
                                elif ready_time <= 0.8:
                                    st.metric("Ready Time", "Moderate", delta="Consider alternatives")
                                else:
                                    st.metric("Ready Time", "Poor", delta="Avoid placement")
                            with col4:
                                st.metric("Disk I/O Usage", f"{rec['vm_metrics']['io_usage']:.1%}")
                            
                            st.subheader("AI Reasoning")
                            st.info(rec['reasoning'])
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Apply to {rec['host_name']}", key=f"apply_{i}"):
                                    st.success(f"Placement applied to {rec['host_name']} (Note: This is a placeholder)")
                            with col2:
                                if st.button(f"View Details", key=f"details_{i}"):
                                    st.json(rec)
                else:
                    st.warning("No suitable placement recommendations found. Check resource constraints and cluster availability.")
                    
            except Exception as e:
                st.error(f"❌ Failed to generate recommendations: {e}")
                st.info("This might be due to insufficient data, connection issues, or model training requirements.")
                
                # Show debug information
                with st.expander("🔍 Debug Information"):
                    st.code(f"""
Error: {e}
VM: {selected_vm}
Cluster Filter: {cluster_filter}
Number of Recommendations: {num_recommendations}
                    """)
        
        # AI Model Predictions section
        if trained_models:
            st.header("🤖 AI Model Predictions")
            st.write("Get predictions from trained AI models for VM placement optimization.")
            
            # AI prediction section
            ai_selected_vm = st.selectbox("Select VM for AI prediction", ["(Select a VM)"] + vm_names, key="ai_vm")
            
            # --- PATCH START: Prometheus cluster-aware host filtering ---
            # Build {host_name: cluster_name} mapping from Prometheus
            prometheus_host_clusters = {}
            try:
                response = requests.get("http://10.65.32.4:9090/api/v1/query?query=vmware_host_cpu_usage_average", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success' and data['data']['result']:
                        for result in data['data']['result']:
                            host_name = result['metric'].get('host_name', '')
                            cluster_name = result['metric'].get('cluster_name', '')
                            if host_name:
                                prometheus_host_clusters[host_name] = cluster_name
            except Exception as e:
                st.warning(f"⚠️ Could not fetch host cluster mapping from Prometheus: {e}")
            # --- PATCH END ---
            
            if ai_selected_vm and ai_selected_vm != "(Select a VM)":
                # Get VM's cluster
                vm_cluster = None
                selected_vm_data = None
                for vm_id, vm_data in vms.items():
                    if vm_data['name'] == ai_selected_vm:
                        vm_cluster = vm_data.get('cluster', '')
                        selected_vm_data = vm_data
                        break
                
                # --- PATCH: Use Prometheus cluster mapping for host selection ---
                available_hosts = []
                if vm_cluster:
                    # Find hosts in Prometheus with matching cluster
                    available_hosts = [host for host, cluster in prometheus_host_clusters.items() if cluster == vm_cluster]
                    if available_hosts:
                        st.info(f"🎯 Found {len(available_hosts)} hosts in VM's cluster ({vm_cluster}): {', '.join(available_hosts)}")
                    else:
                        st.warning(f"⚠️ No hosts from cluster '{vm_cluster}' found in Prometheus data. Using all available hosts.")
                        available_hosts = list(prometheus_host_clusters.keys())
                else:
                    st.warning("⚠️ Could not determine VM's cluster. Using all available hosts.")
                    available_hosts = list(prometheus_host_clusters.keys())
                # Fallback if no real hosts available
                if not available_hosts:
                    available_hosts = ["host-01.zengenti.com", "host-02.zengenti.com"]
                    st.warning("⚠️ No real hosts available. Using dummy hosts for demonstration.")
                # Show selected hosts (read-only for now)
                st.write(f"**Hosts to analyze:** {', '.join(available_hosts)}")
                st.info(f"📊 Will analyze {len(available_hosts)} hosts from VM's cluster for optimal placement")
            else:
                st.info("Please select a VM to see cluster and host analysis.")
            
            if st.button("Get AI Predictions", key="ai_predict"):
                if not available_hosts:
                    st.warning("No hosts available for analysis.")
                else:
                    # Create progress indicators
                    ai_progress = st.progress(0)
                    ai_status = st.empty()
                    
                    try:
                        # Stage 1: Collect VM metrics
                        ai_status.text("Stage 1/3: Collecting VM metrics...")
                        ai_progress.progress(33)
                        time.sleep(0.5)
                        
                        vm_metrics = optimization_engine.get_vm_metrics(ai_selected_vm)
                        
                        # Stage 2: Collect host metrics
                        ai_status.text("Stage 2/3: Collecting host metrics...")
                        ai_progress.progress(66)
                        time.sleep(0.5)
                        
                        host_predictions = []
                        
                        for host_name in available_hosts:
                            try:
                                host_metrics = data_collector.get_host_metrics(host_name)
                                
                                if host_metrics:
                                    # Prepare features for AI models
                                    projected_cpu_usage = min(1.0, host_metrics.get('cpu_usage', 0.0) + vm_metrics.get('cpu_usage', 0.0))
                                    projected_ram_usage = min(1.0, host_metrics.get('ram_usage', 0.0) + vm_metrics.get('ram_usage', 0.0))
                                    projected_io_usage = min(1.0, host_metrics.get('io_usage', 0.0) + vm_metrics.get('io_usage', 0.0))
                                    projected_vm_count = host_metrics.get('vm_count', 0) + 1
                                    
                                    # Create feature vector
                                    feature_vector = [
                                        vm_metrics.get('cpu_usage', 0.0),
                                        vm_metrics.get('ram_usage', 0.0),
                                        vm_metrics.get('ready_time', 0.0),
                                        vm_metrics.get('io_usage', 0.0),
                                        vm_metrics.get('cpu_mhz', 0.0),
                                        vm_metrics.get('ram_mb', 0.0),
                                        host_metrics.get('cpu_usage', 0.0),
                                        host_metrics.get('ram_usage', 0.0),
                                        host_metrics.get('io_usage', 0.0),
                                        host_metrics.get('ready_time', 0.0),
                                        host_metrics.get('vm_count', 0),
                                        host_metrics.get('cpu_max_mhz', 40000),
                                        host_metrics.get('ram_max_mb', 65536),
                                        projected_cpu_usage,
                                        projected_ram_usage,
                                        projected_io_usage,
                                        projected_vm_count
                                    ]
                                    
                                    # Get predictions from each model
                                    predictions = {}
                                    for model_name, model in trained_models.items():
                                        if model_name in model_scalers:
                                            scaler = model_scalers[model_name]
                                            features_scaled = scaler.transform([feature_vector])
                                            prediction = model.predict(features_scaled)[0]
                                            predictions[model_name] = max(0.0, min(1.0, prediction))
                                    
                                    # Calculate ensemble prediction
                                    if len(predictions) > 1:
                                        ensemble_score = sum(predictions.values()) / len(predictions)
                                        predictions['ensemble'] = ensemble_score
                                    
                                    host_predictions.append({
                                        'host_name': host_name,
                                        'predictions': predictions,
                                        'host_metrics': host_metrics,
                                        'vm_metrics': vm_metrics,
                                        'projected_metrics': {
                                            'cpu_usage': projected_cpu_usage,
                                            'ram_usage': projected_ram_usage,
                                            'io_usage': projected_io_usage,
                                            'vm_count': projected_vm_count
                                        }
                                    })
                                    
                            except Exception as e:
                                st.warning(f"Could not get metrics for host {host_name}: {e}")
                        
                        # Stage 3: Display results
                        ai_status.text("Stage 3/3: Generating AI predictions...")
                        ai_progress.progress(100)
                        time.sleep(0.5)
                        ai_progress.empty()
                        ai_status.empty()
                        
                        if host_predictions:
                            st.success(f"✅ AI predictions generated for {len(host_predictions)} hosts")
                            
                            # Sort by best ensemble prediction
                            best_model = 'ensemble' if 'ensemble' in host_predictions[0]['predictions'] else list(host_predictions[0]['predictions'].keys())[0]
                            host_predictions.sort(key=lambda x: x['predictions'].get(best_model, 0), reverse=True)
                            
                            # Display predictions
                            for i, pred in enumerate(host_predictions):
                                with st.expander(f"🏠 {pred['host_name']} - AI Score: {pred['predictions'].get(best_model, 0):.3f}", expanded=(i==0)):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("Current Host Metrics")
                                        st.metric("CPU Usage", f"{pred['host_metrics'].get('cpu_usage', 0):.1%}")
                                        st.metric("RAM Usage", f"{pred['host_metrics'].get('ram_usage', 0):.1%}")
                                        st.metric("I/O Usage", f"{pred['host_metrics'].get('io_usage', 0):.1%}")
                                        st.metric("VM Count", pred['host_metrics'].get('vm_count', 0))
                                    
                                    with col2:
                                        st.subheader("Projected Metrics")
                                        st.metric("CPU Usage", f"{pred['projected_metrics']['cpu_usage']:.1%}")
                                        st.metric("RAM Usage", f"{pred['projected_metrics']['ram_usage']:.1%}")
                                        st.metric("I/O Usage", f"{pred['projected_metrics']['io_usage']:.1%}")
                                        st.metric("VM Count", pred['projected_metrics']['vm_count'])
                                    
                                    st.subheader("AI Model Predictions")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    for j, (model_name, score) in enumerate(pred['predictions'].items()):
                                        with col1 if j == 0 else col2 if j == 1 else col3:
                                            # Color code based on score
                                            if score >= 0.7:
                                                st.metric(f"{model_name.upper()}", f"{score:.3f}", delta="Excellent")
                                            elif score >= 0.5:
                                                st.metric(f"{model_name.upper()}", f"{score:.3f}", delta="Good")
                                            elif score >= 0.3:
                                                st.metric(f"{model_name.upper()}", f"{score:.3f}", delta="Acceptable")
                                            else:
                                                st.metric(f"{model_name.upper()}", f"{score:.3f}", delta="Poor")
                                    
                                    # AI reasoning
                                    st.subheader("AI Reasoning")
                                    best_score = pred['predictions'].get(best_model, 0)
                                    if best_score >= 0.7:
                                        st.success("🎯 Excellent placement candidate - High confidence prediction")
                                    elif best_score >= 0.5:
                                        st.info("✅ Good placement candidate - Moderate confidence")
                                    elif best_score >= 0.3:
                                        st.warning("⚠️ Acceptable placement - Consider alternatives")
                                    else:
                                        st.error("❌ Poor placement candidate - Avoid this host")
                                    
                                    # Show technical details in an expander (no page refresh)
                                    with st.expander(f"🔧 Technical Details for {pred['host_name']}", expanded=False):
                                        st.json({
                                            'vm_metrics': pred['vm_metrics'],
                                            'host_metrics': pred['host_metrics'],
                                            'projected_metrics': pred['projected_metrics'],
                                            'ai_predictions': pred['predictions']
                                        })
                        else:
                            st.warning("No predictions generated. Check data availability.")
                            
                    except Exception as e:
                        st.error(f"❌ AI prediction failed: {e}")
                        with st.expander("🔍 Debug Information"):
                            st.code(f"Error: {e}")
        else:
            st.warning("⚠️ No trained AI models available. Train models first to enable AI predictions.")
        
        # Model training section
        st.header("Model Training")
        st.write("Train the AI models with current VM and host data for better recommendations.")
        
        if st.button("Train Models"):
            # Create progress indicators for training
            train_progress = st.progress(0)
            train_status = st.empty()
            
            try:
                # Stage 1: Data preparation
                train_status.text("Stage 1/4: Preparing training data...")
                train_progress.progress(25)
                time.sleep(0.5)
                
                # Convert data to the format expected by the training function
                vm_list = list(vms.values())
                host_list = list(hosts.values())
                
                # Stage 2: Feature extraction
                train_status.text("Stage 2/4: Extracting features from VM and host data...")
                train_progress.progress(50)
                time.sleep(0.5)
                
                # Stage 3: Model training
                train_status.text("Stage 3/4: Training AI models...")
                train_progress.progress(75)
                time.sleep(0.5)
                
                # Stage 4: Model validation
                train_status.text("Stage 4/4: Validating trained models...")
                train_progress.progress(90)
                time.sleep(0.5)
                
                success = optimization_engine.train_models(vm_list, host_list)
                
                # Complete
                train_progress.progress(100)
                train_status.text("✅ Training complete!")
                time.sleep(0.5)
                train_progress.empty()
                train_status.empty()
                
                if success:
                    st.success("✅ Models trained successfully!")
                    st.info(f"Trained with {len(vm_list)} VMs and {len(host_list)} hosts")
                else:
                    st.warning("⚠️ Model training completed with warnings. Check data availability.")
                        
            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
        
    except ImportError as e:
        st.error(f"❌ Failed to import AI Optimizer modules: {e}")
        st.info("Make sure the ai_optimizer module is properly installed and accessible.") 
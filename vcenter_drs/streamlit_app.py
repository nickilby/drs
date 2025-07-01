import streamlit as st
import sys
import os
from api.collect_and_store_metrics import main as collect_and_store_metrics_main
from rules.rules_engine import evaluate_rules, get_db_state, load_rules, parse_alias_and_role
import time
import threading
from collections import defaultdict
from db.metrics_db import MetricsDB
import json
from prometheus_client import start_http_server, Gauge, Counter, Histogram, REGISTRY
import atexit

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
    SERVICE_UP = REGISTRY._names_to_collectors['vcenter_drs_service_up']
    RULE_VIOLATIONS = REGISTRY._names_to_collectors['vcenter_drs_rule_violations_total']
    VM_COUNT = REGISTRY._names_to_collectors['vcenter_drs_vm_count']
    HOST_COUNT = REGISTRY._names_to_collectors['vcenter_drs_host_count']
    LAST_COLLECTION_TIME = REGISTRY._names_to_collectors['vcenter_drs_last_collection_timestamp']
    COMPLIANCE_CHECK_DURATION = REGISTRY._names_to_collectors['vcenter_drs_compliance_check_duration_seconds']
    UPTIME = REGISTRY._names_to_collectors['vcenter_drs_uptime_seconds']

# Set service as up
SERVICE_UP.set(1)
start_time = time.time()

# Initialize violation metrics with 0 values to ensure they're always present
known_rule_types = ['anti-affinity', 'dataset-affinity', 'affinity']
for rule_type in known_rule_types:
    RULE_VIOLATIONS.labels(rule_type=rule_type).set(0)

# Update uptime periodically
def update_uptime():
    while True:
        UPTIME.set(time.time() - start_time)
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
page = st.sidebar.radio("Navigation", ["Compliance Dashboard", "Exception Management", "Rule Management", "VM Rule Validator"])

if page == "Compliance Dashboard":
    if 'violations' not in st.session_state:
        st.session_state['violations'] = None

    if st.button("Run Compliance Check"):
        # Use the new structured output with timing
        start_time = time.time()
        structured_violations = evaluate_rules(selected_cluster if selected_cluster != "All Clusters" else None, return_structured=True)
        end_time = time.time()
        
        # Record compliance check duration
        COMPLIANCE_CHECK_DURATION.observe(end_time - start_time)
        
        # Update violation metrics
        violation_counts = defaultdict(int)
        if structured_violations:
            for violation in structured_violations:
                rule_type = violation.get('type', 'unknown')
                violation_counts[rule_type] += 1
        
        # Always ensure metrics exist by setting all known rule types to 0 first
        known_rule_types = ['anti-affinity', 'dataset-affinity', 'affinity']
        for rule_type in known_rule_types:
            RULE_VIOLATIONS.labels(rule_type=rule_type).set(0)
        
        # Then set the actual counts
        for rule_type, count in violation_counts.items():
            RULE_VIOLATIONS.labels(rule_type=rule_type).set(count)
        
        st.session_state['violations'] = structured_violations
        if not structured_violations:
            st.success("✅ All VMs in this cluster are compliant! No violations found.")
            st.balloons()

    # Only display violations if they exist in session state
    if st.session_state['violations']:
        from collections import defaultdict
        cluster_grouped = defaultdict(list)
        for v in st.session_state['violations']:
            cluster_grouped[v["cluster"]].append(v)
        for cluster_name, violations in cluster_grouped.items():
            st.markdown(f"### Cluster: {cluster_name}")
            for idx, violation in enumerate(violations):
                expander_title = f"Alias: {violation['alias']} | Rule: {violation['type']}"
                with st.expander(expander_title, expanded=True):
                    st.code(violation["violation_text"])
                    st.write(f"**Rule Type:** {violation['type']}")
                    st.write(f"**Alias:** {violation['alias']}")
                    st.write(f"**Affected VMs:** {', '.join(violation['affected_vms'])}")
                    if st.button(f"Remediate Violation {cluster_name}-{idx+1}", key=f"remediate_{cluster_name}_{idx}"):
                        st.success(f"Remediation triggered for violation {idx+1} in {cluster_name}.\nRule type: {violation['type']} | Alias: {violation['alias']} | VMs: {', '.join(violation['affected_vms'])}")
                    if st.button("Add Exception", key=f"add_exc_{cluster_name}_{idx}"):
                        import hashlib
                        # Normalize fields for robust matching
                        alias = (violation.get('alias') or '').strip().lower()
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
                            st.success(f"Exception added for Alias: {violation.get('alias')} | Rule: {violation.get('type')}. This violation will be ignored in future checks.")
                        except Exception as e:
                            st.error(f"[ERROR] Failed to add exception: {e}")
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
            with st.expander(f"Alias: {exc['alias']} | Rule: {exc['rule_type']} | Cluster: {exc['cluster']}"):
                st.write(f"**Affected VMs:** {', '.join(exc['affected_vms']) if isinstance(exc['affected_vms'], list) else exc['affected_vms']}")
                st.write(f"**Created At:** {exc['created_at']}")
                st.write(f"**Reason:** {exc['reason']}")
                if st.button(f"Remove Exception {exc['id']}", key=f"remove_exc_{exc['id']}"):
                    db = MetricsDB()
                    db.connect()
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
        rule_type = st.selectbox("Type", ["affinity", "anti-affinity", "dataset-affinity"])
        level = st.selectbox("Level", ["host", "cluster", "(none)"])
        role = st.text_input("Role (comma-separated for multiple, leave blank if using name_pattern)")
        name_pattern = st.text_input("Name Pattern (optional)")
        dataset_pattern = st.text_input("Dataset Pattern (comma-separated, for dataset-affinity only)")
        submitted = st.form_submit_button("Add Rule")
        if submitted:
            new_rule = {"type": rule_type}
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
    dataset_names = set(v['dataset_name'] for v in vms.values() if v['dataset_name'])
    dataset_names = sorted(list(dataset_names))

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
                                other_hosts = set(v['host_id'] for v in vms.values() if parse_alias_and_role(v['name']) == (alias, role))
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
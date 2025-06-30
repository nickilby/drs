import streamlit as st
import sys
import os
from api.collect_and_store_metrics import main as collect_and_store_metrics_main
from rules.rules_engine import evaluate_rules, get_db_state
import time
import threading
from collections import defaultdict
from db.metrics_db import MetricsDB
import json

st.set_page_config(page_title="vCenter DRS Compliance Dashboard", layout="wide")

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

if 'violations' not in st.session_state:
    st.session_state['violations'] = None

if st.button("Run Compliance Check"):
    # Use the new structured output
    structured_violations = evaluate_rules(selected_cluster if selected_cluster != "All Clusters" else None, return_structured=True)
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
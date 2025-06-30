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

# Add sidebar navigation
page = st.sidebar.radio("Navigation", ["Compliance Dashboard", "Exception Management", "Rule Management"])

if page == "Compliance Dashboard":
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
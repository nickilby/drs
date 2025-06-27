import streamlit as st
import sys
import os
from api.collect_and_store_metrics import main as collect_and_store_metrics_main
from rules.rules_engine import evaluate_rules, get_db_state

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
def run_rules_engine_for_cluster(selected_cluster):
    import io
    import contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        evaluate_rules(selected_cluster if selected_cluster != "All Clusters" else None)
    return output.getvalue()

if st.button("Run Compliance Check"):
    with st.spinner("Collecting latest data from vCenter..."):
        collect_and_store_metrics_main()  # Refresh the database
    results = run_rules_engine_for_cluster(selected_cluster)
    st.code(results)
else:
    st.info("Click 'Run Compliance Check' to evaluate compliance for the selected cluster.") 
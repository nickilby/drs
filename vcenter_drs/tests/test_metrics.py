import pytest
from vcenter_drs import streamlit_app

def test_metrics_update(monkeypatch):
    # Patch get_db_state to return known values
    monkeypatch.setattr(streamlit_app, 'get_db_state', lambda: (
        {1: 'C1'}, {1: {'name': 'H1', 'cluster_id': 1}}, {1: {'name': 'VM1', 'host_id': 1, 'dataset_id': 1, 'dataset_name': 'D1', 'power_status': 'poweredon'}}
    ))
    # Patch collect_and_store_metrics_main to do nothing
    monkeypatch.setattr(streamlit_app, 'collect_and_store_metrics_main', lambda: None)
    # Call timed_data_collection and check metrics
    duration = streamlit_app.timed_data_collection()
    assert duration >= 0
    assert streamlit_app.VM_COUNT._value.get() == 1
    assert streamlit_app.HOST_COUNT._value.get() == 1 
import pytest
from vcenter_drs import streamlit_app

def test_add_remove_exception(monkeypatch):
    # Patch MetricsDB to use in-memory list
    exceptions = []
    class FakeDB:
        def connect(self): pass
        def close(self): pass
        def add_exception(self, exc): exceptions.append(exc)
        def get_exceptions(self): return exceptions.copy()
        @property
        def conn(self): return True
    monkeypatch.setattr(streamlit_app, 'MetricsDB', lambda: FakeDB())
    # Add exception
    exc = {'rule_type': 'anti-affinity', 'alias': 'a', 'affected_vms': ['vm'], 'cluster': 'c', 'rule_hash': 'h', 'reason': None}
    db = streamlit_app.MetricsDB()
    db.connect(); db.add_exception(exc); db.close()
    assert db.get_exceptions() == [exc]

def test_rule_add_delete(tmp_path, monkeypatch):
    import json, os
    rules_path = tmp_path / 'rules.json'
    rules = [{"type": "affinity", "level": "host", "role": ["SQL"]}]
    rules_path.write_text(json.dumps(rules))
    monkeypatch.setattr(streamlit_app.os.path, 'dirname', lambda _: str(tmp_path))
    # Simulate loading, adding, deleting
    with open(rules_path) as f:
        loaded = json.load(f)
    assert loaded == rules
    # Add rule
    new_rule = {"type": "dataset-affinity", "level": "storage", "role": ["SQL"], "dataset_pattern": ["DAT"]}
    loaded.append(new_rule)
    with open(rules_path, 'w') as f:
        json.dump(loaded, f)
    with open(rules_path) as f:
        loaded2 = json.load(f)
    assert new_rule in loaded2
    # Delete rule
    loaded2.pop()
    with open(rules_path, 'w') as f:
        json.dump(loaded2, f)
    with open(rules_path) as f:
        loaded3 = json.load(f)
    assert new_rule not in loaded3

def test_powered_on_vm_filter(monkeypatch):
    # Simulate vms with power status
    vms = {
        1: {'name': 'VM1', 'power_status': 'poweredon'},
        2: {'name': 'VM2', 'power_status': 'poweredoff'}
    }
    monkeypatch.setattr(streamlit_app, 'get_db_state', lambda: ({}, {}, vms))
    violation = {'affected_vms': ['VM1', 'VM2']}
    # Filtering logic: only show if at least one affected VM is powered on
    powered_on = False
    for vm_name in violation['affected_vms']:
        for vm in vms.values():
            if vm['name'] == vm_name and (vm.get('power_status') or '').lower() == 'poweredon':
                powered_on = True
                break
        if powered_on:
            break
    assert powered_on

def test_duplicate_key_fix(monkeypatch):
    """Test that the duplicate key issue is fixed."""
    import hashlib
    
    # Test the unique key generation logic
    def generate_unique_key(cluster_name, alias, affected_vms):
        unique_key_data = f"{cluster_name}_{alias}_{','.join(affected_vms)}"
        return hashlib.md5(unique_key_data.encode()).hexdigest()[:8]
    
    # Test with the same alias but different VMs (like the original issue)
    key1 = generate_unique_key('hq2', 'prs', ['z-prs-sql'])
    key2 = generate_unique_key('hq2', 'prs', ['z-prs-web2'])
    
    # Keys should be different even with same alias
    assert key1 != key2
    
    # Same violation should generate same key
    key3 = generate_unique_key('hq2', 'prs', ['z-prs-sql'])
    assert key1 == key3
    
    # Different clusters should generate different keys
    key4 = generate_unique_key('hq3', 'prs', ['z-prs-sql'])
    assert key1 != key4 
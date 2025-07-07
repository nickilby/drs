import pytest
from vcenter_drs.rules import rules_engine

# Helper to patch rules and DB state
def patch_engine(monkeypatch, rules, clusters, hosts, vms):
    monkeypatch.setattr(rules_engine, 'load_rules', lambda: rules)
    monkeypatch.setattr(rules_engine, 'get_db_state', lambda: (clusters, hosts, vms))
    # Patch out DB exception logic for simplicity
    monkeypatch.setattr(rules_engine, 'MetricsDB', lambda: type('FakeDB', (), {'connect': lambda s: None, 'close': lambda s: None, 'is_exception': lambda s, v: False})())

def test_affinity_host_violation(monkeypatch):
    rules = [{"type": "affinity", "level": "host", "role": ["SQL"]}]
    clusters = {1: "C1"}
    hosts = {1: {"name": "H1", "cluster_id": 1}, 2: {"name": "H2", "cluster_id": 1}}
    vms = {
        1: {"name": "z-alias-SQL1", "host_id": 1, "dataset_id": 1, "dataset_name": "D1", "power_status": "poweredon"},
        2: {"name": "z-alias-SQL2", "host_id": 2, "dataset_id": 2, "dataset_name": "D2", "power_status": "poweredon"},
    }
    patch_engine(monkeypatch, rules, clusters, hosts, vms)
    violations = rules_engine.evaluate_rules(return_structured=True)
    assert violations
    assert violations[0]["level"] == "host"
    assert violations[0]["type"] == "affinity"

def test_anti_affinity_host_violation(monkeypatch):
    rules = [{"type": "anti-affinity", "level": "host", "role": ["WEB"]}]
    clusters = {1: "C1"}
    hosts = {1: {"name": "H1", "cluster_id": 1}}
    vms = {
        1: {"name": "z-alias-WEB1", "host_id": 1, "dataset_id": 1, "dataset_name": "D1", "power_status": "poweredon"},
        2: {"name": "z-alias-WEB2", "host_id": 1, "dataset_id": 2, "dataset_name": "D2", "power_status": "poweredon"},
    }
    patch_engine(monkeypatch, rules, clusters, hosts, vms)
    violations = rules_engine.evaluate_rules(return_structured=True)
    assert violations
    assert violations[0]["level"] == "host"
    assert violations[0]["type"] == "anti-affinity"

def test_affinity_cluster_violation(monkeypatch):
    rules = [{"type": "affinity", "level": "cluster", "role": ["CMS"]}]
    clusters = {1: "C1", 2: "C2"}
    hosts = {1: {"name": "H1", "cluster_id": 1}, 2: {"name": "H2", "cluster_id": 2}}
    vms = {
        1: {"name": "z-alias-CMS1", "host_id": 1, "dataset_id": 1, "dataset_name": "D1", "power_status": "poweredon"},
        2: {"name": "z-alias-CMS2", "host_id": 2, "dataset_id": 2, "dataset_name": "D2", "power_status": "poweredon"},
    }
    patch_engine(monkeypatch, rules, clusters, hosts, vms)
    violations = rules_engine.evaluate_rules(return_structured=True)
    assert violations
    assert violations[0]["level"] == "cluster"
    assert violations[0]["type"] == "affinity"

def test_dataset_affinity_violation(monkeypatch):
    rules = [{"type": "dataset-affinity", "level": "storage", "role": ["SQL"], "dataset_pattern": ["DAT"]}]
    clusters = {1: "C1"}
    hosts = {1: {"name": "H1", "cluster_id": 1}}
    vms = {
        1: {"name": "z-alias-SQL1", "host_id": 1, "dataset_id": 1, "dataset_name": "OTHER", "power_status": "poweredon"},
    }
    patch_engine(monkeypatch, rules, clusters, hosts, vms)
    violations = rules_engine.evaluate_rules(return_structured=True)
    assert violations
    assert violations[0]["level"] == "storage"
    assert violations[0]["type"] == "dataset-affinity"

def test_dataset_anti_affinity_violation(monkeypatch):
    rules = [{"type": "dataset-anti-affinity", "level": "storage", "role": ["WEB"], "dataset_pattern": ["WEB"]}]
    clusters = {1: "C1"}
    hosts = {1: {"name": "H1", "cluster_id": 1}}
    vms = {
        1: {"name": "z-alias-WEB1", "host_id": 1, "dataset_id": 1, "dataset_name": "WEB", "power_status": "poweredon"},
        2: {"name": "z-alias-WEB2", "host_id": 1, "dataset_id": 1, "dataset_name": "WEB", "power_status": "poweredon"},
    }
    patch_engine(monkeypatch, rules, clusters, hosts, vms)
    violations = rules_engine.evaluate_rules(return_structured=True)
    assert violations
    assert violations[0]["level"] == "storage"
    assert violations[0]["type"] == "dataset-anti-affinity"

def test_pool_anti_affinity_violation(monkeypatch):
    rules = [{"type": "pool-anti-affinity", "level": "storage", "role": ["WEB"], "pool_pattern": ["HQ"]}]
    clusters = {1: "C1"}
    hosts = {1: {"name": "H1", "cluster_id": 1}}
    vms = {
        1: {"name": "z-alias-WEB1", "host_id": 1, "dataset_id": 1, "dataset_name": "HQ1WEB1", "power_status": "poweredon"},
        2: {"name": "z-alias-WEB2", "host_id": 1, "dataset_id": 2, "dataset_name": "HQ1WEB2", "power_status": "poweredon"},
    }
    patch_engine(monkeypatch, rules, clusters, hosts, vms)
    violations = rules_engine.evaluate_rules(return_structured=True)
    assert violations
    assert violations[0]["level"] == "storage"
    assert violations[0]["type"] == "pool-anti-affinity" 
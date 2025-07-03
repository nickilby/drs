"""
Unit tests for compliance rules engine.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from vcenter_drs.rules.rules_engine import load_rules, parse_alias_and_role, evaluate_rules


class TestComplianceRulesEngine:
    """Test cases for compliance rules engine functions."""

    def test_parse_alias_and_role_basic(self):
        """Test basic alias and role parsing."""
        alias, role = parse_alias_and_role("test-vm-web1")
        assert alias == "test-vm"
        assert role == "WEB"

    def test_parse_alias_and_role_with_z_prefix(self):
        """Test alias and role parsing with z- prefix."""
        alias, role = parse_alias_and_role("z-test-vm-web1")
        assert alias == "test-vm"
        assert role == "WEB"

    def test_parse_alias_and_role_no_number(self):
        """Test alias and role parsing without trailing number."""
        alias, role = parse_alias_and_role("test-vm-web")
        assert alias == "test-vm"
        assert role == "WEB"

    def test_parse_alias_and_role_no_dash(self):
        """Test alias and role parsing without dash."""
        alias, role = parse_alias_and_role("testvm")
        assert alias is None
        assert role is None

    def test_load_rules_from_file(self, temp_test_file):
        """Test loading rules from a JSON file."""
        # Create test rules file
        rules_data = [
            {
                "type": "affinity",
                "role": "WEB",
                "level": "cluster"
            },
            {
                "type": "anti-affinity",
                "role": "CACHE",
                "level": "host"
            }
        ]
        
        with open(temp_test_file, 'w') as f:
            json.dump(rules_data, f)
        
        rules = load_rules(temp_test_file)
        
        assert len(rules) == 2
        assert rules[0]["type"] == "affinity"
        assert rules[1]["type"] == "anti-affinity"

    def test_load_rules_from_file_invalid_json(self, temp_test_file):
        """Test loading rules from invalid JSON file."""
        with open(temp_test_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle invalid JSON gracefully
        with pytest.raises(json.JSONDecodeError):
            load_rules(temp_test_file)

    @patch('vcenter_drs.rules.rules_engine.get_db_state')
    @patch('vcenter_drs.rules.rules_engine.MetricsDB')
    def test_evaluate_rules_mock_data(self, mock_metrics_db, mock_get_db_state):
        """Test rule evaluation with mock data."""
        # Mock database state
        mock_clusters = {1: "test-cluster"}
        mock_hosts = {1: {"name": "test-host", "cluster_id": 1}}
        mock_vms = {1: {"name": "test-vm-web1", "host_id": 1, "dataset_id": 1, "dataset_name": "TRA"}}
        
        mock_get_db_state.return_value = (mock_clusters, mock_hosts, mock_vms)
        
        # Mock database connection
        mock_db = Mock()
        mock_db.is_exception.return_value = False
        mock_metrics_db.return_value = mock_db
        
        # Test evaluation
        violations = evaluate_rules(return_structured=True)
        
        # Should return structured violations
        assert isinstance(violations, list)

    def test_parse_alias_and_role_complex_patterns(self):
        """Test parsing of complex VM name patterns."""
        test_cases = [
            ("app-server-web1", "app-server", "WEB"),
            ("z-db-server-cache2", "db-server", "CACHE"),
            ("frontend-app-web", "frontend-app", "WEB"),
            ("simple-name", "simple", "NAME"),
            ("no-dash-name", None, None),
        ]
        
        for vm_name, expected_alias, expected_role in test_cases:
            alias, role = parse_alias_and_role(vm_name)
            assert alias == expected_alias, f"Failed for {vm_name}"
            assert role == expected_role, f"Failed for {vm_name}"

    @patch('vcenter_drs.rules.rules_engine.get_db_state')
    @patch('vcenter_drs.rules.rules_engine.MetricsDB')
    def test_evaluate_rules_with_cluster_filter(self, mock_metrics_db, mock_get_db_state):
        """Test rule evaluation with cluster filtering."""
        # Mock database state with multiple clusters
        mock_clusters = {1: "cluster1", 2: "cluster2"}
        mock_hosts = {
            1: {"name": "host1", "cluster_id": 1},
            2: {"name": "host2", "cluster_id": 2}
        }
        mock_vms = {
            1: {"name": "test-vm-web1", "host_id": 1, "dataset_id": 1, "dataset_name": "TRA"},
            2: {"name": "test-vm-web2", "host_id": 2, "dataset_id": 1, "dataset_name": "TRA"}
        }
        
        mock_get_db_state.return_value = (mock_clusters, mock_hosts, mock_vms)
        
        # Mock database connection
        mock_db = Mock()
        mock_db.is_exception.return_value = False
        mock_metrics_db.return_value = mock_db
        
        # Test evaluation with cluster filter
        violations = evaluate_rules(cluster_filter="cluster1", return_structured=True)
        
        # Should return structured violations
        assert isinstance(violations, list)

    def test_load_rules_default_path(self):
        """Test loading rules from default path."""
        # This test assumes the default rules.json exists
        try:
            rules = load_rules()
            assert isinstance(rules, list)
        except FileNotFoundError:
            # If rules.json doesn't exist, that's okay for testing
            pytest.skip("Default rules.json not found") 
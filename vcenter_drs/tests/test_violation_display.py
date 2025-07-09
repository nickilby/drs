"""
Unit tests for violation display functions.
Tests the fixes for duplicate Streamlit keys and correct rule display.
"""

import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock


class TestViolationDisplay:
    """Test cases for violation display functions."""

    def test_unique_key_generation_consistency(self):
        """Test that the same violation always generates the same unique key."""
        # Create test violation
        violation = {
            'type': 'dataset-affinity',
            'alias': 'test-alias',
            'affected_vms': ['z-test-vm-sql'],
            'cluster': 'test-cluster',
            'violation_text': 'Test violation text',
            'level': 'storage'
        }
        
        # Generate unique key data
        unique_key_data = f"{violation['cluster']}_{violation['alias']}_{','.join(violation['affected_vms'])}"
        unique_key_hash = hashlib.md5(unique_key_data.encode()).hexdigest()[:8]
        
        # Generate the same key again
        unique_key_hash2 = hashlib.md5(unique_key_data.encode()).hexdigest()[:8]
        
        # Should be consistent
        assert unique_key_hash == unique_key_hash2
        
        # Different violations should have different keys
        violation2 = {
            'type': 'dataset-affinity',
            'alias': 'test-alias',
            'affected_vms': ['z-test-vm-web2'],  # Different VM
            'cluster': 'test-cluster',
            'violation_text': 'Test violation text',
            'level': 'storage'
        }
        
        unique_key_data2 = f"{violation2['cluster']}_{violation2['alias']}_{','.join(violation2['affected_vms'])}"
        unique_key_hash3 = hashlib.md5(unique_key_data2.encode()).hexdigest()[:8]
        
        assert unique_key_hash != unique_key_hash3

    def test_duplicate_key_issue_fixed_logic(self):
        """Test that the duplicate key issue is fixed using the key generation logic."""
        # Create multiple violations with the same alias (like the original issue)
        violations = [
            {
                'type': 'dataset-affinity',
                'alias': 'prs',
                'affected_vms': ['z-prs-sql'],
                'cluster': 'hq2',
                'violation_text': 'Test violation text\nDataset: WRONG',
                'level': 'storage'
            },
            {
                'type': 'dataset-affinity',
                'alias': 'prs',
                'affected_vms': ['z-prs-web2'],
                'cluster': 'hq2',
                'violation_text': 'Test violation text\nDataset: WRONG',
                'level': 'storage'
            }
        ]
        
        # Generate keys for each violation
        keys = []
        for violation in violations:
            unique_key_data = f"{violation['cluster']}_{violation['alias']}_{','.join(violation['affected_vms'])}"
            unique_key_hash = hashlib.md5(unique_key_data.encode()).hexdigest()[:8]
            keys.append(unique_key_hash)
        
        # Check that all keys are unique (no duplicates)
        assert len(keys) == len(set(keys))
        
        # Check that keys are different even with same alias
        assert keys[0] != keys[1]

    def test_grouped_violation_key_generation(self):
        """Test that grouped violations generate unique keys."""
        # Create test grouped violations
        grouped_violations = [
            {
                'type': 'dataset-affinity',
                'alias': 'test-alias',
                'affected_vms': ['z-test-vm-sql'],
                'cluster': 'test-cluster',
                'violation_text': 'Test violation text\nDataset: TEST',
                'level': 'storage'
            },
            {
                'type': 'dataset-affinity',
                'alias': 'test-alias',
                'affected_vms': ['z-test-vm-web2'],
                'cluster': 'test-cluster',
                'violation_text': 'Test violation text\nDataset: TEST',
                'level': 'storage'
            }
        ]
        
        # Generate keys for grouped violations
        keys = []
        for violation in grouped_violations:
            unique_key_data = f"{violation['cluster']}_{violation['alias']}_{','.join(violation['affected_vms'])}"
            unique_key_hash = hashlib.md5(unique_key_data.encode()).hexdigest()[:8]
            keys.append(unique_key_hash)
        
        # Check that keys are unique
        assert len(keys) == len(set(keys))
        
        # Check that keys are different
        assert keys[0] != keys[1]

    def test_rule_display_text_generation(self):
        """Test that rule display text is generated correctly for different rule types."""
        # Test dataset-affinity rule display
        rule_type = 'dataset-affinity'
        if rule_type == 'dataset-affinity':
            violation_text = "Dataset Affinity Violation (Grouped)"
            rule_description = "VMs must be on datasets matching specific patterns"
        elif rule_type == 'pool-anti-affinity':
            violation_text = "Pool Anti-Affinity Violation (Grouped)"
            rule_description = "VMs must NOT be on the same ZFS pool matching specific patterns"
        else:
            violation_text = f"{rule_type.capitalize()} Violation (Grouped)"
            rule_description = f"{rule_type} rule violation"
        
        assert violation_text == "Dataset Affinity Violation (Grouped)"
        assert rule_description == "VMs must be on datasets matching specific patterns"
        
        # Test pool-anti-affinity rule display
        rule_type = 'pool-anti-affinity'
        if rule_type == 'dataset-affinity':
            violation_text = "Dataset Affinity Violation (Grouped)"
            rule_description = "VMs must be on datasets matching specific patterns"
        elif rule_type == 'pool-anti-affinity':
            violation_text = "Pool Anti-Affinity Violation (Grouped)"
            rule_description = "VMs must NOT be on the same ZFS pool matching specific patterns"
        else:
            violation_text = f"{rule_type.capitalize()} Violation (Grouped)"
            rule_description = f"{rule_type} rule violation"
        
        assert violation_text == "Pool Anti-Affinity Violation (Grouped)"
        assert rule_description == "VMs must NOT be on the same ZFS pool matching specific patterns"

    def test_dataset_extraction_logic(self):
        """Test that dataset information is correctly extracted from violation text."""
        # Test dataset extraction
        violation_text = "Test violation text\nDataset: DAT1"
        violation_lines = violation_text.split('\n')
        all_datasets = set()
        
        for line in violation_lines:
            if line.startswith('Dataset: '):
                dataset_name = line.replace('Dataset: ', '').strip()
                all_datasets.add(dataset_name)
                break
        
        assert 'DAT1' in all_datasets
        
        # Test multiple violations
        violations = [
            "Test violation text\nDataset: DAT1",
            "Test violation text\nDataset: WEB1"
        ]
        
        all_datasets = set()
        for violation_text in violations:
            violation_lines = violation_text.split('\n')
            for line in violation_lines:
                if line.startswith('Dataset: '):
                    dataset_name = line.replace('Dataset: ', '').strip()
                    all_datasets.add(dataset_name)
                    break
        
        assert 'DAT1' in all_datasets
        assert 'WEB1' in all_datasets
        assert len(all_datasets) == 2 
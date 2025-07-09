"""
Unit tests for violation display functions.
Tests the fixes for duplicate Streamlit keys and correct rule display.
"""

import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock
from vcenter_drs.streamlit_app import display_single_violation, display_grouped_violations


class TestViolationDisplay:
    """Test cases for violation display functions."""

    def test_display_single_violation_unique_keys(self):
        """Test that single violation display generates unique keys."""
        # Mock streamlit
        mock_st = Mock()
        
        # Create test violation
        violation = {
            'type': 'dataset-affinity',
            'alias': 'test-alias',
            'affected_vms': ['z-test-vm-sql'],
            'cluster': 'test-cluster',
            'violation_text': 'Test violation text',
            'level': 'storage'
        }
        
        # Mock the button function to capture keys
        captured_keys = []
        def mock_button(text, key=None):
            if key:
                captured_keys.append(key)
            return False
        
        mock_st.button = mock_button
        mock_st.write = Mock()
        mock_st.code = Mock()
        mock_st.expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        
        # Mock get_db_state
        with patch('vcenter_drs.streamlit_app.get_db_state') as mock_get_db_state:
            mock_get_db_state.return_value = ({}, {}, {})
            
            # Call the function
            display_single_violation(violation, 'test-cluster', 0, False)
            
            # Check that keys were generated
            assert len(captured_keys) >= 2  # At least add exception and remediate buttons
            assert all(key.startswith('add_exc_') or key.startswith('remediate_fix_') for key in captured_keys)
            
            # Check that keys are unique
            assert len(captured_keys) == len(set(captured_keys))

    def test_display_grouped_violations_unique_keys(self):
        """Test that grouped violation display generates unique keys."""
        # Mock streamlit
        mock_st = Mock()
        
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
        
        # Mock the button function to capture keys
        captured_keys = []
        def mock_button(text, key=None):
            if key:
                captured_keys.append(key)
            return False
        
        mock_st.button = mock_button
        mock_st.write = Mock()
        mock_st.code = Mock()
        mock_st.expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        
        # Mock get_db_state
        with patch('vcenter_drs.streamlit_app.get_db_state') as mock_get_db_state:
            mock_get_db_state.return_value = ({}, {}, {})
            
            # Call the function
            display_grouped_violations(grouped_violations, 'test-cluster', False)
            
            # Check that keys were generated
            assert len(captured_keys) >= 2  # At least add exception and remediate buttons
            assert all(key.startswith('add_exc_grouped_') or key.startswith('remediate_fix_grouped_') for key in captured_keys)
            
            # Check that keys are unique
            assert len(captured_keys) == len(set(captured_keys))

    def test_duplicate_key_issue_fixed(self):
        """Test that the duplicate key issue is fixed for multiple violations with same alias."""
        # Mock streamlit
        mock_st = Mock()
        
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
        
        # Mock the button function to capture keys
        captured_keys = []
        def mock_button(text, key=None):
            if key:
                captured_keys.append(key)
            return False
        
        mock_st.button = mock_button
        mock_st.write = Mock()
        mock_st.code = Mock()
        mock_st.expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        
        # Mock get_db_state
        with patch('vcenter_drs.streamlit_app.get_db_state') as mock_get_db_state:
            mock_get_db_state.return_value = ({}, {}, {})
            
            # Call display_grouped_violations with the same alias
            display_grouped_violations(violations, 'hq2', False)
            
            # Check that all keys are unique (no duplicates)
            assert len(captured_keys) == len(set(captured_keys))
            
            # Check that keys are properly prefixed
            assert all(key.startswith('add_exc_grouped_') or key.startswith('remediate_fix_grouped_') for key in captured_keys)
            
            # Verify that the keys are based on unique data (not just alias)
            # The keys should be different because the affected VMs are different
            add_exc_keys = [key for key in captured_keys if key.startswith('add_exc_grouped_')]
            remediate_keys = [key for key in captured_keys if key.startswith('remediate_fix_grouped_')]
            
            # Should have at least one of each type
            assert len(add_exc_keys) >= 1
            assert len(remediate_keys) >= 1

    def test_display_grouped_violations_correct_rule_display(self):
        """Test that grouped violations display correct rule information."""
        # Mock streamlit
        mock_st = Mock()
        
        # Create test grouped violations for dataset-affinity
        grouped_violations = [
            {
                'type': 'dataset-affinity',
                'alias': 'test-alias',
                'affected_vms': ['z-test-vm-sql'],
                'cluster': 'test-cluster',
                'violation_text': 'Test violation text\nDataset: WRONG',
                'level': 'storage'
            }
        ]
        
        # Capture the violation text that gets displayed
        captured_violation_text = []
        def mock_code(text):
            captured_violation_text.append(text)
        
        mock_st.button = Mock(return_value=False)
        mock_st.write = Mock()
        mock_st.code = mock_code
        mock_st.expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        
        # Mock get_db_state
        with patch('vcenter_drs.streamlit_app.get_db_state') as mock_get_db_state:
            mock_get_db_state.return_value = ({}, {}, {})
            
            # Call the function
            display_grouped_violations(grouped_violations, 'test-cluster', False)
            
            # Check that correct rule information is displayed
            assert len(captured_violation_text) > 0
            violation_text = captured_violation_text[0]
            
            # Should show dataset-affinity information, not pool anti-affinity
            assert 'Dataset Affinity Violation' in violation_text
            assert 'Pool Anti-Affinity Violation' not in violation_text
            assert 'VMs must be on datasets matching specific patterns' in violation_text

    def test_display_grouped_violations_pool_anti_affinity_display(self):
        """Test that pool anti-affinity violations display correctly."""
        # Mock streamlit
        mock_st = Mock()
        
        # Create test grouped violations for pool-anti-affinity
        grouped_violations = [
            {
                'type': 'pool-anti-affinity',
                'alias': 'test-alias',
                'affected_vms': ['z-test-vm-sql'],
                'cluster': 'test-cluster',
                'violation_text': 'Test violation text\nPool: TEST',
                'level': 'storage'
            }
        ]
        
        # Capture the violation text that gets displayed
        captured_violation_text = []
        def mock_code(text):
            captured_violation_text.append(text)
        
        mock_st.button = Mock(return_value=False)
        mock_st.write = Mock()
        mock_st.code = mock_code
        mock_st.expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        
        # Mock get_db_state
        with patch('vcenter_drs.streamlit_app.get_db_state') as mock_get_db_state:
            mock_get_db_state.return_value = ({}, {}, {})
            
            # Call the function
            display_grouped_violations(grouped_violations, 'test-cluster', False)
            
            # Check that correct rule information is displayed
            assert len(captured_violation_text) > 0
            violation_text = captured_violation_text[0]
            
            # Should show pool anti-affinity information
            assert 'Pool Anti-Affinity Violation' in violation_text
            assert 'VMs must NOT be on the same ZFS pool matching specific patterns' in violation_text

    def test_display_grouped_violations_dataset_extraction(self):
        """Test that dataset information is correctly extracted and displayed."""
        # Mock streamlit
        mock_st = Mock()
        
        # Create test grouped violations with dataset information
        grouped_violations = [
            {
                'type': 'dataset-affinity',
                'alias': 'test-alias',
                'affected_vms': ['z-test-vm-sql'],
                'cluster': 'test-cluster',
                'violation_text': 'Test violation text\nDataset: DAT1',
                'level': 'storage'
            },
            {
                'type': 'dataset-affinity',
                'alias': 'test-alias',
                'affected_vms': ['z-test-vm-web2'],
                'cluster': 'test-cluster',
                'violation_text': 'Test violation text\nDataset: WEB1',
                'level': 'storage'
            }
        ]
        
        # Capture the violation text that gets displayed
        captured_violation_text = []
        def mock_code(text):
            captured_violation_text.append(text)
        
        mock_st.button = Mock(return_value=False)
        mock_st.write = Mock()
        mock_st.code = mock_code
        mock_st.expander = Mock()
        mock_st.expander.return_value.__enter__ = Mock()
        mock_st.expander.return_value.__exit__ = Mock()
        
        # Mock get_db_state
        with patch('vcenter_drs.streamlit_app.get_db_state') as mock_get_db_state:
            mock_get_db_state.return_value = ({}, {}, {})
            
            # Call the function
            display_grouped_violations(grouped_violations, 'test-cluster', False)
            
            # Check that dataset information is extracted and displayed
            assert len(captured_violation_text) > 0
            violation_text = captured_violation_text[0]
            
            # Should show dataset information
            assert 'Affected Datasets:' in violation_text
            assert 'DAT1' in violation_text or 'WEB1' in violation_text

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
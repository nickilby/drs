import pytest
from unittest.mock import patch, Mock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from vcenter_drs.streamlit_app import trigger_remediation_api, trigger_remediation_alias_api

def test_playbook_selection():
    """Test that the correct playbook is selected based on rule level"""
    # This test remains the same for the original function
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True, 'new_task_id': '12345', 'deduplicated': False}
        mock_response.text = '{"success": true, "new_task_id": "12345", "deduplicated": false}'  # Add text attribute
        mock_post.return_value = mock_response
        
        success, msg = trigger_remediation_api('test-alias', ['vm1', 'vm2'], 'token123', playbook_name='e-vmotion-storage')
        
        assert success
        assert 'Task ID: 12345' in msg
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['playbook_name'] == 'e-vmotion-storage'
        assert call_args[1]['json']['options']['limit'] == ['vm1', 'vm2']

def test_api_error():
    """Test API error handling"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_post.return_value = mock_response
        
        success, msg = trigger_remediation_api('test-alias', ['vm1'], 'token123')
        
        assert not success
        assert 'API call failed: 500' in msg

def test_network_error():
    """Test network error handling"""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = Exception('Connection timeout')
        
        success, msg = trigger_remediation_api('test-alias', ['vm1'], 'token123')
        
        assert not success
        assert 'API call error: Connection timeout' in msg

def test_alias_remediation_api():
    """Test the new alias remediation API function"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True, 'new_task_id': '67890', 'deduplicated': False}
        mock_response.text = '{"success": true, "new_task_id": "67890", "deduplicated": false}'  # Add text attribute
        mock_post.return_value = mock_response
        
        success, msg = trigger_remediation_alias_api('test-alias', 'token123', playbook_name='e-vmotion-server')
        
        assert success
        assert 'Alias remediation triggered!' in msg
        assert 'Task ID: 67890' in msg
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['playbook_name'] == 'e-vmotion-server'
        # Verify that there IS an 'options' with empty 'limit' in the payload (wildcard approach)
        assert 'options' in call_args[1]['json']
        assert call_args[1]['json']['options']['limit'] == []
        assert call_args[1]['json']['alias'] == 'test-alias'

def test_alias_remediation_api_storage():
    """Test alias remediation API with storage playbook"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True, 'new_task_id': '11111', 'deduplicated': True}
        mock_response.text = '{"success": true, "new_task_id": "11111", "deduplicated": true}'  # Add text attribute
        mock_post.return_value = mock_response
        
        success, msg = trigger_remediation_alias_api('test-alias', 'token123', playbook_name='e-vmotion-storage')
        
        assert success
        assert 'Alias remediation triggered!' in msg
        assert 'Task ID: 11111' in msg
        assert '(deduplicated: True)' in msg
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['playbook_name'] == 'e-vmotion-storage'
        # Verify that there IS an 'options' with empty 'limit' in the payload (wildcard approach)
        assert 'options' in call_args[1]['json']
        assert call_args[1]['json']['options']['limit'] == []

def test_alias_remediation_api_error():
    """Test alias remediation API error handling"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_post.return_value = mock_response
        
        success, msg = trigger_remediation_alias_api('test-alias', 'token123')
        
        assert not success
        assert 'API call failed: 400' in msg 
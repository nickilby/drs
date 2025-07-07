import pytest
from vcenter_drs import streamlit_app

class FakeResp:
    def __init__(self, status_code=200, json_data=None, text="OK"):
        self.status_code = status_code
        self._json = json_data or {"success": True, "new_task_id": "123", "deduplicated": False}
        self.text = text
    def json(self):
        return self._json

def test_playbook_selection(monkeypatch):
    # Patch requests.post to not actually call the API
    monkeypatch.setattr(streamlit_app.requests, "post", lambda *a, **k: FakeResp())
    # Host level
    success, msg = streamlit_app.trigger_remediation_api("alias1", ["vm1"], "token", playbook_name="e-vmotion-server")
    assert success
    assert "Task ID" in msg
    # Storage level
    success, msg = streamlit_app.trigger_remediation_api("alias2", ["vm2"], "token", playbook_name="e-vmotion-storage")
    assert success
    assert "Task ID" in msg

def test_api_error(monkeypatch):
    monkeypatch.setattr(streamlit_app.requests, "post", lambda *a, **k: FakeResp(500, {"success": False}, text="Internal Error"))
    success, msg = streamlit_app.trigger_remediation_api("alias", ["vm"], "token")
    assert not success
    assert "API call failed" in msg

def test_network_error(monkeypatch):
    def raise_exc(*a, **k):
        raise Exception("Network down")
    monkeypatch.setattr(streamlit_app.requests, "post", raise_exc)
    success, msg = streamlit_app.trigger_remediation_api("alias", ["vm"], "token")
    assert not success
    assert "API call error" in msg 
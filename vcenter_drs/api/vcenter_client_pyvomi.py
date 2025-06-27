import os
import json
import ssl
from pyVim.connect import SmartConnect, Disconnect

class VCenterPyVmomiClient:
    def __init__(self, host=None, username=None, password=None, credentials_path=None):
        if not (host and username and password):
            if credentials_path is None:
                credentials_path = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')
            with open(os.path.abspath(credentials_path), 'r') as f:
                creds = json.load(f)
            self.host = creds['host']
            self.username = creds['username']
            self.password = creds['password']
        else:
            self.host = host
            self.username = username
            self.password = password
        self.si = None

    def connect(self):
        context = ssl._create_unverified_context()
        self.si = SmartConnect(
            host=self.host,
            user=self.username,
            pwd=self.password,
            sslContext=context
        )
        return self.si

    def disconnect(self):
        if self.si:
            Disconnect(self.si)

# Example usage:
# client = VCenterPyVmomiClient()
# si = client.connect()
# print(si.CurrentTime())
# client.disconnect() 
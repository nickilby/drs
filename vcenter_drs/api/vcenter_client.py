import os
import json
from zeep import Client
from zeep.transports import Transport
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class VCenterClient:
    def __init__(self, host=None, username=None, password=None, credentials_path=None):
        if not (host and username and password):
            # Default path is one directory up from this file
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
        self.wsdl_url = f"https://{self.host}/sdk/vimService?wsdl"
        session = requests.Session()
        session.verify = False  # Ignore SSL verification
        self.client = Client(self.wsdl_url, transport=Transport(session=session))
        # Authentication will be handled in a login method

# Example usage:
vc = VCenterClient()  # Loads from credentials.json by default
# vc = VCenterClient(host="10.65.0.200", username="user", password="pass") 
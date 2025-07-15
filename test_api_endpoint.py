#!/usr/bin/env python3
"""
Simple script to test if the API endpoint is accessible
"""
import requests
import json

API_BASE_URL = "https://pap.zengenti.com/"

def test_endpoint_health():
    """Test if the API endpoint is accessible"""
    try:
        # Test basic connectivity
        response = requests.get(API_BASE_URL, timeout=5)
        print(f"Base URL status: {response.status_code}")
        
        # Test authentication endpoint
        auth_url = API_BASE_URL.rstrip('/') + '/users/authenticate'
        print(f"Testing authentication endpoint: {auth_url}")
        
        # Try a simple GET request to see if the endpoint exists
        try:
            auth_response = requests.get(auth_url, timeout=5)
            print(f"Auth endpoint GET status: {auth_response.status_code}")
        except Exception as e:
            print(f"Auth endpoint GET failed: {e}")
        
        # Test execute_playbook endpoint
        playbook_url = API_BASE_URL.rstrip('/') + '/execute_playbook/'
        print(f"Testing playbook endpoint: {playbook_url}")
        
        try:
            playbook_response = requests.get(playbook_url, timeout=5)
            print(f"Playbook endpoint GET status: {playbook_response.status_code}")
        except Exception as e:
            print(f"Playbook endpoint GET failed: {e}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

def test_authentication(username, password):
    """Test authentication with provided credentials"""
    auth_url = API_BASE_URL.rstrip('/') + '/users/authenticate'
    
    payload = {
        "username": username,
        "password": password
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Testing authentication with username: {username}")
    print(f"Auth URL: {auth_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(auth_url, json=payload, headers=headers, timeout=10)
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code == 200:
            try:
                resp_json = response.json()
                token = resp_json.get("token")
                if token:
                    print(f"Authentication successful! Token: {token[:20]}...")
                    return token
                else:
                    print("Authentication failed: No token in response")
                    return None
            except:
                print("Could not parse response as JSON")
                return None
        else:
            print(f"Authentication failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

if __name__ == "__main__":
    print("API Endpoint Test Script")
    print("=" * 50)
    
    # Test basic connectivity
    print("1. Testing basic connectivity...")
    test_endpoint_health()
    
    print("\n2. Testing authentication...")
    username = input("Enter username: ").strip()
    password = input("Enter password: ").strip()
    
    if username and password:
        token = test_authentication(username, password)
        if token:
            print(f"\n3. Testing playbook execution with token...")
            # Test a simple playbook call
            playbook_url = API_BASE_URL.rstrip('/') + '/execute_playbook/'
            test_payload = {
                "alias": "test-alias",
                "playbook_name": "e-vmotion-server",
                "priority": "normal"
            }
            headers = {
                "Content-Type": "application/json",
                "X-Security-Token": token
            }
            
            print(f"Testing playbook execution...")
            print(f"URL: {playbook_url}")
            print(f"Payload: {json.dumps(test_payload, indent=2)}")
            
            try:
                response = requests.post(playbook_url, json=test_payload, headers=headers, timeout=10)
                print(f"Response status: {response.status_code}")
                print(f"Response text: {response.text}")
            except Exception as e:
                print(f"Playbook execution error: {e}")
    else:
        print("No credentials provided. Skipping authentication test.") 
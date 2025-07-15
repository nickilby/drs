#!/usr/bin/env python3
"""
Debug script to test the API call independently
"""
import requests
import json

API_BASE_URL = "https://pap.zengenti.com/"

def test_api_call(alias, token, playbook_name="e-vmotion-server", priority="normal", use_limit=True, affected_vms=None):
    """Test API call with different configurations"""
    endpoint = API_BASE_URL.rstrip('/') + '/execute_playbook/'
    
    if use_limit and affected_vms:
        payload = {
            "alias": alias,
            "playbook_name": playbook_name,
            "priority": priority,
            "options": {
                "limit": affected_vms
            }
        }
    else:
        payload = {
            "alias": alias,
            "playbook_name": playbook_name,
            "priority": priority
        }
    
    headers = {
        "Content-Type": "application/json",
        "X-Security-Token": token
    }
    
    print(f"Testing API call to: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print("-" * 50)
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            try:
                resp_json = response.json()
                print(f"Response JSON: {json.dumps(resp_json, indent=2)}")
            except:
                print("Could not parse response as JSON")
        else:
            print(f"Error response: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

def test_wildcard_api_call(alias, token, playbook_name="e-vmotion-server", priority="normal", wildcard_type="star"):
    """Test API call with wildcard approaches"""
    endpoint = API_BASE_URL.rstrip('/') + '/execute_playbook/'
    
    if wildcard_type == "star":
        limit_value = ["*"]
    elif wildcard_type == "empty":
        limit_value = []
    elif wildcard_type == "all":
        limit_value = ["all"]
    elif wildcard_type == "asterisk":
        limit_value = ["*"]
    else:
        limit_value = ["*"]
    
    payload = {
        "alias": alias,
        "playbook_name": playbook_name,
        "priority": priority,
        "options": {
            "limit": limit_value
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Security-Token": token
    }
    
    print(f"Testing wildcard API call ({wildcard_type}) to: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print("-" * 50)
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            try:
                resp_json = response.json()
                print(f"Response JSON: {json.dumps(resp_json, indent=2)}")
                return True
            except:
                print("Could not parse response as JSON")
                return False
        else:
            print(f"Error response: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        return False

if __name__ == "__main__":
    # You'll need to provide a valid token and alias for testing
    print("API Debug Test Script")
    print("=" * 50)
    
    # Test with the specific case from the error
    alias = "vlse-dev"
    token = input("Enter your authentication token: ").strip()
    
    if not token:
        print("No token provided. Exiting.")
        exit(1)
    
    affected_vms = ["z-vlse-dev-LB1", "z-vlse-dev-LB2"]
    
    print("\n1. Testing with limit (original function):")
    test_api_call(alias, token, "e-vmotion-server", "normal", True, affected_vms)
    
    print("\n2. Testing without limit (new alias function):")
    test_api_call(alias, token, "e-vmotion-server", "normal", False)
    
    print("\n3. Testing with storage playbook:")
    test_api_call(alias, token, "e-vmotion-storage", "normal", True, affected_vms)
    
    print("\n4. Testing wildcard approaches:")
    print("\n4a. Testing with limit: ['*']")
    success_star = test_wildcard_api_call(alias, token, "e-vmotion-server", "normal", "star")
    
    print("\n4b. Testing with limit: [] (empty array)")
    success_empty = test_wildcard_api_call(alias, token, "e-vmotion-server", "normal", "empty")
    
    print("\n4c. Testing with limit: ['all']")
    success_all = test_wildcard_api_call(alias, token, "e-vmotion-server", "normal", "all")
    
    print("\n" + "=" * 50)
    print("WILDCARD TEST RESULTS:")
    print(f"['*'] approach: {'SUCCESS' if success_star else 'FAILED'}")
    print(f"[] approach: {'SUCCESS' if success_empty else 'FAILED'}")
    print(f"['all'] approach: {'SUCCESS' if success_all else 'FAILED'}")
    print("=" * 50) 
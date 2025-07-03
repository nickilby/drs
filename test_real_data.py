#!/usr/bin/env python3
"""
Test with real data scenario to see why violations aren't being detected
"""

from vcenter_drs.rules.rules_engine import extract_pool_from_dataset, parse_alias_and_role
from vcenter_drs.db.metrics_db import MetricsDB
from vcenter_drs.rules.rules_engine import evaluate_rules

def test_with_real_data():
    """Test with the actual database data"""
    
    print("Testing with Real Database Data")
    print("=" * 50)
    
    # Run the actual compliance check
    print("Running compliance check...")
    violations = evaluate_rules(return_structured=True)
    
    if violations:
        print(f"\nFound {len(violations)} violations:")
        for i, violation in enumerate(violations, 1):
            print(f"\nViolation {i}:")
            print(f"Type: {violation['type']}")
            print(f"Rule: {violation['rule']}")
            print(f"Affected VMs: {violation['affected_vms']}")
            print(f"Cluster: {violation['cluster']}")
            print(f"Text: {violation['violation_text']}")
    else:
        print("No violations found!")
    
    print("\n" + "=" * 50)
    print("Checking specific VMs...")
    
    # Check the specific VMs you mentioned
    db = MetricsDB()
    db.connect()
    cursor = db.conn.cursor(dictionary=True)
    
    # Look for VMs with similar names
    cursor.execute("""
        SELECT v.name, v.host_id, d.name as dataset_name, h.name as host_name
        FROM vms v
        LEFT JOIN datasets d ON v.dataset_id = d.id
        LEFT JOIN hosts h ON v.host_id = h.id
        WHERE v.name LIKE '%bcc%' OR v.name LIKE '%WEB%'
        ORDER BY v.name
    """)
    
    vms = cursor.fetchall()
    
    print(f"Found {len(vms)} VMs with 'bcc' or 'WEB' in name:")
    for vm in vms:
        pool = extract_pool_from_dataset(vm['dataset_name'])
        alias, role = parse_alias_and_role(vm['name'])
        print(f"  {vm['name']} -> Pool: {pool}, Alias: {alias}, Role: {role}, Dataset: {vm['dataset_name']}")
    
    cursor.close()
    db.close()

if __name__ == "__main__":
    test_with_real_data() 
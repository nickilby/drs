from vcenter_drs.db.metrics_db import MetricsDB

test_exception = {
    'rule_type': 'affinity',
    'alias': 'testalias',
    'affected_vms': ['vm1', 'vm2'],
    'cluster': 'testcluster',
    'rule_hash': 'testhash',
    'reason': 'test insert from standalone script'
}

db = MetricsDB()
db.connect()
db.add_exception(test_exception)
db.close()
print("Inserted test exception.") 
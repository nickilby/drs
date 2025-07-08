#!/usr/bin/env python3
"""
Standalone script to check compliance and update Prometheus metrics.
Can be called by cron to keep metrics updated in the background.
"""

import sys
import os
import time

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Set up environment variables for virtual environment
os.environ['PYTHONPATH'] = f"{project_dir}/venv/lib/python3.12/site-packages"
os.environ['VIRTUAL_ENV'] = f"{project_dir}/venv"

try:
    from vcenter_drs.rules.rules_engine import evaluate_rules, get_db_state
    from prometheus_client import Gauge, Histogram, REGISTRY
    from collections import defaultdict
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vcenter-drs-compliance.log'),
            logging.StreamHandler()
        ]
    )
    
    def main():
        """Main function to check compliance and update metrics"""
        start_time = time.time()
        logging.info("Starting compliance check...")
        
        try:
            # Run compliance check with structured output
            structured_violations = evaluate_rules(None, return_structured=True)
            
            # Load VM power status from DB
            _, _, vms_db = get_db_state()
            vm_power_status = {vm['name']: str(vm.get('power_status') or '').lower() for vm in vms_db.values()}
            
            # Update Prometheus metrics
            try:
                # Get or create metrics using the default registry
                from prometheus_client import REGISTRY
                try:
                    # Try to get existing metrics from registry
                    RULE_VIOLATIONS = REGISTRY._names_to_collectors['vcenter_drs_rule_violations_total']
                    COMPLIANCE_CHECK_DURATION = REGISTRY._names_to_collectors['vcenter_drs_compliance_check_duration_seconds']
                except (KeyError, AttributeError):
                    # Create new metrics if they don't exist
                    RULE_VIOLATIONS = Gauge('vcenter_drs_rule_violations_total', 'Current rule violations by type', ['rule_type'])
                    COMPLIANCE_CHECK_DURATION = Histogram('vcenter_drs_compliance_check_duration_seconds', 'Duration of compliance checks')
                
                # Record compliance check duration
                duration = time.time() - start_time
                COMPLIANCE_CHECK_DURATION.observe(duration)
                
                # Count violations by rule type (only if at least one affected VM is powered on)
                violation_counts = defaultdict(int)
                if structured_violations:
                    for violation in structured_violations:
                        affected_vms = violation.get('affected_vms', [])
                        any_powered_on = any(vm_power_status.get(vm_name, '') == 'poweredon' for vm_name in affected_vms)
                        if any_powered_on:
                            rule_type = violation.get('type', 'unknown')
                            violation_counts[rule_type] += 1
                
                # Always ensure metrics exist by setting all known rule types to 0 first
                known_rule_types = ['anti-affinity', 'dataset-affinity', 'dataset-anti-affinity', 'affinity', 'pool-anti-affinity']
                for rule_type in known_rule_types:
                    RULE_VIOLATIONS.labels(rule_type=rule_type).set(0)
                
                # Then set the actual counts
                for rule_type, count in violation_counts.items():
                    RULE_VIOLATIONS.labels(rule_type=rule_type).set(count)
                
                total_violations = sum(violation_counts.values())
                logging.info(f"Compliance check completed: {total_violations} total violations found (powered on VMs only)")
                logging.info(f"Violations by type: {dict(violation_counts)}")
                
            except Exception as e:
                logging.warning(f"Could not update Prometheus metrics: {e}")
            
            logging.info(f"Compliance check completed successfully in {duration:.2f} seconds")
                
        except Exception as e:
            logging.error(f"Failed to check compliance: {e}")
            sys.exit(1)
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project directory with the virtual environment activated")
    sys.exit(1) 
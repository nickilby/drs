#!/usr/bin/env python3
"""
Standalone script to refresh vCenter data.
Can be called by cron or systemd timer.
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
    from vcenter_drs.api.collect_and_store_metrics import main as collect_and_store_metrics_main
    from vcenter_drs.rules.rules_engine import get_db_state
    from prometheus_client import Gauge, REGISTRY
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/var/log/vcenter-drs-refresh.log'),
            logging.StreamHandler()
        ]
    )
    
    def main():
        """Main function to refresh vCenter data"""
        start_time = time.time()
        logging.info("Starting vCenter data refresh...")
        
        try:
            # Collect and store metrics
            collect_and_store_metrics_main()
            
            # Update Prometheus metrics if available
            try:
                # Get VM and host counts
                clusters, hosts, vms = get_db_state()
                
                # Update metrics (these will be created if they don't exist)
                VM_COUNT = Gauge('vcenter_drs_vm_count', 'Total number of VMs monitored')
                HOST_COUNT = Gauge('vcenter_drs_host_count', 'Total number of hosts monitored')
                LAST_COLLECTION_TIME = Gauge('vcenter_drs_last_collection_timestamp', 'Timestamp of last metrics collection')
                
                VM_COUNT.set(len(vms))
                HOST_COUNT.set(len(hosts))
                LAST_COLLECTION_TIME.set(time.time())
                
                logging.info(f"Updated metrics: {len(vms)} VMs, {len(hosts)} hosts")
                
            except Exception as e:
                logging.warning(f"Could not update Prometheus metrics: {e}")
            
            duration = time.time() - start_time
            logging.info(f"vCenter data refresh completed successfully in {duration:.2f} seconds")
            
            # Write duration to file for Streamlit app
            with open(os.path.join(project_dir, "last_collection_time.txt"), "w") as f:
                f.write(str(duration))
                
        except Exception as e:
            logging.error(f"Failed to refresh vCenter data: {e}")
            sys.exit(1)
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project directory with the virtual environment activated")
    sys.exit(1) 
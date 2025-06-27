import sys
import time
from api.vcenter_client_pyvomi import VCenterPyVmomiClient
from api.collect_and_store_metrics import main as collect_and_store_metrics_main

def check_connectivity():
    client = VCenterPyVmomiClient()
    try:
        si = client.connect()
        print("Successfully connected to vCenter.")
        print("vCenter server time:", si.CurrentTime())
    except Exception as e:
        print(f"Failed to connect to vCenter: {e}")
    finally:
        client.disconnect()

def timed_data_collection():
    start = time.time()
    collect_and_store_metrics_main()
    end = time.time()
    duration = end - start
    with open("last_collection_time.txt", "w") as f:
        f.write(str(duration))
    return duration

if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_connectivity()
    else:
        collect_and_store_metrics_main() 
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    duration = end - start
    with open("last_collection_time.txt", "w") as f:
        f.write(str(duration))
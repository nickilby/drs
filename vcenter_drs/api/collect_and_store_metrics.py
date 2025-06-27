from api.vcenter_client_pyvomi import VCenterPyVmomiClient
from pyVmomi import vim
from db.metrics_db import MetricsDB
from datetime import datetime

def get_perf_counter_id(perf_manager, counter_name):
    for counter in perf_manager.perfCounter:
        full_name = f"{counter.groupInfo.key}.{counter.nameInfo.key}.{counter.rollupType}"
        if full_name == counter_name:
            return counter.key
    return None

def get_latest_metric(perf_manager, entity, counter_id, instance=""): 
    query = vim.PerformanceManager.QuerySpec(
        entity=entity,
        metricId=[vim.PerformanceManager.MetricId(counterId=counter_id, instance=instance)],
        intervalId=20,  # 20 = real-time
        maxSample=1
    )
    results = perf_manager.QueryStats(querySpec=[query])
    if results and results[0].value and results[0].value[0].value:
        return results[0].value[0].value[0]
    return None

def get_or_create(cursor, table, name, extra_fields=None):
    # Try to get the id, or insert and return new id
    query = f"SELECT id FROM {table} WHERE name = %s"
    cursor.execute(query, (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    if extra_fields:
        fields = ', '.join(['name'] + list(extra_fields.keys()))
        placeholders = ', '.join(['%s'] * (1 + len(extra_fields)))
        values = [name] + list(extra_fields.values())
        insert = f"INSERT INTO {table} ({fields}) VALUES ({placeholders})"
        cursor.execute(insert, values)
    else:
        insert = f"INSERT INTO {table} (name) VALUES (%s)"
        cursor.execute(insert, (name,))
    return cursor.lastrowid

def cleanup_stale_vms(cursor, current_vm_names):
    """
    Remove VMs from the database that no longer exist in vCenter
    """
    # Get all VMs currently in the database
    cursor.execute("SELECT id, name FROM vms")
    db_vms = cursor.fetchall()
    
    # Find VMs that exist in DB but not in vCenter
    stale_vm_ids = []
    for vm_id, vm_name in db_vms:
        if vm_name not in current_vm_names:
            stale_vm_ids.append(vm_id)
    
    if stale_vm_ids:
        print(f"Found {len(stale_vm_ids)} stale VMs to remove")
        
        # Remove metrics for stale VMs first (due to foreign key constraints)
        placeholders = ','.join(['%s'] * len(stale_vm_ids))
        cursor.execute(
            f"DELETE FROM metrics WHERE object_type = 'vm' AND object_id IN ({placeholders})",
            stale_vm_ids
        )
        print(f"Removed metrics for {cursor.rowcount} stale VMs")
        
        # Remove the stale VMs
        cursor.execute(
            f"DELETE FROM vms WHERE id IN ({placeholders})",
            stale_vm_ids
        )
        print(f"Removed {cursor.rowcount} stale VMs from database")
    else:
        print("No stale VMs found")

def main():
    # Connect to vCenter
    client = VCenterPyVmomiClient()
    si = client.connect()
    content = si.RetrieveContent()
    perf_manager = content.perfManager

    # Connect to MySQL
    db = MetricsDB()
    db.connect()
    db.init_schema()
    cursor = db.conn.cursor()

    # Get performance counter IDs
    cpu_id = get_perf_counter_id(perf_manager, "cpu.usage.average")
    mem_id = get_perf_counter_id(perf_manager, "mem.usage.average")
    now = datetime.now()

    # Track all current VMs for cleanup
    current_vm_names = set()

    for dc in content.rootFolder.childEntity:
        # Handle clusters
        if hasattr(dc, 'hostFolder'):
            for cluster in dc.hostFolder.childEntity:
                if hasattr(cluster, 'name'):
                    cluster_id = get_or_create(cursor, 'clusters', cluster.name)
                    # Handle hosts
                    for host in getattr(cluster, 'host', []):
                        host_id = get_or_create(cursor, 'hosts', host.name, {'cluster_id': cluster_id})
                        # Host metrics
                        cpu = get_latest_metric(perf_manager, host, cpu_id)
                        mem = get_latest_metric(perf_manager, host, mem_id)
                        for metric_name, value in [("cpu.usage.average", cpu), ("mem.usage.average", mem)]:
                            if value is not None:
                                cursor.execute(
                                    "INSERT INTO metrics (object_type, object_id, metric_name, value, timestamp) VALUES (%s, %s, %s, %s, %s)",
                                    ("host", host_id, metric_name, value, now)
                                )
                        # Handle VMs
                        for vm in getattr(host, 'vm', []):
                            current_vm_names.add(vm.name)  # Track current VM
                            cursor.execute(
                                "SELECT id, host_id FROM vms WHERE name = %s", (vm.name,)
                            )
                            row = cursor.fetchone()
                            if row:
                                vm_id, old_host_id = row
                                if old_host_id != host_id:
                                    cursor.execute(
                                        "UPDATE vms SET host_id = %s WHERE id = %s", (host_id, vm_id)
                                    )
                            else:
                                cursor.execute(
                                    "INSERT INTO vms (name, host_id) VALUES (%s, %s)", (vm.name, host_id)
                                )
                                vm_id = cursor.lastrowid
                            cpu = get_latest_metric(perf_manager, vm, cpu_id)
                            mem = get_latest_metric(perf_manager, vm, mem_id)
                            for metric_name, value in [("cpu.usage.average", cpu), ("mem.usage.average", mem)]:
                                if value is not None:
                                    cursor.execute(
                                        "INSERT INTO metrics (object_type, object_id, metric_name, value, timestamp) VALUES (%s, %s, %s, %s, %s)",
                                        ("vm", vm_id, metric_name, value, now)
                                    )
    
    # Clean up stale VMs
    print(f"Found {len(current_vm_names)} VMs currently in vCenter")
    cleanup_stale_vms(cursor, current_vm_names)
    
    db.conn.commit()
    cursor.close()
    db.close()
    client.disconnect()
    print("Metrics collection and storage complete.")

if __name__ == "__main__":
    main() 
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
                            vm_id = get_or_create(cursor, 'vms', vm.name, {'host_id': host_id})
                            cpu = get_latest_metric(perf_manager, vm, cpu_id)
                            mem = get_latest_metric(perf_manager, vm, mem_id)
                            for metric_name, value in [("cpu.usage.average", cpu), ("mem.usage.average", mem)]:
                                if value is not None:
                                    cursor.execute(
                                        "INSERT INTO metrics (object_type, object_id, metric_name, value, timestamp) VALUES (%s, %s, %s, %s, %s)",
                                        ("vm", vm_id, metric_name, value, now)
                                    )
    db.conn.commit()
    cursor.close()
    db.close()
    client.disconnect()
    print("Metrics collection and storage complete.")

if __name__ == "__main__":
    main() 
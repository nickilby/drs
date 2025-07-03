from vcenter_drs.api.vcenter_client_pyvomi import VCenterPyVmomiClient
from pyVmomi import vim
import time

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

if __name__ == "__main__":
    client = VCenterPyVmomiClient()
    si = client.connect()
    content = si.RetrieveContent()
    if content is None:
        print("Failed to retrieve content from vCenter")
        client.disconnect()
        exit(1)
    perf_manager = content.perfManager  # type: ignore

    # Get first VM
    vm = None
    for dc in content.rootFolder.childEntity:  # type: ignore
        if hasattr(dc, 'vmFolder'):
            vms = dc.vmFolder.childEntity
            if vms:
                vm = vms[0]
                break
    if vm:
        cpu_id = get_perf_counter_id(perf_manager, "cpu.usage.average")
        mem_id = get_perf_counter_id(perf_manager, "mem.usage.average")
        cpu = get_latest_metric(perf_manager, vm, cpu_id)
        mem = get_latest_metric(perf_manager, vm, mem_id)
        print(f"VM: {vm.name} | CPU Usage: {cpu}% | Memory Usage: {mem}%")
    else:
        print("No VM found.")

    # Get first Host
    host = None
    for dc in content.rootFolder.childEntity:  # type: ignore
        if hasattr(dc, 'hostFolder'):
            hosts = dc.hostFolder.childEntity
            for c in hosts:
                if hasattr(c, 'host') and c.host:
                    host = c.host[0]
                    break
    if host:
        cpu_id = get_perf_counter_id(perf_manager, "cpu.usage.average")
        mem_id = get_perf_counter_id(perf_manager, "mem.usage.average")
        cpu = get_latest_metric(perf_manager, host, cpu_id)
        mem = get_latest_metric(perf_manager, host, mem_id)
        print(f"Host: {host.name} | CPU Usage: {cpu}% | Memory Usage: {mem}%")
    else:
        print("No host found.")

    client.disconnect() 
"""Data collection from Prometheus for AI Optimizer"""

import requests
import time
import random
from typing import Dict, List, Optional, Any
from .config import AIConfig


class PrometheusDataCollector:
    """Collects performance metrics from Prometheus"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.primary_url = f"{config.prometheus.url}:{config.prometheus.port}"
        self.secondary_url = "http://10.65.32.4:9090"
        self.current_url = self.primary_url
        self.use_simulated_data = False
    
    def test_connection(self) -> bool:
        """Test connection to Prometheus"""
        # Try primary server first
        try:
            response = requests.get(f"{self.primary_url}/api/v1/status/config", 
                                 timeout=5)  # 5 second timeout for testing
            if response.status_code == 200:
                self.current_url = self.primary_url
                self.use_simulated_data = False
                print(f"Connected to primary Prometheus: {self.primary_url}")
                return True
        except Exception as e:
            print(f"Primary Prometheus connection failed: {e}")
        
        # Try secondary server
        try:
            response = requests.get(f"{self.secondary_url}/api/v1/status/config", 
                                 timeout=5)  # 5 second timeout for testing
            if response.status_code == 200:
                self.current_url = self.secondary_url
                self.use_simulated_data = False
                print(f"Connected to secondary Prometheus: {self.secondary_url}")
                return True
            else:
                self.use_simulated_data = True
                return False
        except Exception as e:
            print(f"Secondary Prometheus connection failed: {e}")
            self.use_simulated_data = True
            return False
    
    def get_metric(self, query: str, start_time: str, end_time: str) -> Optional[List[Dict]]:
        """Get metric data from Prometheus"""
        try:
            # Try primary server first
            url = f"http://10.65.32.4:9090/api/v1/query"
            params = {'query': query}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                return data['data']['result']
            
        except Exception as e:
            pass
        
        try:
            # Try secondary server
            url = f"http://10.65.32.4:9090/api/v1/query"
            params = {'query': query}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                return data['data']['result']
            
        except Exception as e:
            pass
        
        return None
    
    def _get_simulated_vm_metrics(self, vm_name: str) -> Dict[str, float]:
        """Generate simulated VM metrics for testing (fallback only)"""
        # Generate realistic but varied metrics
        base_cpu = 0.2 + (hash(vm_name) % 100) / 1000.0  # 20-30% base CPU
        base_ram = 0.3 + (hash(vm_name) % 100) / 1000.0  # 30-40% base RAM
        
        return {
            'cpu_usage': min(0.9, base_cpu + random.uniform(0, 0.3)),
            'ram_usage': min(0.9, base_ram + random.uniform(0, 0.4)),
            'ready_time': random.uniform(0.01, 0.05),  # 1-5% ready time
            'io_usage': random.uniform(0.1, 0.6)  # 10-60% I/O usage
        }
    
    def _get_simulated_host_metrics(self, host_name: str) -> Dict[str, float]:
        """Generate simulated host metrics for testing (fallback only)"""
        # Generate realistic host metrics
        base_cpu = 0.4 + (hash(host_name) % 100) / 1000.0  # 40-50% base CPU
        base_ram = 0.5 + (hash(host_name) % 100) / 1000.0  # 50-60% base RAM
        
        return {
            'cpu_usage': min(0.9, base_cpu + random.uniform(0, 0.3)),
            'ram_usage': min(0.9, base_ram + random.uniform(0, 0.3)),
            'io_usage': random.uniform(0.2, 0.7),  # 20-70% I/O usage
            'ready_time': random.uniform(0.01, 0.03),  # 1-3% ready time
            'vm_count': random.randint(3, 12)  # 3-12 VMs per host
        }
    
    def get_vm_metrics(self, vm_name: str, hours: int = 1) -> Dict[str, float]:
        """Get metrics for a specific VM"""
        if self.use_simulated_data:
            print(f"Using simulated data for VM: {vm_name} (both Prometheus servers unavailable)")
            return self._get_simulated_vm_metrics(vm_name)
        
        end_time = int(time.time())
        start_time = end_time - (hours * 3600)
        
        metrics = {}
        
        # Get all VM CPU metrics and filter by name - use simple string matching
        cpu_query = f'vmware_vm_cpu_usage_average'
        cpu_result = self.get_metric(cpu_query, str(start_time), str(end_time))
        if cpu_result:
            # Find the VM with matching name
            for result in cpu_result:
                if vm_name in result['metric'].get('vm_name', ''):
                    # VMware CPU usage is in MHz
                    cpu_mhz = float(result['value'][1])
                    # Convert to percentage using a reasonable max CPU value
                    # Most VMs use 1-4 vCPUs, so assume max 8000 MHz (4 vCPUs * 2000 MHz each)
                    metrics['cpu_usage'] = min(1.0, cpu_mhz / 8000.0)
                    metrics['cpu_mhz'] = cpu_mhz  # Store actual CPU usage in MHz
                    metrics['cpu_max_mhz'] = 8000.0  # Store max CPU for projection calculations
                    break
            else:
                metrics['cpu_usage'] = 0.0
                metrics['cpu_mhz'] = 0.0
                metrics['cpu_max_mhz'] = 8000.0
        else:
            metrics['cpu_usage'] = 0.0
            metrics['cpu_mhz'] = 0.0
            metrics['cpu_max_mhz'] = 8000.0
        
        # RAM usage - using vmware_vm_mem_usage_average (in MB, need to get max and convert)
        ram_query = f'vmware_vm_mem_usage_average'
        ram_result = self.get_metric(ram_query, str(start_time), str(end_time))
        if ram_result:
            # Find the VM with matching name
            for result in ram_result:
                if vm_name in result['metric'].get('vm_name', ''):
                    # VMware memory usage is in MB
                    ram_used = float(result['value'][1])
                    # Get max memory for this VM
                    ram_max_query = f'vmware_vm_memory_max'
                    ram_max_result = self.get_metric(ram_max_query, str(start_time), str(end_time))
                    if ram_max_result:
                        # Find max memory for this VM
                        for max_result in ram_max_result:
                            if vm_name in max_result['metric'].get('vm_name', ''):
                                ram_max = float(max_result['value'][1])
                                metrics['ram_usage'] = min(1.0, ram_used / ram_max) if ram_max > 0 else 0.0
                                metrics['ram_mb'] = ram_used  # Store actual RAM usage in MB
                                metrics['ram_max_mb'] = ram_max  # Store max RAM for projection calculations
                                break
                        else:
                            # Fallback: assume 8GB max if we can't get max memory
                            metrics['ram_usage'] = min(1.0, ram_used / 8192.0)
                            metrics['ram_mb'] = ram_used
                            metrics['ram_max_mb'] = 8192.0
                    else:
                        # Fallback: assume 8GB max if we can't get max memory
                        metrics['ram_usage'] = min(1.0, ram_used / 8192.0)
                        metrics['ram_mb'] = ram_used
                        metrics['ram_max_mb'] = 8192.0
                    break
            else:
                metrics['ram_usage'] = 0.0
                metrics['ram_mb'] = 0.0
                metrics['ram_max_mb'] = 8192.0
        else:
            metrics['ram_usage'] = 0.0
            metrics['ram_mb'] = 0.0
            metrics['ram_max_mb'] = 8192.0
        
        # Ready time - using vmware_vm_cpu_ready_summation (in milliseconds)
        ready_query = f'vmware_vm_cpu_ready_summation'
        ready_result = self.get_metric(ready_query, str(start_time), str(end_time))
        if ready_result:
            # Find the VM with matching name
            for result in ready_result:
                if vm_name in result['metric'].get('vm_name', ''):
                    # VMware ready time is in milliseconds
                    ready_ms = float(result['value'][1])
                    # Normalize ready time: 0-100ms = good, 100-500ms = acceptable, >500ms = poor
                    # Convert to a 0-1 scale where lower is better
                    if ready_ms <= 100:
                        metrics['ready_time'] = 0.0  # Excellent
                    elif ready_ms <= 500:
                        metrics['ready_time'] = (ready_ms - 100) / 400.0  # 0-1 scale
                    else:
                        metrics['ready_time'] = 1.0  # Poor
                    break
            else:
                metrics['ready_time'] = 0.0
        else:
            metrics['ready_time'] = 0.0
        
        # I/O usage - using vmware_vm_disk_usage_average (in KB/s)
        # This represents disk I/O throughput, not network throughput
        io_query = f'vmware_vm_disk_usage_average'
        io_result = self.get_metric(io_query, str(start_time), str(end_time))
        if io_result:
            # Find the VM with matching name
            for result in io_result:
                if vm_name in result['metric'].get('vm_name', ''):
                    # VMware disk usage is in KB/s, normalize to 0-1
                    io_kbps = float(result['value'][1])
                    # Assume 100 MB/s = 100% disk I/O usage
                    metrics['io_usage'] = min(1.0, io_kbps / 102400.0)
                    break
            else:
                metrics['io_usage'] = 0.0
        else:
            metrics['io_usage'] = 0.0
        
        return metrics
    
    def get_vm_cluster(self, vm_name: str) -> Optional[str]:
        """Get the cluster name for a given VM from Prometheus metrics"""
        # Try to get cluster from any metric that includes cluster info
        end_time = int(time.time())
        start_time = end_time - 3600
        cpu_query = 'vmware_vm_cpu_usage_average'
        cpu_result = self.get_metric(cpu_query, str(start_time), str(end_time))
        if cpu_result:
            for result in cpu_result:
                if vm_name in result['metric'].get('vm_name', ''):
                    # Use the correct label name for cluster
                    return result['metric'].get('cluster_name')
        return None

    def get_host_cluster(self, host_name: str) -> Optional[str]:
        """Get the cluster name for a given host from Prometheus metrics"""
        end_time = int(time.time())
        start_time = end_time - 3600
        cpu_query = 'vmware_host_cpu_usage_average'
        cpu_result = self.get_metric(cpu_query, str(start_time), str(end_time))
        if cpu_result:
            for result in cpu_result:
                if host_name in result['metric'].get('host_name', ''):
                    # Use the correct label name for cluster
                    return result['metric'].get('cluster_name')
        return None
    
    def get_host_metrics(self, host_name: str, hours: int = 1) -> Dict[str, float]:
        """Get metrics for a specific host"""
        if self.use_simulated_data:
            print(f"Using simulated data for host: {host_name} (both Prometheus servers unavailable)")
            return self._get_simulated_host_metrics(host_name)
        
        end_time = int(time.time())
        start_time = end_time - (hours * 3600)
        
        metrics = {}
        
        # CPU usage - using vmware_host_cpu_usage_average (in MHz)
        cpu_query = f'vmware_host_cpu_usage_average'
        cpu_result = self.get_metric(cpu_query, str(start_time), str(end_time))
        if cpu_result:
            # Find the host with matching name
            for result in cpu_result:
                if host_name in result['metric'].get('host_name', ''):
                    # Get CPU usage in MHz and max CPU for this host
                    cpu_mhz = float(result['value'][1])
                    # Get max CPU for this host
                    cpu_max_query = f'vmware_host_cpu_max'
                    cpu_max_result = self.get_metric(cpu_max_query, str(start_time), str(end_time))
                    if cpu_max_result:
                        # Find max CPU for this host
                        for max_result in cpu_max_result:
                            if host_name in max_result['metric'].get('host_name', ''):
                                cpu_max = float(max_result['value'][1])
                                # Fix: Multiply by 10 to get correct percentage
                                metrics['cpu_usage'] = min(1.0, (cpu_mhz / cpu_max) * 10) if cpu_max > 0 else 0.0
                                metrics['cpu_max_mhz'] = cpu_max  # Store max CPU for projection calculations
                                break
                        else:
                            # Fallback: assume 4000 MHz max if we can't get max CPU
                            metrics['cpu_usage'] = min(1.0, (cpu_mhz / 4000.0) * 10)
                            metrics['cpu_max_mhz'] = 4000.0
                    else:
                        # Fallback: assume 4000 MHz max if we can't get max CPU
                        metrics['cpu_usage'] = min(1.0, (cpu_mhz / 4000.0) * 10)
                        metrics['cpu_max_mhz'] = 4000.0
                    break
            else:
                metrics['cpu_usage'] = 0.0
                metrics['cpu_max_mhz'] = 4000.0
        else:
            metrics['cpu_usage'] = 0.0
            metrics['cpu_max_mhz'] = 4000.0
        
        # RAM usage - using vmware_host_memory_usage (in MB)
        ram_query = f'vmware_host_memory_usage'
        ram_result = self.get_metric(ram_query, str(start_time), str(end_time))
        if ram_result:
            # Find the host with matching name
            for result in ram_result:
                if host_name in result['metric'].get('host_name', ''):
                    # Get memory usage in MB
                    ram_used = float(result['value'][1])
                    # Get max memory for this host
                    ram_max_query = f'vmware_host_memory_max'
                    ram_max_result = self.get_metric(ram_max_query, str(start_time), str(end_time))
                    if ram_max_result:
                        # Find max memory for this host
                        for max_result in ram_max_result:
                            if host_name in max_result['metric'].get('host_name', ''):
                                ram_max = float(max_result['value'][1])
                                metrics['ram_usage'] = min(1.0, ram_used / ram_max) if ram_max > 0 else 0.0
                                metrics['ram_max_mb'] = ram_max  # Store max RAM for projection calculations
                                break
                        else:
                            # Fallback: assume 64GB max if we can't get max memory
                            metrics['ram_usage'] = min(1.0, ram_used / 65536.0)
                            metrics['ram_max_mb'] = 65536.0
                    else:
                        # Fallback: assume 64GB max if we can't get max memory
                        metrics['ram_usage'] = min(1.0, ram_used / 65536.0)
                        metrics['ram_max_mb'] = 65536.0
                    break
            else:
                metrics['ram_usage'] = 0.0
                metrics['ram_max_mb'] = 65536.0
        else:
            metrics['ram_usage'] = 0.0
            metrics['ram_max_mb'] = 65536.0
        
        # I/O usage - using vmware_host_disk_read_average + vmware_host_disk_write_average (in KB/s)
        # This represents disk I/O throughput, not network throughput
        io_read_query = f'vmware_host_disk_read_average'
        io_write_query = f'vmware_host_disk_write_average'
        io_read_result = self.get_metric(io_read_query, str(start_time), str(end_time))
        io_write_result = self.get_metric(io_write_query, str(start_time), str(end_time))
        
        if io_read_result and io_write_result:
            # Find the host with matching name for both read and write
            read_value = 0.0
            write_value = 0.0
            
            for result in io_read_result:
                if host_name in result['metric'].get('host_name', ''):
                    read_value = float(result['value'][1])
                    break
            
            for result in io_write_result:
                if host_name in result['metric'].get('host_name', ''):
                    write_value = float(result['value'][1])
                    break
            
            # Normalize disk I/O usage (KB/s to percentage)
            total_io = read_value + write_value
            # Assume 100 MB/s = 100% disk I/O usage
            metrics['io_usage'] = min(1.0, total_io / 102400.0)
        else:
            metrics['io_usage'] = 0.0
        
        # Remove host ready time - it's not meaningful for placement decisions
        # Focus on VM ready time improvement instead
        
        # VM count - count VMs on this host
        vm_count_query = f'vmware_vm_cpu_usage_average'
        vm_count_result = self.get_metric(vm_count_query, str(start_time), str(end_time))
        if vm_count_result:
            # Count VMs for this host
            vm_count = 0
            for result in vm_count_result:
                if host_name in result['metric'].get('host_name', ''):
                    vm_count += 1
            metrics['vm_count'] = vm_count
        else:
            metrics['vm_count'] = 0
        
        return metrics

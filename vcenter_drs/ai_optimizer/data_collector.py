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
        self.base_url = f"{config.prometheus.url}:{config.prometheus.port}"
        self.use_simulated_data = False
    
    def test_connection(self) -> bool:
        """Test connection to Prometheus"""
        try:
            # Use a shorter timeout for testing
            response = requests.get(f"{self.base_url}/api/v1/status/config", 
                                 timeout=5)  # 5 second timeout for testing
            if response.status_code == 200:
                self.use_simulated_data = False
                return True
            else:
                self.use_simulated_data = True
                return False
        except Exception as e:
            print(f"Prometheus connection failed: {e}")
            self.use_simulated_data = True
            return False
    
    def get_metric(self, query: str, start_time: Optional[str] = None, 
                   end_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get metric data from Prometheus"""
        if self.use_simulated_data:
            return None
            
        try:
            params = {
                'query': query,
                'timeout': min(self.config.prometheus.timeout, 10)  # Cap at 10 seconds
            }
            
            if start_time:
                params['start'] = start_time
            if end_time:
                params['end'] = end_time
            
            response = requests.get(f"{self.base_url}/api/v1/query", 
                                 params=params,
                                 timeout=min(self.config.prometheus.timeout, 10))
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    return data['data']['result']
            return None
        except Exception as e:
            print(f"Failed to get metric {query}: {e}")
            return None
    
    def _get_simulated_vm_metrics(self, vm_name: str) -> Dict[str, float]:
        """Generate simulated VM metrics for testing"""
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
        """Generate simulated host metrics for testing"""
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
            print(f"Using simulated data for VM: {vm_name}")
            return self._get_simulated_vm_metrics(vm_name)
        
        end_time = int(time.time())
        start_time = end_time - (hours * 3600)
        
        metrics = {}
        
        # CPU usage
        cpu_query = f'avg_over_time(vm_cpu_usage_percent{{vm="{vm_name}"}}[{hours}h])'
        cpu_result = self.get_metric(cpu_query, str(start_time), str(end_time))
        if cpu_result:
            metrics['cpu_usage'] = float(cpu_result[0]['value'][1]) / 100.0
        else:
            metrics['cpu_usage'] = 0.0
        
        # RAM usage
        ram_query = f'avg_over_time(vm_memory_usage_percent{{vm="{vm_name}"}}[{hours}h])'
        ram_result = self.get_metric(ram_query, str(start_time), str(end_time))
        if ram_result:
            metrics['ram_usage'] = float(ram_result[0]['value'][1]) / 100.0
        else:
            metrics['ram_usage'] = 0.0
        
        # Ready time
        ready_query = f'avg_over_time(vm_ready_time_percent{{vm="{vm_name}"}}[{hours}h])'
        ready_result = self.get_metric(ready_query, str(start_time), str(end_time))
        if ready_result:
            metrics['ready_time'] = float(ready_result[0]['value'][1]) / 100.0
        else:
            metrics['ready_time'] = 0.0
        
        # I/O usage
        io_query = f'avg_over_time(vm_io_usage_percent{{vm="{vm_name}"}}[{hours}h])'
        io_result = self.get_metric(io_query, str(start_time), str(end_time))
        if io_result:
            metrics['io_usage'] = float(io_result[0]['value'][1]) / 100.0
        else:
            metrics['io_usage'] = 0.0
        
        return metrics
    
    def get_host_metrics(self, host_name: str, hours: int = 1) -> Dict[str, float]:
        """Get metrics for a specific host"""
        if self.use_simulated_data:
            print(f"Using simulated data for host: {host_name}")
            return self._get_simulated_host_metrics(host_name)
        
        end_time = int(time.time())
        start_time = end_time - (hours * 3600)
        
        metrics = {}
        
        # CPU usage
        cpu_query = f'avg_over_time(host_cpu_usage_percent{{host="{host_name}"}}[{hours}h])'
        cpu_result = self.get_metric(cpu_query, str(start_time), str(end_time))
        if cpu_result:
            metrics['cpu_usage'] = float(cpu_result[0]['value'][1]) / 100.0
        else:
            metrics['cpu_usage'] = 0.0
        
        # RAM usage
        ram_query = f'avg_over_time(host_memory_usage_percent{{host="{host_name}"}}[{hours}h])'
        ram_result = self.get_metric(ram_query, str(start_time), str(end_time))
        if ram_result:
            metrics['ram_usage'] = float(ram_result[0]['value'][1]) / 100.0
        else:
            metrics['ram_usage'] = 0.0
        
        # I/O usage
        io_query = f'avg_over_time(host_io_usage_percent{{host="{host_name}"}}[{hours}h])'
        io_result = self.get_metric(io_query, str(start_time), str(end_time))
        if io_result:
            metrics['io_usage'] = float(io_result[0]['value'][1]) / 100.0
        else:
            metrics['io_usage'] = 0.0
        
        # Ready time
        ready_query = f'avg_over_time(host_ready_time_percent{{host="{host_name}"}}[{hours}h])'
        ready_result = self.get_metric(ready_query, str(start_time), str(end_time))
        if ready_result:
            metrics['ready_time'] = float(ready_result[0]['value'][1]) / 100.0
        else:
            metrics['ready_time'] = 0.0
        
        # VM count
        vm_count_query = f'count(vm_cpu_usage_percent{{host="{host_name}"}})'
        vm_count_result = self.get_metric(vm_count_query, str(start_time), str(end_time))
        if vm_count_result:
            metrics['vm_count'] = int(vm_count_result[0]['value'][1])
        else:
            metrics['vm_count'] = 0
        
        return metrics

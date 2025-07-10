"""Data collection from Prometheus for AI Optimizer"""

import requests
import time
from typing import Dict, List, Optional, Any
from .config import AIConfig


class PrometheusDataCollector:
    """Collects performance metrics from Prometheus"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.base_url = f"{config.prometheus.url}:{config.prometheus.port}"
    
    def test_connection(self) -> bool:
        """Test connection to Prometheus"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/status/config", 
                                 timeout=self.config.prometheus.timeout)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_metric(self, query: str, start_time: Optional[str] = None, 
                   end_time: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get metric data from Prometheus"""
        try:
            params = {
                'query': query,
                'timeout': self.config.prometheus.timeout
            }
            
            if start_time:
                params['start'] = start_time
            if end_time:
                params['end'] = end_time
            
            response = requests.get(f"{self.base_url}/api/v1/query", 
                                 params=params,
                                 timeout=self.config.prometheus.timeout)
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    return data['data']['result']
            return None
        except Exception:
            return None
    
    def get_vm_metrics(self, vm_name: str, hours: int = 1) -> Dict[str, float]:
        """Get metrics for a specific VM"""
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

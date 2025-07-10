"""
Prometheus Data Collector for AI Optimizer

This module provides enhanced data collection from Prometheus for AI-driven
VM placement optimization. It collects performance metrics with different
time windows for trend analysis.
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from .config import AIConfig
from .exceptions import PrometheusConnectionError, DataCollectionError


class PrometheusDataCollector:
    """
    Enhanced Prometheus data collector for AI optimization.
    
    This class provides methods to collect performance metrics from Prometheus
    with different time windows for trend analysis and AI model training.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        """
        Initialize the Prometheus data collector.
        
        Args:
            config: AI configuration instance (uses global config if None)
        """
        self.config = config or AIConfig()
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        # Set timeout for individual requests instead of session
        self.timeout = self.config.prometheus.timeout
        
    def _query_prometheus(self, query: str, start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Execute a Prometheus query.
        
        Args:
            query: PromQL query string
            start_time: Start time for range query
            end_time: End time for range query
            
        Returns:
            Dict containing query results
            
        Raises:
            PrometheusConnectionError: If connection fails
        """
        url = f"{self.config.get_prometheus_url()}/api/v1/query"
        
        # Initialize query parameters
        query_params: Dict[str, Any] = {'query': query}
        
        if start_time and end_time:
            # Range query
            url = f"{self.config.get_prometheus_url()}/api/v1/query_range"
            query_params.update({
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': '60s'  # 1-minute intervals
            })
        
        for attempt in range(self.config.prometheus.retry_attempts):
            try:
                response = self.session.get(url, params=query_params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                if data.get('status') != 'success':
                    raise PrometheusConnectionError(
                        f"Prometheus query failed: {data.get('error', 'Unknown error')}"
                    )
                
                return data
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Prometheus query attempt {attempt + 1} failed: {e}")
                if attempt == self.config.prometheus.retry_attempts - 1:
                    raise PrometheusConnectionError(f"Failed to query Prometheus: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # This should never be reached, but satisfies the type checker
        raise PrometheusConnectionError("All retry attempts failed")
    
    def get_vm_cpu_trend(self, vm_name: str, hours: int = 1) -> List[Tuple[datetime, float]]:
        """
        Get CPU usage trend for a VM over specified hours.
        
        Args:
            vm_name: Name of the VM
            hours: Number of hours to analyze
            
        Returns:
            List of (timestamp, cpu_usage) tuples
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        query = f'vmware_vm_cpu_usage_average{{vm_name="{vm_name}"}}'
        
        try:
            data = self._query_prometheus(query, start_time, end_time)
            results = []
            
            if 'data' in data and 'result' in data['data']:
                for result in data['data']['result']:
                    if 'values' in result:
                        for timestamp, value in result['values']:
                            dt = datetime.fromtimestamp(timestamp)
                            results.append((dt, float(value)))
            
            return sorted(results, key=lambda x: x[0])
            
        except Exception as e:
            self.logger.error(f"Failed to get CPU trend for VM {vm_name}: {e}")
            return []
    
    def get_vm_ram_trend(self, vm_name: str, hours: int = 6) -> List[Tuple[datetime, float]]:
        """
        Get RAM usage trend for a VM over specified hours.
        
        Args:
            vm_name: Name of the VM
            hours: Number of hours to analyze
            
        Returns:
            List of (timestamp, ram_usage) tuples
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        query = f'vmware_vm_memory_usage_average{{vm_name="{vm_name}"}}'
        
        try:
            data = self._query_prometheus(query, start_time, end_time)
            results = []
            
            if 'data' in data and 'result' in data['data']:
                for result in data['data']['result']:
                    if 'values' in result:
                        for timestamp, value in result['values']:
                            dt = datetime.fromtimestamp(timestamp)
                            results.append((dt, float(value)))
            
            return sorted(results, key=lambda x: x[0])
            
        except Exception as e:
            self.logger.error(f"Failed to get RAM trend for VM {vm_name}: {e}")
            return []
    
    def get_vm_ready_time_trend(self, vm_name: str, hours: int = 1) -> List[Tuple[datetime, float]]:
        """
        Get CPU ready time trend for a VM over specified hours.
        
        Args:
            vm_name: Name of the VM
            hours: Number of hours to analyze
            
        Returns:
            List of (timestamp, ready_time) tuples
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        query = f'vmware_vm_cpu_ready_average{{vm_name="{vm_name}"}}'
        
        try:
            data = self._query_prometheus(query, start_time, end_time)
            results = []
            
            if 'data' in data and 'result' in data['data']:
                for result in data['data']['result']:
                    if 'values' in result:
                        for timestamp, value in result['values']:
                            dt = datetime.fromtimestamp(timestamp)
                            results.append((dt, float(value)))
            
            return sorted(results, key=lambda x: x[0])
            
        except Exception as e:
            self.logger.error(f"Failed to get ready time trend for VM {vm_name}: {e}")
            return []
    
    def get_vm_io_trend(self, vm_name: str, days: int = 2) -> List[Tuple[datetime, float]]:
        """
        Get I/O usage trend for a VM over specified days.
        
        Args:
            vm_name: Name of the VM
            days: Number of days to analyze
            
        Returns:
            List of (timestamp, io_usage) tuples
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        query = f'vmware_vm_disk_usage_average{{vm_name="{vm_name}"}}'
        
        try:
            data = self._query_prometheus(query, start_time, end_time)
            results = []
            
            if 'data' in data and 'result' in data['data']:
                for result in data['data']['result']:
                    if 'values' in result:
                        for timestamp, value in result['values']:
                            dt = datetime.fromtimestamp(timestamp)
                            results.append((dt, float(value)))
            
            return sorted(results, key=lambda x: x[0])
            
        except Exception as e:
            self.logger.error(f"Failed to get I/O trend for VM {vm_name}: {e}")
            return []
    
    def get_host_performance_metrics(self, host_name: str) -> Dict[str, float]:
        """
        Get current performance metrics for a host.
        
        Args:
            host_name: Name of the host
            
        Returns:
            Dict with current metrics (cpu, ram, io)
        """
        metrics = {}
        
        # CPU usage
        cpu_query = f'vmware_host_cpu_usage_average{{host_name="{host_name}"}}'
        try:
            data = self._query_prometheus(cpu_query)
            if 'data' in data and 'result' in data['data'] and data['data']['result']:
                metrics['cpu'] = float(data['data']['result'][0]['value'][1])
        except Exception as e:
            self.logger.warning(f"Failed to get CPU for host {host_name}: {e}")
            metrics['cpu'] = 0.0
        
        # RAM usage
        ram_query = f'vmware_host_memory_usage_average{{host_name="{host_name}"}}'
        try:
            data = self._query_prometheus(ram_query)
            if 'data' in data and 'result' in data['data'] and data['data']['result']:
                metrics['ram'] = float(data['data']['result'][0]['value'][1])
        except Exception as e:
            self.logger.warning(f"Failed to get RAM for host {host_name}: {e}")
            metrics['ram'] = 0.0
        
        # I/O usage
        io_query = f'vmware_host_disk_usage_average{{host_name="{host_name}"}}'
        try:
            data = self._query_prometheus(io_query)
            if 'data' in data and 'result' in data['data'] and data['data']['result']:
                metrics['io'] = float(data['data']['result'][0]['value'][1])
        except Exception as e:
            self.logger.warning(f"Failed to get I/O for host {host_name}: {e}")
            metrics['io'] = 0.0
        
        return metrics
    
    def get_host_metrics(self, host_name: str) -> Dict[str, float]:
        """
        Get current performance metrics for a host (alias for get_host_performance_metrics).
        
        Args:
            host_name: Name of the host
            
        Returns:
            Dict with current metrics (cpu, ram, io, ready_time, vm_count)
        """
        metrics = self.get_host_performance_metrics(host_name)
        
        # Add additional metrics that might be missing
        if 'ready_time' not in metrics:
            # Try to get ready time
            ready_query = f'vmware_host_cpu_ready_average{{host_name="{host_name}"}}'
            try:
                data = self._query_prometheus(ready_query)
                if 'data' in data and 'result' in data['data'] and data['data']['result']:
                    metrics['ready_time'] = float(data['data']['result'][0]['value'][1])
                else:
                    metrics['ready_time'] = 0.0
            except Exception:
                metrics['ready_time'] = 0.0
        
        if 'vm_count' not in metrics:
            # Try to get VM count
            vm_count_query = f'vmware_host_vm_count{{host_name="{host_name}"}}'
            try:
                data = self._query_prometheus(vm_count_query)
                if 'data' in data and 'result' in data['data'] and data['data']['result']:
                    metrics['vm_count'] = int(float(data['data']['result'][0]['value'][1]))
                else:
                    metrics['vm_count'] = 0
            except Exception:
                metrics['vm_count'] = 0
        
        # Ensure all required metrics are present
        required_metrics = ['cpu_usage', 'ram_usage', 'io_usage', 'ready_time', 'vm_count']
        for metric in required_metrics:
            if metric not in metrics:
                metrics[metric] = 0.0
        
        return metrics
    
    def get_all_vms_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current metrics for all VMs.
        
        Returns:
            Dict mapping VM names to their current metrics
        """
        query = 'vmware_vm_cpu_usage_average'
        
        try:
            data = self._query_prometheus(query)
            vms = {}
            
            if 'data' in data and 'result' in data['data']:
                for result in data['data']['result']:
                    vm_name = result['metric'].get('vm_name', 'unknown')
                    cpu_usage = float(result['value'][1])
                    
                    vms[vm_name] = {
                        'cpu': cpu_usage,
                        'ram': 0.0,  # Will be populated separately
                        'ready_time': 0.0,  # Will be populated separately
                        'io': 0.0  # Will be populated separately
                    }
            
            # Get RAM metrics
            ram_query = 'vmware_vm_memory_usage_average'
            try:
                ram_data = self._query_prometheus(ram_query)
                if 'data' in ram_data and 'result' in ram_data['data']:
                    for result in ram_data['data']['result']:
                        vm_name = result['metric'].get('vm_name', 'unknown')
                        if vm_name in vms:
                            vms[vm_name]['ram'] = float(result['value'][1])
            except Exception as e:
                self.logger.warning(f"Failed to get RAM metrics: {e}")
            
            # Get ready time metrics
            ready_query = 'vmware_vm_cpu_ready_average'
            try:
                ready_data = self._query_prometheus(ready_query)
                if 'data' in ready_data and 'result' in ready_data['data']:
                    for result in ready_data['data']['result']:
                        vm_name = result['metric'].get('vm_name', 'unknown')
                        if vm_name in vms:
                            vms[vm_name]['ready_time'] = float(result['value'][1])
            except Exception as e:
                self.logger.warning(f"Failed to get ready time metrics: {e}")
            
            return vms
            
        except Exception as e:
            self.logger.error(f"Failed to get all VMs metrics: {e}")
            return {}
    
    def get_storage_trends(self, vm_name: str, days: int = 3) -> List[Tuple[datetime, float]]:
        """
        Get storage usage trends for a VM over specified days.
        
        Args:
            vm_name: Name of the VM
            days: Number of days to analyze
            
        Returns:
            List of (timestamp, storage_usage) tuples
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        query = f'vmware_vm_disk_usage_average{{vm_name="{vm_name}"}}'
        
        try:
            data = self._query_prometheus(query, start_time, end_time)
            results = []
            
            if 'data' in data and 'result' in data['data']:
                for result in data['data']['result']:
                    if 'values' in result:
                        for timestamp, value in result['values']:
                            dt = datetime.fromtimestamp(timestamp)
                            results.append((dt, float(value)))
            
            return sorted(results, key=lambda x: x[0])
            
        except Exception as e:
            self.logger.error(f"Failed to get storage trends for VM {vm_name}: {e}")
            return []
    
    def test_connection(self) -> bool:
        """
        Test connection to Prometheus.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            query = 'up'
            data = self._query_prometheus(query)
            return data.get('status') == 'success'
        except Exception as e:
            self.logger.error(f"Prometheus connection test failed: {e}")
            return False 
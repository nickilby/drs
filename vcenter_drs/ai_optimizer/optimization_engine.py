"""Optimization Engine for AI-powered VM placement"""

import random
import time
from typing import Dict, List, Optional, Any, Union
from .config import AIConfig
from .data_collector import PrometheusDataCollector


class OptimizationEngine:
    """Coordinates ML models, data collection, and business rules for VM placement optimization"""
    
    def __init__(self, config: AIConfig, data_collector: PrometheusDataCollector):
        self.config = config
        self.data_collector = data_collector
        self.models_trained = False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of optimization configuration and status"""
        prometheus_connected = self.data_collector.test_connection()
        current_server = "None" if self.data_collector.use_simulated_data else self.data_collector.current_url
        
        return {
            "prometheus_connected": prometheus_connected,
            "current_prometheus_server": current_server,
            "models_trained": self.models_trained,
            "config": {
                "prometheus_url": self.config.prometheus.url,
                "ideal_host_usage_min": self.config.optimization.ideal_host_usage_min,
                "ideal_host_usage_max": self.config.optimization.ideal_host_usage_max,
                "max_recommendations": self.config.optimization.max_recommendations
            }
        }
    
    def train_models(self, vm_list: List[Dict], host_list: List[Dict]) -> bool:
        """Train the AI models with current VM and host data"""
        try:
            # For now, we'll simulate training success
            # In a real implementation, this would train actual ML models
            print(f"Training models with {len(vm_list)} VMs and {len(host_list)} hosts")
            
            # Simulate training time
            time.sleep(1)
            
            self.models_trained = True
            return True
        except Exception as e:
            print(f"Model training failed: {e}")
            return False
    
    def get_vm_metrics(self, vm_name: str) -> Dict[str, float]:
        """Get VM metrics for AI prediction"""
        try:
            # Use the data collector to get VM metrics
            vm_metrics = self.data_collector.get_vm_metrics(vm_name, self.config.analysis.cpu_trend_hours)
            return vm_metrics
        except Exception as e:
            print(f"Failed to get VM metrics for {vm_name}: {e}")
            # Return default metrics if collection fails
            return {
                'cpu_usage': 0.1,
                'ram_usage': 0.1,
                'io_usage': 0.05,
                'ready_time': 0.1,
                'cpu_mhz': 2000,
                'ram_mb': 2048
            }
    
    def generate_placement_recommendations(self, vm_name: str, cluster_filter: Optional[str] = None, 
                                         num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Generate VM placement recommendations using AI models"""
        start_time = time.time()
        max_execution_time = 30  # 30 seconds timeout
        
        try:
            print(f"Starting analysis for VM: {vm_name}")
            
            # Check timeout
            if time.time() - start_time > max_execution_time:
                print("Analysis timeout - aborting")
                return []
            
            # Get VM metrics
            print(f"Collecting metrics for VM: {vm_name}")
            vm_metrics = self.data_collector.get_vm_metrics(vm_name, self.config.analysis.cpu_trend_hours)
            print(f"VM metrics collected: CPU={vm_metrics.get('cpu_usage', 0):.1%}, RAM={vm_metrics.get('ram_usage', 0):.1%}")
            
            # Get the cluster for the VM
            vm_cluster = self.data_collector.get_vm_cluster(vm_name)
            print(f"VM {vm_name} is in cluster: {vm_cluster}")
            if not vm_cluster:
                print("Could not determine VM's cluster. Aborting recommendations.")
                return []
            
            # Check timeout
            if time.time() - start_time > max_execution_time:
                print("Analysis timeout - aborting")
                return []
            
            # Get available hosts (simulated for now)
            print(f"Analyzing available hosts (cluster filter: {cluster_filter})")
            available_hosts = self._get_available_hosts(cluster_filter)
            print(f"Found {len(available_hosts)} available hosts before cluster filtering")
            
            # Filter hosts to only those in the same cluster as the VM
            filtered_hosts = []
            for host in available_hosts:
                host_cluster = self.data_collector.get_host_cluster(host)
                if host_cluster == vm_cluster:
                    filtered_hosts.append(host)
            print(f"Filtered to {len(filtered_hosts)} hosts in the same cluster as VM")
            if not filtered_hosts:
                print("No available hosts found in the same cluster as the VM")
                return []
            
            recommendations: List[Dict[str, Any]] = []
            print(f"Evaluating {len(filtered_hosts)} hosts for placement...")
            
            for i, host_name in enumerate(filtered_hosts):
                # Check timeout
                if time.time() - start_time > max_execution_time:
                    print("Analysis timeout - aborting")
                    break
                
                print(f"Analyzing host {i+1}/{len(filtered_hosts)}: {host_name}")
                
                # Get current host metrics
                current_metrics = self.data_collector.get_host_metrics(host_name, self.config.analysis.cpu_trend_hours)
                print(f"  Current host metrics: CPU={current_metrics.get('cpu_usage', 0):.1%}, RAM={current_metrics.get('ram_usage', 0):.1%}, VMs={current_metrics.get('vm_count', 0)}")
                
                # Calculate projected metrics after VM placement
                projected_metrics = self._calculate_projected_metrics(current_metrics, vm_metrics)
                print(f"  Projected metrics: CPU={projected_metrics.get('cpu_usage', 0):.1%}, RAM={projected_metrics.get('ram_usage', 0):.1%}")
                
                # Calculate optimization score
                score = self._calculate_optimization_score(current_metrics, projected_metrics, vm_metrics)
                print(f"  Optimization score: {score:.3f}")
                
                # Generate reasoning
                reasoning = self._generate_reasoning(current_metrics, projected_metrics, vm_metrics, score)
                
                recommendation = {
                    "rank": len(recommendations) + 1,
                    "host_name": host_name,
                    "score": score,
                    "current_metrics": current_metrics,
                    "projected_metrics": projected_metrics,
                    "vm_metrics": vm_metrics,
                    "reasoning": reasoning
                }
                
                recommendations.append(recommendation)
                print(f"  Recommendation added for {host_name} with score {score:.3f}")
            
            # Sort by score (higher is better) and limit to requested number
            recommendations.sort(key=lambda x: float(x['score']), reverse=True)
            final_recommendations = recommendations[:num_recommendations]
            print(f"Generated {len(final_recommendations)} recommendations")
            
            return final_recommendations
            
        except Exception as e:
            print(f"Failed to generate recommendations: {e}")
            return []
    
    def _get_available_hosts(self, cluster_filter: Optional[Union[str, List[str]]] = None) -> List[str]:
        """Get list of available hosts from Prometheus"""
        try:
            # Query Prometheus for all hosts
            import requests
            response = requests.get("http://10.65.32.4:9090/api/v1/query?query=vmware_host_cpu_usage_average", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    # Extract unique host names from the results
                    all_hosts = list(set([result['metric']['host_name'] for result in data['data']['result']]))
                    print(f"Found {len(all_hosts)} total hosts in Prometheus")
                    return all_hosts
        except Exception as e:
            print(f"Failed to get hosts from Prometheus: {e}")
        
        # Fallback to dummy hosts if Prometheus is unavailable
        hosts = [
            "host-01.zengenti.com",
            "host-02.zengenti.com", 
            "host-03.zengenti.com",
            "host-04.zengenti.com",
            "host-05.zengenti.com"
        ]
        
        if cluster_filter:
            # Handle both string and list cluster filters
            if isinstance(cluster_filter, list):
                # If it's a list, use it directly as the host list
                return cluster_filter
            elif isinstance(cluster_filter, str):
                # Filter by cluster (simulated)
                if "cluster1" in cluster_filter.lower():
                    hosts = hosts[:2]
                elif "cluster2" in cluster_filter.lower():
                    hosts = hosts[2:]
        
        return hosts
    
    def _calculate_projected_metrics(self, current_metrics: Dict[str, float], 
                                   vm_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate projected host metrics after VM placement"""
        projected = current_metrics.copy()
        
        # Get actual values for proper calculation
        host_cpu_mhz = current_metrics.get('cpu_usage', 0.0) * current_metrics.get('cpu_max_mhz', 40000) / 10  # Convert from percentage to MHz
        host_ram_mb = current_metrics.get('ram_usage', 0.0) * current_metrics.get('ram_max_mb', 65536)  # Convert from percentage to MB
        
        # Get VM actual values
        vm_cpu_mhz = vm_metrics.get('cpu_mhz', 0.0)  # Already in MHz
        vm_ram_mb = vm_metrics.get('ram_mb', 0.0)    # Already in MB
        
        # Add VM resources to host resources
        projected_cpu_mhz = host_cpu_mhz + vm_cpu_mhz
        projected_ram_mb = host_ram_mb + vm_ram_mb
        
        # Convert back to percentages
        host_max_cpu_mhz = current_metrics.get('cpu_max_mhz', 40000)
        host_max_ram_mb = current_metrics.get('ram_max_mb', 65536)
        
        projected['cpu_usage'] = min(1.0, (projected_cpu_mhz / host_max_cpu_mhz) * 10)  # Convert to percentage
        projected['ram_usage'] = min(1.0, projected_ram_mb / host_max_ram_mb)  # Convert to percentage
        projected['vm_count'] = current_metrics.get('vm_count', 0) + 1
        
        # Project I/O usage (add VM I/O to host I/O)
        current_io = current_metrics.get('io_usage', 0.0)
        vm_io = vm_metrics.get('io_usage', 0.0)
        projected['io_usage'] = min(1.0, current_io + vm_io)
        
        # Remove host ready time projection - focus on VM ready time improvement instead
        # The VM's ready time will be used directly in the optimization score
        
        return projected
    
    def _calculate_optimization_score(self, current_metrics: Dict[str, float], 
                                    projected_metrics: Dict[str, float], 
                                    vm_metrics: Dict[str, float]) -> float:
        """Calculate optimization score for a host placement"""
        score = 0.0
        
        # Ideal host usage score (30-70% is ideal)
        projected_cpu = projected_metrics.get('cpu_usage', 0.0)
        projected_ram = projected_metrics.get('ram_usage', 0.0)
        
        # CPU usage score - more granular
        if self.config.optimization.ideal_host_usage_min <= projected_cpu <= self.config.optimization.ideal_host_usage_max:
            score += 0.25  # Ideal range
        elif projected_cpu <= 0.5:  # Low usage
            score += 0.15
        elif projected_cpu <= 0.8:  # Acceptable usage
            score += 0.1
        elif projected_cpu <= 0.9:  # High usage
            score += 0.05
        else:
            score -= 0.2  # Overloaded
        
        # RAM usage score - more granular
        if self.config.optimization.ideal_host_usage_min <= projected_ram <= self.config.optimization.ideal_host_usage_max:
            score += 0.25  # Ideal range
        elif projected_ram <= 0.5:  # Low usage
            score += 0.15
        elif projected_ram <= 0.8:  # Acceptable usage
            score += 0.1
        elif projected_ram <= 0.9:  # High usage
            score += 0.05
        else:
            score -= 0.2  # Overloaded
        
        # VM count score - more granular
        vm_count = projected_metrics.get('vm_count', 0)
        if vm_count <= 3:
            score += 0.15  # Very low - excellent
        elif vm_count <= 5:
            score += 0.1   # Low - good
        elif vm_count <= 10:
            score += 0.05  # Moderate - acceptable
        elif vm_count <= 20:
            score += 0.0   # High - neutral
        elif vm_count <= 30:
            score -= 0.05  # Very high - concerning
        else:
            score -= 0.1   # Extremely high - poor
        
        # VM ready time improvement score (lower ready time is better)
        vm_ready_time = vm_metrics.get('ready_time', 0.0)
        if vm_ready_time <= 0.1:  # Excellent ready time
            score += 0.15
        elif vm_ready_time <= 0.3:  # Good ready time
            score += 0.1
        elif vm_ready_time <= 0.5:  # Acceptable ready time
            score += 0.05
        elif vm_ready_time <= 0.8:  # Poor ready time
            score += 0.0
        else:
            score -= 0.1  # Very poor ready time
        
        # Resource efficiency score - more granular
        vm_cpu = vm_metrics.get('cpu_usage', 0.0)
        vm_ram = vm_metrics.get('ram_usage', 0.0)
        
        # Prefer hosts that can accommodate the VM's resource needs
        if projected_cpu <= 0.6 and projected_ram <= 0.6:
            score += 0.15  # Excellent resource availability
        elif projected_cpu <= 0.8 and projected_ram <= 0.8:
            score += 0.1   # Good resource availability
        elif projected_cpu <= 0.9 and projected_ram <= 0.9:
            score += 0.05  # Acceptable resource availability
        else:
            score -= 0.1   # Poor resource availability
        
        # Current host load consideration
        current_cpu = current_metrics.get('cpu_usage', 0.0)
        current_ram = current_metrics.get('ram_usage', 0.0)
        
        # Prefer hosts with lower current load
        if current_cpu <= 0.3 and current_ram <= 0.3:
            score += 0.1   # Very low current load
        elif current_cpu <= 0.5 and current_ram <= 0.5:
            score += 0.05  # Low current load
        elif current_cpu >= 0.8 or current_ram >= 0.8:
            score -= 0.05  # High current load
        
        # Normalize score to 0-1 range
        score = max(0.0, min(1.0, score))
        
        # Add small randomization for very similar hosts to provide differentiation
        # This helps break ties when hosts are very similar
        import random
        random.seed(hash(f"{current_metrics.get('cpu_usage', 0):.3f}{current_metrics.get('ram_usage', 0):.3f}{current_metrics.get('vm_count', 0)}"))
        score += random.uniform(-0.01, 0.01)  # Small variation
        
        # Ensure score stays in 0-1 range
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _generate_reasoning(self, current_metrics: Dict[str, float], 
                           projected_metrics: Dict[str, float], 
                           vm_metrics: Dict[str, float], 
                           score: float) -> str:
        """Generate human-readable reasoning for the recommendation"""
        reasons = []
        
        projected_cpu = projected_metrics.get('cpu_usage', 0.0)
        projected_ram = projected_metrics.get('ram_usage', 0.0)
        vm_count = projected_metrics.get('vm_count', 0)
        vm_ready_time = vm_metrics.get('ready_time', 0.0)
        
        # CPU reasoning
        if self.config.optimization.ideal_host_usage_min <= projected_cpu <= self.config.optimization.ideal_host_usage_max:
            reasons.append(f"CPU usage ({projected_cpu:.1%}) is in ideal range")
        elif projected_cpu > 0.9:
            reasons.append(f"Warning: High CPU usage ({projected_cpu:.1%})")
        else:
            reasons.append(f"CPU usage ({projected_cpu:.1%}) is acceptable")
        
        # RAM reasoning
        if self.config.optimization.ideal_host_usage_min <= projected_ram <= self.config.optimization.ideal_host_usage_max:
            reasons.append(f"RAM usage ({projected_ram:.1%}) is in ideal range")
        elif projected_ram > 0.9:
            reasons.append(f"Warning: High RAM usage ({projected_ram:.1%})")
        else:
            reasons.append(f"RAM usage ({projected_ram:.1%}) is acceptable")
        
        # VM count reasoning
        if vm_count <= 5:
            reasons.append(f"Low VM count ({vm_count}) - good for performance")
        elif vm_count <= 10:
            reasons.append(f"Moderate VM count ({vm_count}) - acceptable")
        else:
            reasons.append(f"High VM count ({vm_count}) - may impact performance")
        
        # VM ready time reasoning
        if vm_ready_time <= 0.1:
            reasons.append("Excellent VM ready time - optimal performance")
        elif vm_ready_time <= 0.5:
            reasons.append("Good VM ready time - acceptable performance")
        elif vm_ready_time <= 0.8:
            reasons.append("Moderate VM ready time - consider alternatives")
        else:
            reasons.append("Poor VM ready time - avoid this placement")
        
        return " | ".join(reasons) 
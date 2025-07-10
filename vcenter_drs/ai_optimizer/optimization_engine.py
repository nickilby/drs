"""Optimization Engine for AI-powered VM placement"""

import random
from typing import Dict, List, Optional, Any
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
        return {
            "prometheus_connected": self.data_collector.test_connection(),
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
            import time
            time.sleep(1)
            
            self.models_trained = True
            return True
        except Exception as e:
            print(f"Model training failed: {e}")
            return False
    
    def generate_placement_recommendations(self, vm_name: str, cluster_filter: Optional[str] = None, 
                                         num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Generate VM placement recommendations using AI models"""
        try:
            print(f"Starting analysis for VM: {vm_name}")
            
            # Get VM metrics
            print(f"Collecting metrics for VM: {vm_name}")
            vm_metrics = self.data_collector.get_vm_metrics(vm_name, self.config.analysis.cpu_trend_hours)
            print(f"VM metrics collected: CPU={vm_metrics.get('cpu_usage', 0):.1%}, RAM={vm_metrics.get('ram_usage', 0):.1%}")
            
            # Get available hosts (simulated for now)
            print(f"Analyzing available hosts (cluster filter: {cluster_filter})")
            available_hosts = self._get_available_hosts(cluster_filter)
            print(f"Found {len(available_hosts)} available hosts")
            
            if not available_hosts:
                print("No available hosts found")
                return []
            
            recommendations = []
            print(f"Evaluating {len(available_hosts)} hosts for placement...")
            
            for i, host_name in enumerate(available_hosts):
                print(f"Analyzing host {i+1}/{len(available_hosts)}: {host_name}")
                
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
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            final_recommendations = recommendations[:num_recommendations]
            print(f"Generated {len(final_recommendations)} recommendations")
            
            return final_recommendations
            
        except Exception as e:
            print(f"Failed to generate recommendations: {e}")
            return []
    
    def _get_available_hosts(self, cluster_filter: Optional[str] = None) -> List[str]:
        """Get list of available hosts (simulated)"""
        # In a real implementation, this would query the database
        # For now, return simulated host names
        hosts = [
            "host-01.zengenti.com",
            "host-02.zengenti.com", 
            "host-03.zengenti.com",
            "host-04.zengenti.com",
            "host-05.zengenti.com"
        ]
        
        if cluster_filter:
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
        
        # Add VM resource usage to host usage
        projected['cpu_usage'] = min(1.0, current_metrics.get('cpu_usage', 0.0) + vm_metrics.get('cpu_usage', 0.0))
        projected['ram_usage'] = min(1.0, current_metrics.get('ram_usage', 0.0) + vm_metrics.get('ram_usage', 0.0))
        projected['vm_count'] = current_metrics.get('vm_count', 0) + 1
        
        return projected
    
    def _calculate_optimization_score(self, current_metrics: Dict[str, float], 
                                    projected_metrics: Dict[str, float], 
                                    vm_metrics: Dict[str, float]) -> float:
        """Calculate optimization score for a host placement"""
        score = 0.0
        
        # Ideal host usage score (30-70% is ideal)
        projected_cpu = projected_metrics.get('cpu_usage', 0.0)
        projected_ram = projected_metrics.get('ram_usage', 0.0)
        
        # CPU usage score
        if self.config.optimization.ideal_host_usage_min <= projected_cpu <= self.config.optimization.ideal_host_usage_max:
            score += 0.3  # Ideal range
        elif projected_cpu < 0.9:  # Not overloaded
            score += 0.1
        else:
            score -= 0.2  # Overloaded
        
        # RAM usage score
        if self.config.optimization.ideal_host_usage_min <= projected_ram <= self.config.optimization.ideal_host_usage_max:
            score += 0.3  # Ideal range
        elif projected_ram < 0.9:  # Not overloaded
            score += 0.1
        else:
            score -= 0.2  # Overloaded
        
        # VM count score (prefer hosts with fewer VMs)
        vm_count = projected_metrics.get('vm_count', 0)
        if vm_count <= 5:
            score += 0.2
        elif vm_count <= 10:
            score += 0.1
        else:
            score -= 0.1
        
        # Resource efficiency score
        vm_cpu = vm_metrics.get('cpu_usage', 0.0)
        vm_ram = vm_metrics.get('ram_usage', 0.0)
        
        # Prefer hosts that can accommodate the VM's resource needs
        if projected_cpu <= 0.8 and projected_ram <= 0.8:
            score += 0.2
        
        # Normalize score to 0-1 range
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
            reasons.append(f"Moderate VM count ({vm_count})")
        else:
            reasons.append(f"High VM count ({vm_count}) - may impact performance")
        
        # Overall score reasoning
        if score >= 0.8:
            reasons.append("Excellent placement option")
        elif score >= 0.6:
            reasons.append("Good placement option")
        elif score >= 0.4:
            reasons.append("Acceptable placement option")
        else:
            reasons.append("Suboptimal placement - consider alternatives")
        
        return " | ".join(reasons) 
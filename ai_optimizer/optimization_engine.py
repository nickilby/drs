"""
Optimization Engine for AI Optimizer

This module provides the core optimization logic for VM placement,
combining ML predictions with business rules and compliance constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
from .config import AIConfig
from .data_collector import PrometheusDataCollector
from .ml_engine import MLEngine
from .exceptions import PlacementRecommendationError, ValidationError


class OptimizationEngine:
    """
    Main optimization engine for VM placement recommendations.
    
    This class coordinates between ML models, data collection, and business rules
    to provide optimal VM placement recommendations while respecting compliance constraints.
    """
    
    def __init__(self, config: AIConfig, data_collector: PrometheusDataCollector):
        """
        Initialize the optimization engine.
        
        Args:
            config: AI configuration
            data_collector: Prometheus data collector
        """
        self.config = config
        self.data_collector = data_collector
        self.ml_engine = MLEngine(config, data_collector)
        self.logger = logging.getLogger(__name__)
        
        # Load trained models if available
        self.ml_engine.load_trained_models()
    
    def get_vm_metrics(self, vm_name: str) -> Dict[str, float]:
        """
        Get comprehensive metrics for a VM.
        
        Args:
            vm_name: Name of the VM
            
        Returns:
            Dictionary of VM metrics
        """
        try:
            # Get trend data
            cpu_trend = self.data_collector.get_vm_cpu_trend(vm_name, self.config.analysis.cpu_trend_hours)
            ram_trend = self.data_collector.get_vm_ram_trend(vm_name, self.config.analysis.ram_trend_hours)
            ready_trend = self.data_collector.get_vm_ready_time_trend(vm_name, self.config.analysis.ready_time_window)
            io_trend = self.data_collector.get_vm_io_trend(vm_name, self.config.analysis.io_trend_days)
            storage_trend = self.data_collector.get_storage_trends(vm_name, self.config.analysis.storage_trend_days)
            
            # Calculate averages
            metrics = {
                'cpu_usage': np.mean([v for _, v in cpu_trend]) if cpu_trend else 0.0,
                'ram_usage': np.mean([v for _, v in ram_trend]) if ram_trend else 0.0,
                'ready_time': np.mean([v for _, v in ready_trend]) if ready_trend else 0.0,
                'io_usage': np.mean([v for _, v in io_trend]) if io_trend else 0.0,
                'storage_usage': np.mean([v for _, v in storage_trend]) if storage_trend else 0.0
            }
            
            # Add trend indicators
            if cpu_trend:
                cpu_values = [v for _, v in cpu_trend]
                metrics['cpu_trend'] = 'increasing' if cpu_values[-1] > cpu_values[0] else 'decreasing'
                metrics['cpu_volatility'] = np.std(cpu_values)
            
            if ram_trend:
                ram_values = [v for _, v in ram_trend]
                metrics['ram_trend'] = 'increasing' if ram_values[-1] > ram_values[0] else 'decreasing'
                metrics['ram_volatility'] = np.std(ram_values)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics for VM {vm_name}: {e}")
            return {
                'cpu_usage': 0.0,
                'ram_usage': 0.0,
                'ready_time': 0.0,
                'io_usage': 0.0,
                'storage_usage': 0.0,
                'cpu_trend': 'stable',
                'ram_trend': 'stable',
                'cpu_volatility': 0.0,
                'ram_volatility': 0.0
            }
    
    def get_host_candidates(self, cluster_name: str = None) -> List[Dict[str, Any]]:
        """
        Get candidate hosts for VM placement.
        
        Args:
            cluster_name: Optional cluster filter
            
        Returns:
            List of candidate host dictionaries
        """
        try:
            # Get all hosts from Prometheus
            all_hosts = self.data_collector.get_all_vms_metrics()
            
            candidates = []
            for host_name, host_data in all_hosts.items():
                if cluster_name and host_data.get('cluster') != cluster_name:
                    continue
                
                # Get host performance metrics
                host_metrics = self.data_collector.get_host_performance_metrics(host_name)
                
                candidate = {
                    'name': host_name,
                    'cluster': host_data.get('cluster', ''),
                    'cpu_usage': host_metrics.get('cpu_usage', 0.0),
                    'ram_usage': host_metrics.get('ram_usage', 0.0),
                    'io_usage': host_metrics.get('io_usage', 0.0),
                    'ready_time': host_metrics.get('ready_time', 0.0),
                    'vm_count': host_metrics.get('vm_count', 0),
                    'utilization_score': host_metrics.get('utilization_score', 0.0),
                    'available_cpu': max(0, 1.0 - host_metrics.get('cpu_usage', 0.0)),
                    'available_ram': max(0, 1.0 - host_metrics.get('ram_usage', 0.0))
                }
                
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Failed to get host candidates: {e}")
            return []
    
    def filter_hosts_by_constraints(self, hosts: List[Dict[str, Any]], vm_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Filter hosts based on resource constraints.
        
        Args:
            hosts: List of candidate hosts
            vm_metrics: VM resource requirements
            
        Returns:
            List of hosts that meet resource constraints
        """
        filtered_hosts = []
        
        for host in hosts:
            # Check if host has enough resources
            cpu_available = host.get('available_cpu', 0.0)
            ram_available = host.get('available_ram', 0.0)
            
            vm_cpu = vm_metrics.get('cpu_usage', 0.0)
            vm_ram = vm_metrics.get('ram_usage', 0.0)
            
            # Add safety margin (20%)
            required_cpu = vm_cpu * 1.2
            required_ram = vm_ram * 1.2
            
            if cpu_available >= required_cpu and ram_available >= required_ram:
                # Calculate projected utilization
                projected_cpu = host.get('cpu_usage', 0.0) + vm_cpu
                projected_ram = host.get('ram_usage', 0.0) + vm_ram
                
                # Check if projected utilization is within ideal range
                ideal_min = self.config.optimization.ideal_host_usage_min
                ideal_max = self.config.optimization.ideal_host_usage_max
                
                if (ideal_min <= projected_cpu <= ideal_max and 
                    ideal_min <= projected_ram <= ideal_max):
                    filtered_hosts.append(host)
        
        return filtered_hosts
    
    def rank_hosts_by_ml_score(self, hosts: List[Dict[str, Any]], vm_metrics: Dict[str, float]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank hosts using ML predictions.
        
        Args:
            hosts: List of candidate hosts
            vm_metrics: VM performance metrics
            
        Returns:
            List of (host, score) tuples sorted by score
        """
        try:
            # Get ML predictions
            scores = self.ml_engine.predict_placement_scores(vm_metrics, hosts)
            
            # Combine hosts with scores
            host_scores = list(zip(hosts, scores))
            
            # Sort by score (descending)
            host_scores.sort(key=lambda x: x[1], reverse=True)
            
            return host_scores
            
        except Exception as e:
            self.logger.error(f"Failed to rank hosts by ML score: {e}")
            # Fallback to simple ranking
            return [(host, 0.5) for host in hosts]
    
    def apply_business_rules(self, host_scores: List[Tuple[Dict[str, Any], float]], vm_name: str) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply business rules to adjust host rankings.
        
        Args:
            host_scores: List of (host, score) tuples
            vm_name: Name of the VM being placed
            
        Returns:
            List of (host, adjusted_score) tuples
        """
        adjusted_scores = []
        
        for host, score in host_scores:
            adjusted_score = score
            
            # Rule 1: Prefer hosts with lower VM count (better resource distribution)
            vm_count = host.get('vm_count', 0)
            if vm_count > 10:  # Penalty for hosts with many VMs
                adjusted_score *= 0.9
            
            # Rule 2: Prefer hosts with lower ready time
            ready_time = host.get('ready_time', 0.0)
            if ready_time > 0.1:  # Penalty for high ready time
                adjusted_score *= (1.0 - ready_time)
            
            # Rule 3: Prefer hosts with balanced CPU/RAM usage
            cpu_usage = host.get('cpu_usage', 0.0)
            ram_usage = host.get('ram_usage', 0.0)
            usage_diff = abs(cpu_usage - ram_usage)
            if usage_diff > 0.3:  # Penalty for imbalanced usage
                adjusted_score *= 0.8
            
            # Rule 4: Prefer hosts with stable performance (low volatility)
            if 'cpu_volatility' in host:
                volatility = host.get('cpu_volatility', 0.0)
                if volatility > 0.1:  # Penalty for high volatility
                    adjusted_score *= 0.9
            
            adjusted_scores.append((host, adjusted_score))
        
        # Re-sort by adjusted scores
        adjusted_scores.sort(key=lambda x: x[1], reverse=True)
        
        return adjusted_scores
    
    def generate_placement_recommendations(self, vm_name: str, cluster_name: str = None, 
                                         max_recommendations: int = None) -> List[Dict[str, Any]]:
        """
        Generate VM placement recommendations.
        
        Args:
            vm_name: Name of the VM to place
            cluster_name: Optional cluster constraint
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of placement recommendations
        """
        try:
            # Get VM metrics
            vm_metrics = self.get_vm_metrics(vm_name)
            
            # Get host candidates
            host_candidates = self.get_host_candidates(cluster_name)
            
            if not host_candidates:
                raise PlacementRecommendationError(vm_name, "No host candidates available")
            
            # Filter hosts by resource constraints
            filtered_hosts = self.filter_hosts_by_constraints(host_candidates, vm_metrics)
            
            if not filtered_hosts:
                raise PlacementRecommendationError(vm_name, "No hosts meet resource constraints")
            
            # Rank hosts using ML
            host_scores = self.rank_hosts_by_ml_score(filtered_hosts, vm_metrics)
            
            # Apply business rules
            adjusted_scores = self.apply_business_rules(host_scores, vm_name)
            
            # Generate recommendations
            max_recs = max_recommendations or self.config.optimization.max_recommendations
            recommendations: List[Dict[str, Any]] = []
            
            for i, (host, score) in enumerate(adjusted_scores[:max_recs]):
                # Calculate projected metrics
                projected_cpu = min(1.0, host.get('cpu_usage', 0.0) + vm_metrics.get('cpu_usage', 0.0))
                projected_ram = min(1.0, host.get('ram_usage', 0.0) + vm_metrics.get('ram_usage', 0.0))
                
                recommendation = {
                    'rank': i + 1,
                    'host_name': host.get('name', ''),
                    'cluster': host.get('cluster', ''),
                    'score': round(score, 3),
                    'current_metrics': {
                        'cpu_usage': round(host.get('cpu_usage', 0.0), 3),
                        'ram_usage': round(host.get('ram_usage', 0.0), 3),
                        'io_usage': round(host.get('io_usage', 0.0), 3),
                        'ready_time': round(host.get('ready_time', 0.0), 3),
                        'vm_count': host.get('vm_count', 0)
                    },
                    'projected_metrics': {
                        'cpu_usage': round(projected_cpu, 3),
                        'ram_usage': round(projected_ram, 3)
                    },
                    'vm_metrics': {
                        'cpu_usage': round(vm_metrics.get('cpu_usage', 0.0), 3),
                        'ram_usage': round(vm_metrics.get('ram_usage', 0.0), 3),
                        'ready_time': round(vm_metrics.get('ready_time', 0.0), 3),
                        'io_usage': round(vm_metrics.get('io_usage', 0.0), 3)
                    },
                    'reasoning': self._generate_reasoning(host, vm_metrics, score)
                }
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations for VM {vm_name}: {e}")
            raise PlacementRecommendationError(vm_name, str(e))
    
    def _generate_reasoning(self, host: Dict[str, Any], vm_metrics: Dict[str, float], score: float) -> str:
        """
        Generate human-readable reasoning for a placement recommendation.
        
        Args:
            host: Host information
            vm_metrics: VM metrics
            score: Placement score
            
        Returns:
            String explaining the recommendation
        """
        reasons = []
        
        # CPU reasoning
        cpu_usage = host.get('cpu_usage', 0.0)
        vm_cpu = vm_metrics.get('cpu_usage', 0.0)
        projected_cpu = min(1.0, cpu_usage + vm_cpu)
        
        if 0.3 <= projected_cpu <= 0.7:
            reasons.append("Optimal CPU utilization projected")
        elif projected_cpu < 0.3:
            reasons.append("Low CPU utilization - good for performance")
        else:
            reasons.append("High CPU utilization - monitor closely")
        
        # RAM reasoning
        ram_usage = host.get('ram_usage', 0.0)
        vm_ram = vm_metrics.get('ram_usage', 0.0)
        projected_ram = min(1.0, ram_usage + vm_ram)
        
        if 0.3 <= projected_ram <= 0.7:
            reasons.append("Optimal RAM utilization projected")
        elif projected_ram < 0.3:
            reasons.append("Low RAM utilization - good for performance")
        else:
            reasons.append("High RAM utilization - monitor closely")
        
        # VM count reasoning
        vm_count = host.get('vm_count', 0)
        if vm_count < 5:
            reasons.append("Low VM density - good for resource isolation")
        elif vm_count > 15:
            reasons.append("High VM density - consider resource contention")
        
        # Ready time reasoning
        ready_time = host.get('ready_time', 0.0)
        if ready_time < 0.05:
            reasons.append("Low CPU ready time - good performance")
        elif ready_time > 0.1:
            reasons.append("High CPU ready time - performance concern")
        
        # Score-based reasoning
        if score > 0.8:
            reasons.append("Excellent placement score")
        elif score > 0.6:
            reasons.append("Good placement score")
        else:
            reasons.append("Acceptable placement score")
        
        return "; ".join(reasons)
    
    def train_models(self, vms: List[Dict[str, Any]], hosts: List[Dict[str, Any]]) -> bool:
        """
        Train the ML models using provided data.
        
        Args:
            vms: List of VM dictionaries
            hosts: List of host dictionaries
            
        Returns:
            bool: True if training successful
        """
        try:
            return self.ml_engine.train_models(vms, hosts)
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization system status.
        
        Returns:
            Dictionary with optimization system information
        """
        return {
            'models_available': {
                'baseline_model': self.ml_engine.baseline_model.is_trained,
                'neural_network': self.ml_engine.neural_network.is_trained
            },
            'configuration': {
                'ideal_host_usage_min': self.config.optimization.ideal_host_usage_min,
                'ideal_host_usage_max': self.config.optimization.ideal_host_usage_max,
                'max_recommendations': self.config.optimization.max_recommendations
            },
            'prometheus_connection': self.data_collector.test_connection(),
            'last_training': datetime.now().isoformat() if self.ml_engine.baseline_model.is_trained else None
        } 
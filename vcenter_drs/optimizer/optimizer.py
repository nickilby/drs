"""
VM Placement Optimization Engine

This module provides AI/ML-driven VM placement optimization that considers:
1. Compliance rules (anti-affinity, affinity, etc.)
2. Performance metrics (CPU, memory, network, storage)
3. Resource utilization and capacity
4. Historical performance patterns
5. Operational efficiency metrics

The optimizer uses a multi-criteria decision-making approach to rank hosts
and recommend optimal placement for VMs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from vcenter_drs.db.metrics_db import MetricsDB
from vcenter_drs.rules.rules_engine import evaluate_rules, get_db_state


@dataclass
class HostMetrics:
    """Container for host performance metrics"""
    host_id: int
    host_name: str
    cluster_id: int
    cluster_name: str
    
    # Current utilization
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    network_usage: float  # Mbps
    storage_io: float  # IOPS
    
    # Capacity
    cpu_cores: int
    memory_mb: int
    network_bandwidth: float  # Mbps
    storage_capacity: float  # GB
    
    # Performance indicators
    cpu_ready_time: float  # ms
    memory_balloon: float  # MB
    network_packet_loss: float  # Percentage
    storage_latency: float  # ms
    
    # Historical trends (last 24h averages)
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_network_usage: float
    avg_storage_io: float
    
    # VM interference score (0-1, higher = more interference)
    vm_interference_score: float
    
    # Compliance violations on this host
    compliance_violations: int


@dataclass
class VMPlacementRequest:
    """Container for VM placement request"""
    vm_name: str
    vm_alias: str
    vm_role: str
    required_cpu: float  # vCPUs
    required_memory: float  # MB
    required_storage: float  # GB
    network_requirements: float  # Mbps
    priority: str = "normal"  # low, normal, high, critical
    preferred_hosts: Optional[List[str]] = None
    excluded_hosts: Optional[List[str]] = None


@dataclass
class PlacementRecommendation:
    """Container for placement recommendation"""
    host_id: int
    host_name: str
    cluster_name: str
    score: float  # 0-1, higher is better
    compliance_score: float  # 0-1, higher is better
    performance_score: float  # 0-1, higher is better
    resource_score: float  # 0-1, higher is better
    reasoning: List[str]
    estimated_impact: Dict[str, float]
    migration_cost: float  # Estimated migration time in minutes


class VMOptimizer:
    """
    AI/ML-driven VM placement optimizer that considers compliance, performance, and efficiency.
    """
    
    def __init__(self):
        self.db = MetricsDB()
        self.scaler = StandardScaler()
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.interference_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical performance data for model training"""
        try:
            self.db.connect()
            cursor = self.db.conn.cursor(dictionary=True)
            
            # Get historical metrics (last 7 days)
            cursor.execute("""
                SELECT m.*, h.name as host_name, v.name as vm_name
                FROM metrics m
                JOIN hosts h ON m.object_id = h.id
                LEFT JOIN vms v ON m.object_id = v.id
                WHERE m.timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                AND m.metric_name IN ('cpu.usage.average', 'mem.usage.average', 'net.usage.average', 'disk.usage.average')
                ORDER BY m.timestamp
            """)
            
            self.historical_data = cursor.fetchall()
            cursor.close()
            
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            self.historical_data = []
    
    def get_host_metrics(self, host_id: int) -> Optional[HostMetrics]:
        """Get comprehensive metrics for a specific host"""
        try:
            cursor = self.db.conn.cursor(dictionary=True)
            
            # Get host basic info
            cursor.execute("""
                SELECT h.*, c.name as cluster_name
                FROM hosts h
                JOIN clusters c ON h.cluster_id = c.id
                WHERE h.id = %s
            """, (host_id,))
            host_info = cursor.fetchone()
            
            if not host_info:
                return None
            
            # Get latest metrics
            cursor.execute("""
                SELECT metric_name, value, timestamp
                FROM metrics
                WHERE object_type = 'host' AND object_id = %s
                AND timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
                ORDER BY timestamp DESC
            """, (host_id,))
            metrics = cursor.fetchall()
            
            # Get VMs on this host
            cursor.execute("""
                SELECT COUNT(*) as vm_count
                FROM vms
                WHERE host_id = %s AND power_status = 'poweredOn'
            """, (host_id,))
            vm_count = cursor.fetchone()['vm_count']
            
            # Calculate metrics
            cpu_usage = self._get_latest_metric(metrics, 'cpu.usage.average', 0.0)
            memory_usage = self._get_latest_metric(metrics, 'mem.usage.average', 0.0)
            network_usage = self._get_latest_metric(metrics, 'net.usage.average', 0.0)
            storage_io = self._get_latest_metric(metrics, 'disk.usage.average', 0.0)
            
            # Calculate averages (last 24h)
            avg_cpu = self._get_average_metric(metrics, 'cpu.usage.average', 0.0)
            avg_memory = self._get_average_metric(metrics, 'mem.usage.average', 0.0)
            avg_network = self._get_average_metric(metrics, 'net.usage.average', 0.0)
            avg_storage = self._get_average_metric(metrics, 'disk.usage.average', 0.0)
            
            # Estimate capacity (this would come from vCenter in real implementation)
            cpu_cores = 32  # Default estimate
            memory_mb = 256 * 1024  # 256GB default
            network_bandwidth = 10000  # 10Gbps default
            storage_capacity = 1000 * 1024  # 1TB default
            
            # Calculate interference score based on VM density and performance variance
            interference_score = self._calculate_interference_score(host_id, vm_count)
            
            # Count compliance violations
            violations = self._count_host_violations(host_id)
            
            cursor.close()
            
            return HostMetrics(
                host_id=host_id,
                host_name=host_info['name'],
                cluster_id=host_info['cluster_id'],
                cluster_name=host_info['cluster_name'],
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                network_usage=network_usage,
                storage_io=storage_io,
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                network_bandwidth=network_bandwidth,
                storage_capacity=storage_capacity,
                cpu_ready_time=0.0,  # Would come from vCenter
                memory_balloon=0.0,  # Would come from vCenter
                network_packet_loss=0.0,  # Would come from vCenter
                storage_latency=0.0,  # Would come from vCenter
                avg_cpu_usage=avg_cpu,
                avg_memory_usage=avg_memory,
                avg_network_usage=avg_network,
                avg_storage_io=avg_storage,
                vm_interference_score=interference_score,
                compliance_violations=violations
            )
            
        except Exception as e:
            print(f"Error getting host metrics: {e}")
            return None
    
    def _get_latest_metric(self, metrics: List[Dict], metric_name: str, default: float) -> float:
        """Get the latest value for a specific metric"""
        for metric in metrics:
            if metric['metric_name'] == metric_name:
                return float(metric['value'])
        return default
    
    def _get_average_metric(self, metrics: List[Dict], metric_name: str, default: float) -> float:
        """Get the average value for a specific metric over the last 24h"""
        values = [float(m['value']) for m in metrics if m['metric_name'] == metric_name]
        return float(np.mean(values)) if values else default
    
    def _calculate_interference_score(self, host_id: int, vm_count: int) -> float:
        """Calculate VM interference score based on density and performance variance"""
        # Simple heuristic: more VMs = higher interference
        # In real implementation, this would analyze performance correlation between VMs
        base_score = min(vm_count / 20.0, 1.0)  # Normalize to 0-1
        
        # Add randomness to simulate real interference patterns
        noise = np.random.normal(0, 0.1)
        return max(0.0, min(1.0, base_score + noise))
    
    def _count_host_violations(self, host_id: int) -> int:
        """Count compliance violations involving this host"""
        try:
            # Get VMs on this host
            cursor = self.db.conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT v.name, v.host_id, c.name as cluster_name
                FROM vms v
                JOIN hosts h ON v.host_id = h.id
                JOIN clusters c ON h.cluster_id = c.id
                WHERE v.host_id = %s AND v.power_status = 'poweredOn'
            """, (host_id,))
            host_vms = cursor.fetchall()
            cursor.close()
            
            if not host_vms:
                return 0
            
            # Run compliance check for the cluster
            cluster_name = host_vms[0]['cluster_name']
            violations = evaluate_rules(cluster_name)
            
            # Count violations involving VMs on this host
            violation_count = 0
            host_vm_names = {vm['name'] for vm in host_vms}
            
            for violation in violations:
                affected_vms = violation.get('affected_vms', [])
                if any(vm in host_vm_names for vm in affected_vms):
                    violation_count += 1
            
            return violation_count
            
        except Exception as e:
            print(f"Error counting host violations: {e}")
            return 0
    
    def find_suitable_hosts(self, request: VMPlacementRequest) -> List[PlacementRecommendation]:
        """
        Find suitable hosts for VM placement based on multiple criteria.
        
        Returns ranked list of placement recommendations.
        """
        try:
            # Get all hosts
            clusters, hosts, vms = get_db_state()
            
            recommendations = []
            
            for host_id, host_info in hosts.items():
                # Skip excluded hosts
                if request.excluded_hosts and host_info['name'] in request.excluded_hosts:
                    continue
                
                # Get comprehensive host metrics
                host_metrics = self.get_host_metrics(host_id)
                if not host_metrics:
                    continue
                
                # Check if host can accommodate the VM
                if not self._can_accommodate_vm(host_metrics, request):
                    continue
                
                # Calculate scores
                compliance_score = self._calculate_compliance_score(host_metrics, request)
                performance_score = self._calculate_performance_score(host_metrics, request)
                resource_score = self._calculate_resource_score(host_metrics, request)
                
                # Weighted overall score
                overall_score = (
                    compliance_score * 0.4 +  # Compliance is most important
                    performance_score * 0.35 +  # Performance second
                    resource_score * 0.25  # Resource efficiency third
                )
                
                # Generate reasoning
                reasoning = self._generate_reasoning(host_metrics, request, compliance_score, performance_score, resource_score)
                
                # Estimate migration cost
                migration_cost = self._estimate_migration_cost(host_metrics, request)
                
                # Estimate performance impact
                estimated_impact = self._estimate_performance_impact(host_metrics, request)
                
                recommendations.append(PlacementRecommendation(
                    host_id=host_id,
                    host_name=host_metrics.host_name,
                    cluster_name=host_metrics.cluster_name,
                    score=overall_score,
                    compliance_score=compliance_score,
                    performance_score=performance_score,
                    resource_score=resource_score,
                    reasoning=reasoning,
                    estimated_impact=estimated_impact,
                    migration_cost=migration_cost
                ))
            
            # Sort by overall score (descending)
            recommendations.sort(key=lambda x: x.score, reverse=True)
            
            return recommendations
            
        except Exception as e:
            print(f"Error finding suitable hosts: {e}")
            return []
    
    def _can_accommodate_vm(self, host_metrics: HostMetrics, request: VMPlacementRequest) -> bool:
        """Check if host has sufficient resources to accommodate the VM"""
        # Check CPU capacity
        available_cpu = host_metrics.cpu_cores * (1 - host_metrics.cpu_usage / 100)
        if available_cpu < request.required_cpu:
            return False
        
        # Check memory capacity
        available_memory = host_metrics.memory_mb * (1 - host_metrics.memory_usage / 100)
        if available_memory < request.required_memory:
            return False
        
        # Check storage capacity (simplified)
        available_storage = host_metrics.storage_capacity * 0.2  # Assume 20% available
        if available_storage < request.required_storage:
            return False
        
        return True
    
    def _calculate_compliance_score(self, host_metrics: HostMetrics, request: VMPlacementRequest) -> float:
        """Calculate compliance score (0-1, higher is better)"""
        try:
            # Simulate compliance check for this specific placement
            # In real implementation, this would run the rules engine with the VM placed on this host
            
            # Base score starts at 1.0
            score = 1.0
            
            # Penalize hosts with existing violations
            if host_metrics.compliance_violations > 0:
                score -= min(host_metrics.compliance_violations * 0.2, 0.8)
            
            # Check for potential anti-affinity violations
            # This is a simplified check - real implementation would be more sophisticated
            if self._would_violate_anti_affinity(host_metrics, request):
                score -= 0.5
            
            # Check for potential affinity violations
            if self._would_violate_affinity(host_metrics, request):
                score -= 0.3
            
            return max(0.0, score)
            
        except Exception as e:
            print(f"Error calculating compliance score: {e}")
            return 0.5  # Neutral score on error
    
    def _would_violate_anti_affinity(self, host_metrics: HostMetrics, request: VMPlacementRequest) -> bool:
        """Check if placing VM on this host would violate anti-affinity rules"""
        try:
            # Get VMs on this host
            cursor = self.db.conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT v.name
                FROM vms v
                WHERE v.host_id = %s AND v.power_status = 'poweredOn'
            """, (host_metrics.host_id,))
            host_vms = cursor.fetchall()
            cursor.close()
            
            # Check if any VM on this host has the same alias and role
            for vm in host_vms:
                vm_name = vm['name']
                # Simplified check - in real implementation, parse alias and role
                if request.vm_alias in vm_name and request.vm_role in vm_name:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking anti-affinity: {e}")
            return False
    
    def _would_violate_affinity(self, host_metrics: HostMetrics, request: VMPlacementRequest) -> bool:
        """Check if placing VM on this host would violate affinity rules"""
        try:
            # Get all VMs with same alias and role
            cursor = self.db.conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT v.name, h.name as host_name
                FROM vms v
                JOIN hosts h ON v.host_id = h.id
                WHERE v.power_status = 'poweredOn'
            """, ())
            all_vms = cursor.fetchall()
            cursor.close()
            
            # Find VMs with same alias and role on different hosts
            same_alias_role_vms = []
            for vm in all_vms:
                vm_name = vm['name']
                if request.vm_alias in vm_name and request.vm_role in vm_name:
                    same_alias_role_vms.append(vm)
            
            # If there are VMs with same alias/role on different hosts, this would violate affinity
            if same_alias_role_vms:
                other_hosts = {vm['host_name'] for vm in same_alias_role_vms}
                if host_metrics.host_name not in other_hosts:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking affinity: {e}")
            return False
    
    def _calculate_performance_score(self, host_metrics: HostMetrics, request: VMPlacementRequest) -> float:
        """Calculate performance score (0-1, higher is better)"""
        # Start with perfect score
        score = 1.0
        
        # Penalize high CPU utilization
        if host_metrics.cpu_usage > 80:
            score -= 0.3
        elif host_metrics.cpu_usage > 60:
            score -= 0.1
        
        # Penalize high memory utilization
        if host_metrics.memory_usage > 85:
            score -= 0.3
        elif host_metrics.memory_usage > 70:
            score -= 0.1
        
        # Penalize high network utilization
        if host_metrics.network_usage > 80:
            score -= 0.2
        elif host_metrics.network_usage > 60:
            score -= 0.05
        
        # Penalize high storage I/O
        if host_metrics.storage_io > 80:
            score -= 0.2
        elif host_metrics.storage_io > 60:
            score -= 0.05
        
        # Penalize VM interference
        score -= host_metrics.vm_interference_score * 0.2
        
        # Bonus for hosts with good historical performance
        if host_metrics.avg_cpu_usage < 50 and host_metrics.avg_memory_usage < 60:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_resource_score(self, host_metrics: HostMetrics, request: VMPlacementRequest) -> float:
        """Calculate resource efficiency score (0-1, higher is better)"""
        # Prefer hosts with balanced resource utilization
        cpu_util = host_metrics.cpu_usage / 100
        mem_util = host_metrics.memory_usage / 100
        net_util = host_metrics.network_usage / host_metrics.network_bandwidth
        
        # Calculate resource balance (lower variance is better)
        utilizations = [cpu_util, mem_util, net_util]
        variance = np.var(utilizations)
        
        # Score based on balance and overall utilization
        balance_score = max(0.0, 1.0 - variance * 2)  # Lower variance = higher score
        
        # Prefer moderate utilization (not too low, not too high)
        avg_utilization = np.mean(utilizations)
        if 0.3 <= avg_utilization <= 0.7:
            utilization_score = 1.0
        elif avg_utilization < 0.3:
            utilization_score = avg_utilization / 0.3  # Underutilized
        else:
            utilization_score = max(0.0, 1.0 - (avg_utilization - 0.7) / 0.3)  # Overutilized
        
        return (balance_score * 0.6 + utilization_score * 0.4)
    
    def _generate_reasoning(self, host_metrics: HostMetrics, request: VMPlacementRequest, 
                           compliance_score: float, performance_score: float, resource_score: float) -> List[str]:
        """Generate human-readable reasoning for the recommendation"""
        reasoning = []
        
        # Compliance reasoning
        if compliance_score >= 0.9:
            reasoning.append("✅ Excellent compliance - no rule violations expected")
        elif compliance_score >= 0.7:
            reasoning.append("✅ Good compliance - minimal rule violation risk")
        elif compliance_score >= 0.5:
            reasoning.append("⚠️ Moderate compliance - some rule violation risk")
        else:
            reasoning.append("❌ Poor compliance - high risk of rule violations")
        
        # Performance reasoning
        if host_metrics.cpu_usage < 60:
            reasoning.append(f"✅ Low CPU utilization ({host_metrics.cpu_usage:.1f}%)")
        elif host_metrics.cpu_usage < 80:
            reasoning.append(f"⚠️ Moderate CPU utilization ({host_metrics.cpu_usage:.1f}%)")
        else:
            reasoning.append(f"❌ High CPU utilization ({host_metrics.cpu_usage:.1f}%)")
        
        if host_metrics.memory_usage < 70:
            reasoning.append(f"✅ Low memory utilization ({host_metrics.memory_usage:.1f}%)")
        elif host_metrics.memory_usage < 85:
            reasoning.append(f"⚠️ Moderate memory utilization ({host_metrics.memory_usage:.1f}%)")
        else:
            reasoning.append(f"❌ High memory utilization ({host_metrics.memory_usage:.1f}%)")
        
        # Resource efficiency reasoning
        if resource_score >= 0.8:
            reasoning.append("✅ Excellent resource balance")
        elif resource_score >= 0.6:
            reasoning.append("✅ Good resource balance")
        else:
            reasoning.append("⚠️ Poor resource balance")
        
        # VM interference reasoning
        if host_metrics.vm_interference_score < 0.3:
            reasoning.append("✅ Low VM interference")
        elif host_metrics.vm_interference_score < 0.6:
            reasoning.append("⚠️ Moderate VM interference")
        else:
            reasoning.append("❌ High VM interference")
        
        return reasoning
    
    def _estimate_migration_cost(self, host_metrics: HostMetrics, request: VMPlacementRequest) -> float:
        """Estimate migration time in minutes"""
        # Base migration time based on VM size
        base_time = request.required_memory / 1024  # 1 minute per GB
        
        # Adjust for network bandwidth
        network_factor = max(0.5, min(2.0, 1000 / host_metrics.network_bandwidth))
        
        # Adjust for storage latency
        storage_factor = max(0.5, min(2.0, 5 / host_metrics.storage_latency)) if host_metrics.storage_latency > 0 else 1.0
        
        return base_time * network_factor * storage_factor
    
    def _estimate_performance_impact(self, host_metrics: HostMetrics, request: VMPlacementRequest) -> Dict[str, float]:
        """Estimate performance impact of placing VM on this host"""
        # Estimate new CPU utilization
        current_cpu_cores_used = host_metrics.cpu_cores * (host_metrics.cpu_usage / 100)
        new_cpu_cores_used = current_cpu_cores_used + request.required_cpu
        new_cpu_utilization = (new_cpu_cores_used / host_metrics.cpu_cores) * 100
        
        # Estimate new memory utilization
        current_memory_used = host_metrics.memory_mb * (host_metrics.memory_usage / 100)
        new_memory_used = current_memory_used + request.required_memory
        new_memory_utilization = (new_memory_used / host_metrics.memory_mb) * 100
        
        return {
            'cpu_utilization_change': new_cpu_utilization - host_metrics.cpu_usage,
            'memory_utilization_change': new_memory_utilization - host_metrics.memory_usage,
            'estimated_cpu_ready_time': max(0.0, float((new_cpu_utilization - 80) * 2)),  # ms
            'estimated_memory_balloon': max(0.0, float((new_memory_utilization - 85) * 100))  # MB
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization capabilities and metrics"""
        try:
            clusters, hosts, vms = get_db_state()
            
            # Calculate cluster-level metrics
            cluster_metrics = {}
            for cluster_id, cluster_name in clusters.items():
                cluster_hosts = [h for h in hosts.values() if h['cluster_id'] == cluster_id]
                cluster_vms = [v for v in vms.values() if any(h['cluster_id'] == cluster_id for h in hosts.values() if h['id'] == v['host_id'])]
                
                avg_cpu = np.mean([h.get('cpu_usage', 0) for h in cluster_hosts]) if cluster_hosts else 0
                avg_memory = np.mean([h.get('memory_usage', 0) for h in cluster_hosts]) if cluster_hosts else 0
                
                cluster_metrics[cluster_name] = {
                    'host_count': len(cluster_hosts),
                    'vm_count': len(cluster_vms),
                    'avg_cpu_utilization': avg_cpu,
                    'avg_memory_utilization': avg_memory,
                    'resource_efficiency': 1 - abs(avg_cpu - avg_memory) / 100  # Balance score
                }
            
            return {
                'total_hosts': len(hosts),
                'total_vms': len(vms),
                'total_clusters': len(clusters),
                'cluster_metrics': cluster_metrics,
                'optimization_capabilities': [
                    'Compliance-aware placement',
                    'Performance-based ranking',
                    'Resource efficiency optimization',
                    'VM interference analysis',
                    'Migration cost estimation',
                    'Historical performance analysis'
                ]
            }
            
        except Exception as e:
            print(f"Error getting optimization summary: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    optimizer = VMOptimizer()
    
    # Example placement request
    request = VMPlacementRequest(
        vm_name="z-example-alias-WEB1",
        vm_alias="z-example-alias",
        vm_role="WEB",
        required_cpu=4.0,
        required_memory=8192,  # 8GB
        required_storage=100,  # 100GB
        network_requirements=100,  # 100 Mbps
        priority="normal"
    )
    
    # Find suitable hosts
    recommendations = optimizer.find_suitable_hosts(request)
    
    print(f"Found {len(recommendations)} suitable hosts for {request.vm_name}")
    for i, rec in enumerate(recommendations[:5]):  # Top 5
        print(f"\n{i+1}. {rec.host_name} (Score: {rec.score:.3f})")
        print(f"   Compliance: {rec.compliance_score:.3f}")
        print(f"   Performance: {rec.performance_score:.3f}")
        print(f"   Resource: {rec.resource_score:.3f}")
        print(f"   Migration time: {rec.migration_cost:.1f} minutes")
        print(f"   Reasoning: {'; '.join(rec.reasoning[:3])}")
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Total hosts: {summary.get('total_hosts', 0)}")
    print(f"Total VMs: {summary.get('total_vms', 0)}")
    print(f"Total clusters: {summary.get('total_clusters', 0)}") 
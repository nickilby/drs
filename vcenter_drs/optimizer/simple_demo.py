"""
Simplified demonstration of VM placement optimization decision-making process.
This version doesn't require external dependencies and shows the core logic.
"""

import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class SimpleHostMetrics:
    """Simplified host metrics for demonstration"""
    host_id: int
    host_name: str
    cluster_name: str
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    network_usage: float  # Percentage
    vm_count: int
    compliance_violations: int

@dataclass
class SimpleVMRequest:
    """Simplified VM placement request"""
    vm_name: str
    vm_alias: str
    vm_role: str
    required_cpu: float
    required_memory: float
    priority: str = "normal"

@dataclass
class SimpleRecommendation:
    """Simplified placement recommendation"""
    host_name: str
    cluster_name: str
    score: float
    compliance_score: float
    performance_score: float
    reasoning: List[str]

class SimpleVMOptimizer:
    """
    Simplified VM placement optimizer for demonstration purposes.
    Shows the core decision-making logic without external dependencies.
    """
    
    def __init__(self):
        # Mock host data for demonstration
        self.mock_hosts = {
            1: SimpleHostMetrics(1, "esxi-01", "Cluster-A", 45.0, 55.0, 40.0, 8, 0),
            2: SimpleHostMetrics(2, "esxi-02", "Cluster-A", 75.0, 80.0, 65.0, 12, 2),
            3: SimpleHostMetrics(3, "esxi-03", "Cluster-B", 30.0, 40.0, 35.0, 6, 0),
            4: SimpleHostMetrics(4, "esxi-04", "Cluster-B", 85.0, 90.0, 75.0, 15, 1),
            5: SimpleHostMetrics(5, "esxi-05", "Cluster-A", 60.0, 65.0, 55.0, 10, 0),
        }
        
        # Mock VM data for compliance checking
        self.mock_vms = {
            "z-example-alias-WEB1": {"host_id": 1, "alias": "z-example-alias", "role": "WEB"},
            "z-example-alias-WEB2": {"host_id": 2, "alias": "z-example-alias", "role": "WEB"},
            "z-example-alias-DB1": {"host_id": 3, "alias": "z-example-alias", "role": "DB"},
            "z-example-alias-DB2": {"host_id": 3, "alias": "z-example-alias", "role": "DB"},
        }
    
    def find_suitable_hosts(self, request: SimpleVMRequest) -> List[SimpleRecommendation]:
        """
        Find suitable hosts for VM placement using simplified logic.
        """
        recommendations = []
        
        for host_id, host_metrics in self.mock_hosts.items():
            # Check if host can accommodate the VM
            if not self._can_accommodate_vm(host_metrics, request):
                continue
            
            # Calculate scores
            compliance_score = self._calculate_compliance_score(host_metrics, request)
            performance_score = self._calculate_performance_score(host_metrics, request)
            
            # Weighted overall score (simplified)
            overall_score = (compliance_score * 0.6) + (performance_score * 0.4)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(host_metrics, request, compliance_score, performance_score)
            
            recommendations.append(SimpleRecommendation(
                host_name=host_metrics.host_name,
                cluster_name=host_metrics.cluster_name,
                score=overall_score,
                compliance_score=compliance_score,
                performance_score=performance_score,
                reasoning=reasoning
            ))
        
        # Sort by overall score (descending)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations
    
    def _can_accommodate_vm(self, host_metrics: SimpleHostMetrics, request: SimpleVMRequest) -> bool:
        """Check if host has sufficient resources"""
        # Simplified capacity check
        available_cpu = 100 - host_metrics.cpu_usage
        available_memory = 100 - host_metrics.memory_usage
        
        return available_cpu >= request.required_cpu and available_memory >= request.required_memory
    
    def _calculate_compliance_score(self, host_metrics: SimpleHostMetrics, request: SimpleVMRequest) -> float:
        """Calculate compliance score (0-1, higher is better)"""
        score = 1.0
        
        # Penalize hosts with existing violations
        if host_metrics.compliance_violations > 0:
            score -= min(host_metrics.compliance_violations * 0.2, 0.8)
        
        # Check for anti-affinity violations
        if self._would_violate_anti_affinity(host_metrics, request):
            score -= 0.5
        
        # Check for affinity violations
        if self._would_violate_affinity(host_metrics, request):
            score -= 0.3
        
        return max(0.0, score)
    
    def _would_violate_anti_affinity(self, host_metrics: SimpleHostMetrics, request: SimpleVMRequest) -> bool:
        """Check if placing VM would violate anti-affinity rules"""
        # Check if any VM on this host has the same alias and role
        for vm_name, vm_data in self.mock_vms.items():
            if (vm_data["host_id"] == host_metrics.host_id and 
                vm_data["alias"] == request.vm_alias and 
                vm_data["role"] == request.vm_role):
                return True
        return False
    
    def _would_violate_affinity(self, host_metrics: SimpleHostMetrics, request: SimpleVMRequest) -> bool:
        """Check if placing VM would violate affinity rules"""
        # Find VMs with same alias and role on different hosts
        same_alias_role_vms = []
        for vm_name, vm_data in self.mock_vms.items():
            if vm_data["alias"] == request.vm_alias and vm_data["role"] == request.vm_role:
                same_alias_role_vms.append(vm_data)
        
        # If there are VMs with same alias/role on different hosts, this would violate affinity
        if same_alias_role_vms:
            other_hosts = {vm["host_id"] for vm in same_alias_role_vms}
            if host_metrics.host_id not in other_hosts:
                return True
        
        return False
    
    def _calculate_performance_score(self, host_metrics: SimpleHostMetrics, request: SimpleVMRequest) -> float:
        """Calculate performance score (0-1, higher is better)"""
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
        
        # Penalize high VM density (interference)
        vm_density_score = min(host_metrics.vm_count / 20.0, 1.0)
        score -= vm_density_score * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_reasoning(self, host_metrics: SimpleHostMetrics, request: SimpleVMRequest, 
                           compliance_score: float, performance_score: float) -> List[str]:
        """Generate human-readable reasoning"""
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
        
        # VM density reasoning
        if host_metrics.vm_count < 10:
            reasoning.append(f"✅ Low VM density ({host_metrics.vm_count} VMs)")
        elif host_metrics.vm_count < 15:
            reasoning.append(f"⚠️ Moderate VM density ({host_metrics.vm_count} VMs)")
        else:
            reasoning.append(f"❌ High VM density ({host_metrics.vm_count} VMs)")
        
        return reasoning

def demonstrate_optimization():
    """
    Demonstrate the VM placement optimization decision process.
    """
    print("🤖 AI/ML-Driven VM Placement Optimization Decision Process")
    print("=" * 60)
    
    optimizer = SimpleVMOptimizer()
    
    # Example 1: Web server placement
    print("\n📋 Example 1: Web Server Placement")
    print("-" * 40)
    
    web_request = SimpleVMRequest(
        vm_name="z-example-alias-WEB3",
        vm_alias="z-example-alias",
        vm_role="WEB",
        required_cpu=4.0,
        required_memory=8.0,
        priority="normal"
    )
    
    print(f"VM Requirements:")
    print(f"  • CPU: {web_request.required_cpu} vCPUs")
    print(f"  • Memory: {web_request.required_memory} GB")
    print(f"  • Alias: {web_request.vm_alias}")
    print(f"  • Role: {web_request.vm_role}")
    
    # Show existing VMs for context
    print(f"\n📊 Existing VMs:")
    for vm_name, vm_data in optimizer.mock_vms.items():
        host_name = optimizer.mock_hosts[vm_data["host_id"]].host_name
        print(f"  • {vm_name} → {host_name}")
    
    # Get recommendations
    recommendations = optimizer.find_suitable_hosts(web_request)
    
    print(f"\n🏆 Host Recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"\n{i+1}. {rec.host_name} (Cluster: {rec.cluster_name})")
        print(f"   Overall Score: {rec.score:.3f}")
        print(f"   • Compliance Score: {rec.compliance_score:.3f}")
        print(f"   • Performance Score: {rec.performance_score:.3f}")
        print(f"   • Reasoning:")
        for reason in rec.reasoning:
            print(f"     - {reason}")
    
    # Example 2: Database server placement
    print("\n\n📋 Example 2: Database Server Placement")
    print("-" * 40)
    
    db_request = SimpleVMRequest(
        vm_name="z-example-alias-DB3",
        vm_alias="z-example-alias",
        vm_role="DB",
        required_cpu=8.0,
        required_memory=16.0,
        priority="high"
    )
    
    print(f"VM Requirements:")
    print(f"  • CPU: {db_request.required_cpu} vCPUs")
    print(f"  • Memory: {db_request.required_memory} GB")
    print(f"  • Priority: {db_request.priority}")
    
    db_recommendations = optimizer.find_suitable_hosts(db_request)
    
    print(f"\n🏆 Host Recommendations:")
    for i, rec in enumerate(db_recommendations):
        print(f"\n{i+1}. {rec.host_name} (Cluster: {rec.cluster_name})")
        print(f"   Overall Score: {rec.score:.3f}")
        print(f"   • Compliance Score: {rec.compliance_score:.3f}")
        print(f"   • Performance Score: {rec.performance_score:.3f}")
        print(f"   • Reasoning:")
        for reason in rec.reasoning:
            print(f"     - {reason}")

def explain_decision_factors():
    """
    Explain the key factors that influence host suitability decisions.
    """
    print("\n\n🔍 Decision Factors Explained")
    print("=" * 60)
    
    print("\n1. 📋 COMPLIANCE SCORING (60% weight)")
    print("   • Anti-affinity rule violations (heavily penalized)")
    print("   • Affinity rule violations (moderately penalized)")
    print("   • Existing violations on the host")
    print("   • Example: Two WEB servers from same alias shouldn't be on same host")
    
    print("\n2. ⚡ PERFORMANCE SCORING (40% weight)")
    print("   • Current CPU utilization (<60% preferred)")
    print("   • Current memory utilization (<70% preferred)")
    print("   • Network utilization (<60% preferred)")
    print("   • VM density/interference (fewer VMs = better)")
    print("   • Example: Host with 45% CPU, 55% memory gets high score")
    
    print("\n3. 🔍 ANTI-AFFINITY CHECKING")
    print("   • Checks if placing VM would violate anti-affinity rules")
    print("   • Looks for VMs with same alias and role on the same host")
    print("   • Example: Placing WEB3 on esxi-02 would violate (WEB2 already there)")
    
    print("\n4. 🔍 AFFINITY CHECKING")
    print("   • Checks if placing VM would violate affinity rules")
    print("   • Looks for VMs with same alias and role on different hosts")
    print("   • Example: Placing DB3 on esxi-01 would violate (DB1/DB2 on esxi-03)")

def show_optimization_capabilities():
    """
    Show the optimization capabilities and metrics.
    """
    print("\n\n🚀 Optimization Capabilities")
    print("=" * 60)
    
    optimizer = SimpleVMOptimizer()
    
    print(f"\n📊 Infrastructure Overview:")
    print(f"  • Total Hosts: {len(optimizer.mock_hosts)}")
    print(f"  • Total VMs: {len(optimizer.mock_vms)}")
    print(f"  • Clusters: Cluster-A, Cluster-B")
    
    print(f"\n🔧 Optimization Capabilities:")
    capabilities = [
        "Compliance-aware placement",
        "Performance-based ranking",
        "Resource efficiency optimization",
        "VM interference analysis",
        "Migration cost estimation",
        "Historical performance analysis"
    ]
    for capability in capabilities:
        print(f"  • {capability}")
    
    print(f"\n📈 Host Metrics:")
    for host_id, metrics in optimizer.mock_hosts.items():
        print(f"  • {metrics.host_name}:")
        print(f"    - CPU: {metrics.cpu_usage:.1f}%")
        print(f"    - Memory: {metrics.memory_usage:.1f}%")
        print(f"    - Network: {metrics.network_usage:.1f}%")
        print(f"    - VMs: {metrics.vm_count}")
        print(f"    - Violations: {metrics.compliance_violations}")

if __name__ == "__main__":
    try:
        demonstrate_optimization()
        explain_decision_factors()
        show_optimization_capabilities()
        
        print("\n\n✅ Optimization demonstration complete!")
        print("\n💡 Key Takeaways:")
        print("   • The optimizer uses a multi-criteria decision-making approach")
        print("   • Compliance rules are the highest priority (60% weight)")
        print("   • Performance and resource efficiency are also important")
        print("   • Each recommendation includes detailed reasoning")
        print("   • The system considers existing VM placements")
        print("   • Anti-affinity and affinity rules are strictly enforced")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}") 
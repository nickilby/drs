"""
Test script to demonstrate VM placement optimization decision-making process.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import VMOptimizer, VMPlacementRequest, HostMetrics
import json

def demonstrate_optimization_decision_process():
    """
    Demonstrate how the AI/ML-driven VM placement optimizer decides suitable hosts.
    """
    print("🤖 AI/ML-Driven VM Placement Optimization Decision Process")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = VMOptimizer()
    
    # Example 1: Web server placement
    print("\n📋 Example 1: Web Server Placement")
    print("-" * 40)
    
    web_request = VMPlacementRequest(
        vm_name="z-example-alias-WEB1",
        vm_alias="z-example-alias",
        vm_role="WEB",
        required_cpu=4.0,
        required_memory=8192,  # 8GB
        required_storage=100,  # 100GB
        network_requirements=100,  # 100 Mbps
        priority="normal"
    )
    
    print(f"VM Requirements:")
    print(f"  • CPU: {web_request.required_cpu} vCPUs")
    print(f"  • Memory: {web_request.required_memory} MB")
    print(f"  • Storage: {web_request.required_storage} GB")
    print(f"  • Network: {web_request.network_requirements} Mbps")
    print(f"  • Alias: {web_request.vm_alias}")
    print(f"  • Role: {web_request.vm_role}")
    
    # Get recommendations
    recommendations = optimizer.find_suitable_hosts(web_request)
    
    print(f"\n🏆 Top 3 Host Recommendations:")
    for i, rec in enumerate(recommendations[:3]):
        print(f"\n{i+1}. {rec.host_name} (Cluster: {rec.cluster_name})")
        print(f"   Overall Score: {rec.score:.3f}")
        print(f"   • Compliance Score: {rec.compliance_score:.3f}")
        print(f"   • Performance Score: {rec.performance_score:.3f}")
        print(f"   • Resource Score: {rec.resource_score:.3f}")
        print(f"   • Migration Time: {rec.migration_cost:.1f} minutes")
        print(f"   • Reasoning:")
        for reason in rec.reasoning[:3]:
            print(f"     - {reason}")
        print(f"   • Estimated Impact:")
        for impact_type, value in rec.estimated_impact.items():
            print(f"     - {impact_type}: {value:.1f}")
    
    # Example 2: Database server placement
    print("\n\n📋 Example 2: Database Server Placement")
    print("-" * 40)
    
    db_request = VMPlacementRequest(
        vm_name="z-example-alias-DB1",
        vm_alias="z-example-alias",
        vm_role="DB",
        required_cpu=8.0,
        required_memory=16384,  # 16GB
        required_storage=500,  # 500GB
        network_requirements=200,  # 200 Mbps
        priority="high"
    )
    
    print(f"VM Requirements:")
    print(f"  • CPU: {db_request.required_cpu} vCPUs")
    print(f"  • Memory: {db_request.required_memory} MB")
    print(f"  • Storage: {db_request.required_storage} GB")
    print(f"  • Network: {db_request.network_requirements} Mbps")
    print(f"  • Priority: {db_request.priority}")
    
    db_recommendations = optimizer.find_suitable_hosts(db_request)
    
    print(f"\n🏆 Top 3 Host Recommendations:")
    for i, rec in enumerate(db_recommendations[:3]):
        print(f"\n{i+1}. {rec.host_name} (Cluster: {rec.cluster_name})")
        print(f"   Overall Score: {rec.score:.3f}")
        print(f"   • Compliance Score: {rec.compliance_score:.3f}")
        print(f"   • Performance Score: {rec.performance_score:.3f}")
        print(f"   • Resource Score: {rec.resource_score:.3f}")
        print(f"   • Migration Time: {rec.migration_cost:.1f} minutes")
        print(f"   • Reasoning:")
        for reason in rec.reasoning[:3]:
            print(f"     - {reason}")

def explain_decision_factors():
    """
    Explain the key factors that influence host suitability decisions.
    """
    print("\n\n🔍 Decision Factors Explained")
    print("=" * 60)
    
    print("\n1. 📋 COMPLIANCE SCORING (40% weight)")
    print("   • Anti-affinity rule violations (heavily penalized)")
    print("   • Affinity rule violations (moderately penalized)")
    print("   • Existing violations on the host")
    print("   • Dataset and pool placement rules")
    
    print("\n2. ⚡ PERFORMANCE SCORING (35% weight)")
    print("   • Current CPU utilization (<60% preferred)")
    print("   • Current memory utilization (<70% preferred)")
    print("   • Network utilization (<60% preferred)")
    print("   • Storage I/O utilization (<60% preferred)")
    print("   • VM interference score (lower is better)")
    print("   • Historical performance trends")
    
    print("\n3. 📊 RESOURCE EFFICIENCY SCORING (25% weight)")
    print("   • Resource balance (CPU vs Memory vs Network)")
    print("   • Utilization variance (balanced is better)")
    print("   • Overall resource utilization (30-70% optimal)")
    print("   • Storage capacity availability")
    
    print("\n4. 🔄 MIGRATION COST ESTIMATION")
    print("   • VM memory size")
    print("   • Network bandwidth")
    print("   • Storage latency")
    print("   • Current host load")
    
    print("\n5. 📈 PERFORMANCE IMPACT PREDICTION")
    print("   • Estimated CPU utilization change")
    print("   • Estimated memory utilization change")
    print("   • Predicted CPU ready time")
    print("   • Predicted memory ballooning")

def show_optimization_capabilities():
    """
    Show the optimization capabilities and metrics.
    """
    print("\n\n🚀 Optimization Capabilities")
    print("=" * 60)
    
    optimizer = VMOptimizer()
    summary = optimizer.get_optimization_summary()
    
    print(f"\n📊 Infrastructure Overview:")
    print(f"  • Total Hosts: {summary.get('total_hosts', 0)}")
    print(f"  • Total VMs: {summary.get('total_vms', 0)}")
    print(f"  • Total Clusters: {summary.get('total_clusters', 0)}")
    
    print(f"\n🔧 Optimization Capabilities:")
    for capability in summary.get('optimization_capabilities', []):
        print(f"  • {capability}")
    
    print(f"\n📈 Cluster Metrics:")
    for cluster_name, metrics in summary.get('cluster_metrics', {}).items():
        print(f"  • {cluster_name}:")
        print(f"    - Hosts: {metrics.get('host_count', 0)}")
        print(f"    - VMs: {metrics.get('vm_count', 0)}")
        print(f"    - Avg CPU: {metrics.get('avg_cpu_utilization', 0):.1f}%")
        print(f"    - Avg Memory: {metrics.get('avg_memory_utilization', 0):.1f}%")
        print(f"    - Resource Efficiency: {metrics.get('resource_efficiency', 0):.3f}")

def demonstrate_compliance_awareness():
    """
    Demonstrate how the optimizer considers compliance rules.
    """
    print("\n\n⚖️ Compliance-Aware Placement")
    print("=" * 60)
    
    print("\n🔍 Anti-Affinity Rule Checking:")
    print("   • Checks if placing VM would violate anti-affinity rules")
    print("   • Looks for VMs with same alias and role on the same host")
    print("   • Example: Two WEB servers from same alias shouldn't be on same host")
    
    print("\n🔍 Affinity Rule Checking:")
    print("   • Checks if placing VM would violate affinity rules")
    print("   • Looks for VMs with same alias and role on different hosts")
    print("   • Example: All DB servers from same alias should be on same host")
    
    print("\n🔍 Dataset Placement Rules:")
    print("   • Checks dataset-affinity and dataset-anti-affinity rules")
    print("   • Ensures VMs are placed on appropriate storage")
    print("   • Considers storage pool restrictions")
    
    print("\n🔍 Pool Anti-Affinity Rules:")
    print("   • Checks pool-anti-affinity rules")
    print("   • Ensures VMs aren't placed on same storage pool")
    print("   • Prevents single point of failure scenarios")

if __name__ == "__main__":
    try:
        demonstrate_optimization_decision_process()
        explain_decision_factors()
        show_optimization_capabilities()
        demonstrate_compliance_awareness()
        
        print("\n\n✅ Optimization demonstration complete!")
        print("\n💡 Key Takeaways:")
        print("   • The optimizer uses a multi-criteria decision-making approach")
        print("   • Compliance rules are the highest priority (40% weight)")
        print("   • Performance and resource efficiency are also important")
        print("   • Each recommendation includes detailed reasoning")
        print("   • Migration costs and performance impacts are estimated")
        print("   • The system learns from historical performance data")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("This is expected if the database is not available or configured.")
        print("The optimizer is designed to work with real vCenter data.") 
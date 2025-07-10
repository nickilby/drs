#!/usr/bin/env python3
"""
Debug script to check cluster mismatch between database and Prometheus
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vcenter_drs.db.metrics_db import MetricsDB
from vcenter_drs.ai_optimizer.data_collector import PrometheusDataCollector
from vcenter_drs.ai_optimizer.config import ai_config
import requests
import json

def check_cluster_mismatch():
    """Check what clusters are available in database vs Prometheus"""
    
    print("🔍 Checking cluster information mismatch...")
    
    # Check database clusters
    print("\n📊 Database Clusters:")
    db = MetricsDB()
    db.connect()
    cursor = db.conn.cursor()
    
    cursor.execute("SELECT id, name FROM clusters")
    db_clusters = cursor.fetchall()
    
    for cluster_id, cluster_name in db_clusters:
        print(f"  - {cluster_name} (ID: {cluster_id})")
    
    # Check hosts in database with their clusters
    print("\n🏠 Database Hosts by Cluster:")
    cursor.execute("""
        SELECT h.name as host_name, c.name as cluster_name
        FROM hosts h
        JOIN clusters c ON h.cluster_id = c.id
        ORDER BY c.name, h.name
    """)
    db_hosts = cursor.fetchall()
    
    cluster_hosts = {}
    for host_name, cluster_name in db_hosts:
        if cluster_name not in cluster_hosts:
            cluster_hosts[cluster_name] = []
        cluster_hosts[cluster_name].append(host_name)
    
    for cluster_name, hosts in cluster_hosts.items():
        print(f"  {cluster_name}: {len(hosts)} hosts")
        for host in hosts[:5]:  # Show first 5 hosts
            print(f"    - {host}")
        if len(hosts) > 5:
            print(f"    ... and {len(hosts) - 5} more")
    
    # Check Prometheus clusters
    print("\n📈 Prometheus Clusters:")
    try:
        # Query Prometheus for host metrics with cluster info
        response = requests.get("http://10.65.32.4:9090/api/v1/query?query=vmware_host_cpu_usage_average", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                prometheus_clusters = set()
                prometheus_hosts = {}
                
                for result in data['data']['result']:
                    host_name = result['metric'].get('host_name', '')
                    cluster_name = result['metric'].get('cluster_name', 'unknown')
                    
                    if cluster_name not in prometheus_clusters:
                        prometheus_clusters.add(cluster_name)
                        prometheus_hosts[cluster_name] = []
                    
                    if host_name not in prometheus_hosts[cluster_name]:
                        prometheus_hosts[cluster_name].append(host_name)
                
                for cluster_name in sorted(prometheus_clusters):
                    hosts = prometheus_hosts[cluster_name]
                    print(f"  {cluster_name}: {len(hosts)} hosts")
                    for host in hosts[:5]:  # Show first 5 hosts
                        print(f"    - {host}")
                    if len(hosts) > 5:
                        print(f"    ... and {len(hosts) - 5} more")
            else:
                print("  ❌ No data returned from Prometheus")
        else:
            print(f"  ❌ Prometheus query failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error querying Prometheus: {e}")
    
    # Check specific VM cluster
    print("\n🎯 Checking specific VM cluster:")
    vm_name = "z-actegy-WEB1"
    
    # Check database
    cursor.execute("""
        SELECT v.name as vm_name, c.name as cluster_name
        FROM vms v
        JOIN hosts h ON v.host_id = h.id
        JOIN clusters c ON h.cluster_id = c.id
        WHERE v.name = %s
    """, (vm_name,))
    vm_db_info = cursor.fetchone()
    
    if vm_db_info:
        print(f"  Database: VM '{vm_name}' is in cluster '{vm_db_info[1]}'")
    else:
        print(f"  Database: VM '{vm_name}' not found")
    
    # Check Prometheus
    try:
        data_collector = PrometheusDataCollector(ai_config)
        vm_cluster = data_collector.get_vm_cluster(vm_name)
        print(f"  Prometheus: VM '{vm_name}' is in cluster '{vm_cluster}'")
    except Exception as e:
        print(f"  Prometheus: Error getting VM cluster: {e}")
    
    # Test host filtering logic
    print("\n🔍 Testing host filtering logic:")
    target_cluster = "zengenti-man1"
    
    # Get all hosts from Prometheus
    try:
        response = requests.get("http://10.65.32.4:9090/api/v1/query?query=vmware_host_cpu_usage_average", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                all_hosts = []
                cluster_hosts = []
                
                for result in data['data']['result']:
                    host_name = result['metric'].get('host_name', '')
                    cluster_name = result['metric'].get('cluster_name', 'unknown')
                    
                    if host_name not in all_hosts:
                        all_hosts.append(host_name)
                    
                    if cluster_name == target_cluster and host_name not in cluster_hosts:
                        cluster_hosts.append(host_name)
                
                print(f"  Total hosts in Prometheus: {len(all_hosts)}")
                print(f"  Hosts in cluster '{target_cluster}': {len(cluster_hosts)}")
                
                if cluster_hosts:
                    print(f"  Cluster hosts: {', '.join(cluster_hosts)}")
                else:
                    print(f"  ❌ No hosts found in cluster '{target_cluster}'")
            else:
                print("  ❌ No data returned from Prometheus")
        else:
            print(f"  ❌ Prometheus query failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error testing host filtering: {e}")
    
    cursor.close()
    db.close()
    
    print("\n💡 Summary:")
    print("  - Database has cluster information from vCenter")
    print("  - Prometheus has cluster information from VMware metrics")
    print("  - These may not match if cluster names are different")
    print("  - The AI Model Predictions uses Prometheus data for host filtering")

if __name__ == "__main__":
    check_cluster_mismatch() 
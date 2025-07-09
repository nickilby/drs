# AI/ML-Driven VM Placement Optimization

## Overview

The VM Placement Optimizer is an intelligent system that uses multi-criteria decision-making to determine the most suitable hosts for VM placement. It considers compliance rules, performance metrics, resource efficiency, and operational factors to provide ranked recommendations.

## Decision-Making Process

### 1. Multi-Criteria Scoring System

The optimizer uses a weighted scoring system with three main criteria:

#### Compliance Score (40% weight)
- **Anti-affinity rule violations**: Heavily penalized (-0.5 score)
- **Affinity rule violations**: Moderately penalized (-0.3 score)
- **Existing violations**: Penalized based on count (-0.2 per violation)
- **Dataset/pool placement rules**: Considered for storage placement

#### Performance Score (35% weight)
- **CPU utilization**: Prefers <60%, penalizes >80%
- **Memory utilization**: Prefers <70%, penalizes >85%
- **Network utilization**: Prefers <60%, penalizes >80%
- **Storage I/O**: Prefers <60%, penalizes >80%
- **VM interference**: Lower score = better performance
- **Historical trends**: Bonus for consistently good performance

#### Resource Efficiency Score (25% weight)
- **Resource balance**: Prefers balanced CPU/Memory/Network utilization
- **Utilization variance**: Lower variance = better score
- **Optimal utilization**: 30-70% range preferred
- **Storage capacity**: Considers available storage

### 2. Host Suitability Assessment

For each host, the system:

1. **Checks resource capacity**:
   - Available CPU cores
   - Available memory
   - Available storage
   - Network bandwidth

2. **Evaluates compliance**:
   - Runs anti-affinity checks
   - Runs affinity checks
   - Checks dataset placement rules
   - Counts existing violations

3. **Analyzes performance**:
   - Current utilization metrics
   - Historical performance trends
   - VM interference patterns
   - Performance predictions

4. **Calculates efficiency**:
   - Resource balance analysis
   - Utilization optimization
   - Migration cost estimation

### 3. Recommendation Generation

Each recommendation includes:

- **Overall score** (0-1, higher is better)
- **Individual component scores** (compliance, performance, resource)
- **Detailed reasoning** with specific factors
- **Migration cost estimate** in minutes
- **Performance impact predictions**

## Key Decision Factors

### Compliance Rules
```
Anti-Affinity Example:
- VM: z-example-alias-WEB1
- Rule: WEB servers from same alias shouldn't be on same host
- Check: Are there other WEB VMs with same alias on this host?
- Penalty: -0.5 score if violation detected

Affinity Example:
- VM: z-example-alias-DB1
- Rule: DB servers from same alias should be on same host
- Check: Are there other DB VMs with same alias on different hosts?
- Penalty: -0.3 score if violation detected
```

### Performance Metrics
```
CPU Utilization:
- <60%: No penalty
- 60-80%: -0.1 penalty
- >80%: -0.3 penalty

Memory Utilization:
- <70%: No penalty
- 70-85%: -0.1 penalty
- >85%: -0.3 penalty

VM Interference:
- Score based on VM density and performance correlation
- Higher interference = lower performance score
```

### Resource Efficiency
```
Resource Balance:
- Calculate variance between CPU/Memory/Network utilization
- Lower variance = better balance score
- Formula: balance_score = max(0, 1 - variance * 2)

Optimal Utilization:
- 30-70%: Perfect score (1.0)
- <30%: Underutilized (score = utilization/0.3)
- >70%: Overutilized (score = max(0, 1 - (utilization - 0.7) / 0.3))
```

## Example Decision Process

### Scenario: Placing a Web Server VM

**VM Requirements:**
- CPU: 4 vCPUs
- Memory: 8GB
- Storage: 100GB
- Network: 100 Mbps
- Alias: z-example-alias
- Role: WEB

**Host A Analysis:**
```
Compliance Check:
- No existing WEB VMs with same alias ✓
- No anti-affinity violations ✓
- Compliance Score: 1.0

Performance Check:
- CPU: 45% (good) ✓
- Memory: 55% (good) ✓
- Network: 40% (good) ✓
- VM interference: 0.2 (low) ✓
- Performance Score: 0.95

Resource Efficiency:
- CPU/Memory/Network variance: 0.05 (balanced) ✓
- Overall utilization: 47% (optimal) ✓
- Resource Score: 0.92

Overall Score: (1.0 * 0.4) + (0.95 * 0.35) + (0.92 * 0.25) = 0.96
```

**Host B Analysis:**
```
Compliance Check:
- Has existing WEB VM with same alias ✗
- Anti-affinity violation detected ✗
- Compliance Score: 0.5

Performance Check:
- CPU: 75% (moderate) ⚠️
- Memory: 80% (high) ⚠️
- Network: 65% (moderate) ⚠️
- VM interference: 0.6 (high) ✗
- Performance Score: 0.65

Resource Efficiency:
- CPU/Memory/Network variance: 0.15 (unbalanced) ⚠️
- Overall utilization: 73% (high) ⚠️
- Resource Score: 0.45

Overall Score: (0.5 * 0.4) + (0.65 * 0.35) + (0.45 * 0.25) = 0.54
```

**Result:** Host A is recommended (score 0.96 vs 0.54)

## Advanced Features

### 1. Historical Performance Analysis
- Analyzes 7 days of performance data
- Identifies performance patterns and trends
- Uses machine learning to predict future performance

### 2. VM Interference Detection
- Analyzes performance correlation between VMs
- Identifies resource contention patterns
- Calculates interference scores based on VM density

### 3. Migration Cost Estimation
```
Migration Time = Base Time × Network Factor × Storage Factor

Base Time = VM Memory (GB) × 1 minute/GB
Network Factor = max(0.5, min(2.0, 1000 / bandwidth))
Storage Factor = max(0.5, min(2.0, 5 / latency))
```

### 4. Performance Impact Prediction
- Estimates new CPU utilization after placement
- Predicts memory utilization changes
- Forecasts CPU ready time increases
- Estimates memory ballooning

## Integration with Existing Systems

### Prometheus Metrics
The optimizer leverages existing Prometheus metrics:
- `cpu.usage.average`
- `mem.usage.average`
- `net.usage.average`
- `disk.usage.average`

### Database Integration
Uses the existing MySQL database schema:
- `hosts` table for host information
- `vms` table for VM placement
- `metrics` table for performance data
- `clusters` table for cluster relationships

### Rules Engine Integration
Integrates with the existing compliance rules engine:
- Uses `evaluate_rules()` for compliance checking
- Considers all rule types (anti-affinity, affinity, dataset, pool)
- Maintains compliance during optimization

## Usage Example

```python
from optimizer import VMOptimizer, VMPlacementRequest

# Initialize optimizer
optimizer = VMOptimizer()

# Create placement request
request = VMPlacementRequest(
    vm_name="z-example-alias-WEB1",
    vm_alias="z-example-alias",
    vm_role="WEB",
    required_cpu=4.0,
    required_memory=8192,
    required_storage=100,
    network_requirements=100,
    priority="normal"
)

# Get recommendations
recommendations = optimizer.find_suitable_hosts(request)

# Display top recommendations
for i, rec in enumerate(recommendations[:3]):
    print(f"{i+1}. {rec.host_name} (Score: {rec.score:.3f})")
    print(f"   Compliance: {rec.compliance_score:.3f}")
    print(f"   Performance: {rec.performance_score:.3f}")
    print(f"   Resource: {rec.resource_score:.3f}")
    print(f"   Migration: {rec.migration_cost:.1f} minutes")
    for reason in rec.reasoning:
        print(f"   - {reason}")
```

## Benefits

1. **Compliance-First Approach**: Ensures all placement decisions respect existing rules
2. **Performance Optimization**: Considers current and historical performance data
3. **Resource Efficiency**: Balances resource utilization across hosts
4. **Transparency**: Provides detailed reasoning for each recommendation
5. **Cost Awareness**: Estimates migration costs and performance impacts
6. **Scalability**: Handles large environments with multiple clusters and hosts

## Future Enhancements

1. **Machine Learning Models**: Train models on historical placement decisions
2. **Real-time Optimization**: Continuous optimization based on live metrics
3. **Predictive Analytics**: Forecast future resource needs and capacity planning
4. **Automated Remediation**: Automatic VM migration for optimization
5. **Cost Optimization**: Consider power consumption and licensing costs 
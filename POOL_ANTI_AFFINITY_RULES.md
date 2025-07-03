# Pool Anti-Affinity Rules

## Overview

Pool anti-affinity rules ensure that VMs are distributed across different ZFS pools for enhanced resilience and availability. This is particularly important for critical services where you want to avoid having multiple instances of the same service on the same storage pool.

## Problem Statement

In ZFS environments, you may have multiple datasets on the same pool:
- `HQS1WEB1` and `HQS1DAT1` are different datasets but on the same pool (`hqs1`)
- If the pool fails, all datasets on that pool become unavailable
- For resilience, you want critical services distributed across different pools

## Solution: Pool Anti-Affinity Rules

The new `pool-anti-affinity` rule type ensures VMs are distributed across different ZFS pools rather than just different datasets.

## Rule Configuration

### Basic Pool Anti-Affinity Rule

```json
{
  "type": "pool-anti-affinity",
  "role": ["WEB"],
  "pool_pattern": ["HQ", "L", "M"]
}
```

This rule ensures that all VMs with role "WEB" are distributed across different pools matching the patterns. The pattern `["HQ", "L", "M"]` will match:
- Any pool starting with "HQ" (e.g., hqs1, hqs2, hqs3, etc.)
- Any pool starting with "L" (e.g., l1, l2, l3, etc.)
- Any pool starting with "M" (e.g., m1, m2, m3, etc.)

### Name Pattern Based Rule

```json
{
  "type": "pool-anti-affinity",
  "name_pattern": "webserver",
  "pool_pattern": ["HQ", "L", "M"]
}
```

This rule ensures that VMs with "webserver" in their name are distributed across different pools.

### Multiple Role Rule

```json
{
  "type": "pool-anti-affinity",
  "role": ["LB", "CACHE"],
  "pool_pattern": ["HQ", "L", "M"]
}
```

This rule ensures that load balancers and cache servers are distributed across different pools.

## Pool Name Extraction

The system automatically extracts pool names from dataset names using these patterns:

1. **Letters before numbers**: `HQS5WEB1` → `HQS5`
2. **Letters before dash**: `PROD-WEB1` → `PROD`
3. **First word**: `POOL1_DATASET` → `POOL1`

### Examples

| Dataset Name | Extracted Pool |
|--------------|----------------|
| HQS5WEB1     | HQS5           |
| HQS5DAT1     | HQS5           |
| HQS6WEB1     | HQS6           |
| PROD-WEB1    | PROD           |
| TEST-DAT1    | TEST           |
| POOL1_DS     | POOL1          |

## Violation Examples

### Scenario 1: Multiple Web Servers on Same Pool

**VMs:**
- `z-app1-web1` on dataset `HQS5WEB1` (pool: HQS5)
- `z-app1-web2` on dataset `HQS5WEB2` (pool: HQS5)

**Rule:**
```json
{
  "type": "pool-anti-affinity",
  "role": ["WEB"],
  "pool_pattern": ["HQ", "L", "M"]
}
```

**Violation Message:**
```
Pool Anti-Affinity Violation
Rule: VMs with role(s) ['WEB'] must NOT be on the same ZFS pool matching ['HQ', 'L', 'M']
Pool: hqs1
VMs on this pool:
  - z-app1-web1
  - z-app1-web2
Suggestions:
  - Move VM z-app1-web2 to a different ZFS pool
```

### Scenario 2: Load Balancers on Same Pool

**VMs:**
- `z-app1-lb1` on dataset `HQS5WEB1` (pool: HQS5)
- `z-app1-lb2` on dataset `HQS5WEB2` (pool: HQS5)

**Rule:**
```json
{
  "type": "pool-anti-affinity",
  "role": ["LB"],
  "pool_pattern": ["HQ", "L", "M"]
}
```

**Violation Message:**
```
Pool Anti-Affinity Violation
Rule: VMs with role(s) ['LB'] must NOT be on the same ZFS pool matching ['HQ', 'L', 'M']
Pool: hqs1
VMs on this pool:
  - z-app1-lb1
  - z-app1-lb2
Suggestions:
  - Move VM z-app1-lb2 to a different ZFS pool
```

## Implementation Details

### Database Schema Changes

The `datasets` table now includes a `pool_name` column:

```sql
CREATE TABLE IF NOT EXISTS datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    pool_name VARCHAR(255),
    description TEXT,
    INDEX idx_pool (pool_name)
) ENGINE=InnoDB;
```

### Rules Engine Integration

The rules engine now includes pool anti-affinity checking:

1. **Pool Extraction**: Automatically extracts pool names from dataset names
2. **Grouping**: Groups VMs by pool for violation detection
3. **Pattern Matching**: Supports multiple pool patterns for flexible matching
4. **Violation Reporting**: Provides clear violation messages with suggestions

### Performance Considerations

- Pool name extraction is done in-memory for efficiency
- Database queries include pool information in a single join
- Index on `pool_name` column for fast lookups

## Best Practices

### 1. Pool Naming Convention

Use consistent naming conventions for your ZFS pools:
- `HQS5`, `HQS6`, `HQS7` for different hardware systems
- `PROD`, `STAGE`, `TEST` for different environments
- `POOL1`, `POOL2`, `POOL3` for generic pools

### 2. Rule Granularity

Create specific rules for different service types:
```json
[
  {
    "type": "pool-anti-affinity",
    "role": ["WEB"],
    "pool_pattern": ["HQ", "L", "M"]
  },
  {
    "type": "pool-anti-affinity",
    "role": ["DB"],
    "pool_pattern": ["HQ", "L", "M"]
  },
  {
    "type": "pool-anti-affinity",
    "role": ["LB"],
    "pool_pattern": ["HQ", "L", "M"]
  }
]
```

### 3. Monitoring and Alerts

- Monitor pool anti-affinity violations in your dashboard
- Set up alerts for critical services that violate pool distribution
- Regular compliance checks to ensure rules are being followed

### 4. Migration Planning

When migrating VMs to comply with pool anti-affinity rules:
1. Identify target pools with available capacity
2. Plan maintenance windows for migrations
3. Test migrations in non-production first
4. Update documentation and runbooks

## Troubleshooting

### Common Issues

1. **Pool Name Not Extracted**: Check dataset naming convention
2. **False Positives**: Verify pool patterns match your actual pool names
3. **Performance Impact**: Monitor query performance with large datasets

### Debugging

Enable debug logging to see pool extraction details:
```python
# In rules_engine.py, add debug logging
print(f"Dataset: {dataset_name}, Extracted Pool: {pool_name}")
```

### Testing

Test your rules with sample data:
```bash
# Run compliance check with specific cluster
python -c "from vcenter_drs.rules.rules_engine import evaluate_rules; evaluate_rules('YourCluster')"
```

## Migration from Dataset Rules

If you currently use dataset anti-affinity rules, consider migrating to pool anti-affinity:

**Before (Dataset Anti-Affinity):**
```json
{
  "type": "dataset-anti-affinity",
  "role": ["WEB"],
  "dataset_pattern": ["WEB1", "WEB2", "WEB3"]
}
```

**After (Pool Anti-Affinity):**
```json
{
  "type": "pool-anti-affinity",
  "role": ["WEB"],
  "pool_pattern": ["HQ", "L", "M"]
}
```

This provides better resilience as it ensures distribution across different storage pools rather than just different datasets. 
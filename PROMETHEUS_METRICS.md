# Prometheus Metrics for vCenter DRS Compliance Dashboard

## Metrics Endpoint

The vCenter DRS Compliance Dashboard exposes Prometheus metrics on port 8081:

```
http://localhost:8081/metrics
```

## Available Metrics

### Service Status
- **`vcenter_drs_service_up`** (gauge)
  - Description: Service status (1=up, 0=down)
  - Value: 1 when service is running, 0 when stopped

### Rule Violations
- **`vcenter_drs_rule_violations_total`** (counter)
  - Description: Total rule violations by type
  - Labels: `rule_type` (anti-affinity, dataset-affinity, etc.)
  - Increments each time a violation is detected

### Infrastructure Counts
- **`vcenter_drs_vm_count`** (gauge)
  - Description: Total number of VMs monitored
  - Updated after each data collection

- **`vcenter_drs_host_count`** (gauge)
  - Description: Total number of hosts monitored
  - Updated after each data collection

### Performance Metrics
- **`vcenter_drs_last_collection_timestamp`** (gauge)
  - Description: Timestamp of last metrics collection
  - Value: Unix timestamp

- **`vcenter_drs_compliance_check_duration_seconds`** (histogram)
  - Description: Duration of compliance checks
  - Buckets: 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, +Inf

- **`vcenter_drs_uptime_seconds`** (gauge)
  - Description: Service uptime in seconds
  - Updated every minute

## Example Prometheus Query

```promql
# Get current violation counts by rule type
vcenter_drs_rule_violations_total

# Get service uptime in hours
vcenter_drs_uptime_seconds / 3600

# Get average compliance check duration
rate(vcenter_drs_compliance_check_duration_seconds_sum[5m]) / rate(vcenter_drs_compliance_check_duration_seconds_count[5m])
```

## Configuration

The metrics server starts automatically when the Streamlit app starts. No additional configuration is required.

## Port Configuration

- **Streamlit App**: Port 8080
- **Prometheus Metrics**: Port 8081

Both ports are configurable in the systemd service file if needed. 
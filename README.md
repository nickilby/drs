# vCenter DRS Compliance Dashboard

A comprehensive VMware vCenter compliance monitoring and optimization system that enforces VM placement rules based on affinity, anti-affinity, and dataset requirements.

## 🚀 Features

- **Real-time vCenter Integration**: Polls vCenter APIs for VM and Host metrics
- **Compliance Rules Engine**: Enforces complex VM placement rules
- **Dataset Affinity**: Ensures VMs are placed on appropriate storage datastores
- **Host Affinity/Anti-affinity**: Manages VM distribution across hosts
- **Web Dashboard**: Streamlit-based UI for compliance monitoring
- **Automated Cleanup**: Removes stale VM records automatically
- **Prometheus Metrics**: Exposes monitoring metrics for observability
- **Systemd Service**: Runs as a system service with auto-restart
- **Automated Data Collection**: Cron jobs for background data refresh

## 📋 Requirements

- Python 3.8+
- VMware vCenter Server
- MySQL Database
- Network access to vCenter APIs

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd vcenter_drs
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r vcenter_drs/requirements.txt
   ```

4. **Configure credentials:**
   Create `vcenter_drs/credentials.json`:
   ```json
   {
     "host": "your-vcenter-server.com",
     "username": "your-username",
     "password": "your-password",
     "db_host": "localhost",
     "db_user": "your-db-user",
     "db_password": "your-db-password",
     "db_database": "vcenter_drs"
   }
   ```

## 🏃‍♂️ Quick Start

1. **Initialize the database:**
   ```bash
   python -c "from vcenter_drs.db.metrics_db import MetricsDB; db = MetricsDB(); db.connect(); db.init_schema(); db.close()"
   ```

2. **Test vCenter connectivity:**
   ```bash
   python vcenter_drs/main.py check
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run vcenter_drs/streamlit_app.py
   ```

## 🚀 Production Deployment

### Systemd Service (Recommended)

1. **Deploy as systemd service:**
   ```bash
   ./deploy.sh
   ```

2. **Manage the service:**
   ```bash
   sudo systemctl start vcenter-drs
   sudo systemctl stop vcenter-drs
   sudo systemctl restart vcenter-drs
   sudo systemctl status vcenter-drs
   ```

### Automated Data Collection

Set up cron jobs for automated data collection and compliance checking:

```bash
./setup_cron.sh
```

This creates two cron jobs:
- **Data Collection**: Every 15 minutes (`refresh_vcenter_data.py`)
- **Compliance Checking**: Every 5 minutes (`check_compliance.py`)

## 📊 Usage

### Web Dashboard
- **Refresh Data**: Collects latest metrics from vCenter
- **Display Results**: Shows compliance violations by cluster
- **Cluster Filtering**: View violations for specific clusters
- **Exception Management**: Add/remove compliance exceptions
- **Rule Management**: View and manage compliance rules

### Command Line
```bash
# Collect metrics
python vcenter_drs/main.py

# Check connectivity
python vcenter_drs/main.py check
```

### Prometheus Metrics

The application exposes Prometheus metrics on port 8081:

```bash
curl http://localhost:8081/metrics
```

Available metrics:
- `vcenter_drs_service_up` - Service status (1=up, 0=down)
- `vcenter_drs_rule_violations_total` - Current rule violations by type
- `vcenter_drs_vm_count` - Total number of VMs monitored
- `vcenter_drs_host_count` - Total number of hosts monitored
- `vcenter_drs_last_collection_timestamp` - Timestamp of last metrics collection
- `vcenter_drs_compliance_check_duration_seconds` - Duration of compliance checks
- `vcenter_drs_uptime_seconds` - Service uptime in seconds

## 🔧 Configuration

### Rules Configuration
Edit `vcenter_drs/rules/rules.json` to define compliance rules:

```json
[
  {
    "type": "dataset-affinity",
    "name_pattern": "-dr-",
    "dataset_pattern": ["TRA"]
  },
  {
    "type": "anti-affinity",
    "level": "host",
    "role": "CACHE"
  }
]
```

### Rule Types
- **`dataset-affinity`**: Ensures VMs are on specific datastores
- **`affinity`**: Keeps VMs together on same host/cluster
- **`anti-affinity`**: Prevents VMs from being on same host/cluster

### Port Configuration
- **Streamlit Dashboard**: Port 8080
- **Prometheus Metrics**: Port 8081

Both ports are configurable in the systemd service file.

## 🏗️ Architecture

```
vcenter_drs/
├── api/                    # vCenter API integration
│   ├── vcenter_client_pyvomi.py
│   └── collect_and_store_metrics.py
├── db/                     # Database layer
│   └── metrics_db.py
├── rules/                  # Compliance rules engine
│   ├── rules_engine.py
│   └── rules.json
├── streamlit_app.py        # Web dashboard
├── main.py                 # CLI entry point
├── refresh_vcenter_data.py # Automated data collection
├── check_compliance.py     # Automated compliance checking
└── vcenter-drs.service     # Systemd service definition
```

## 🔍 Monitoring

The system tracks:
- **VM Metrics**: CPU, memory usage
- **Host Metrics**: Resource utilization
- **Compliance Violations**: Rule violations by cluster
- **Dataset Placement**: Storage datastore assignments
- **Service Health**: Uptime and performance metrics

### Log Files
- **Service Logs**: `sudo journalctl -u vcenter-drs -f`
- **Data Collection**: `/var/log/vcenter-drs-refresh.log`
- **Compliance Checks**: `/var/log/vcenter-drs-compliance.log`

## 🧪 Development

### Running Tests
```bash
cd vcenter_drs
pytest tests/
```

### Code Formatting
```bash
black vcenter_drs/
flake8 vcenter_drs/
mypy vcenter_drs/
```

### Type Checking
```bash
mypy vcenter_drs/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 📚 Additional Documentation

- [Prometheus Metrics Documentation](PROMETHEUS_METRICS.md)
- [Deployment Guide](deploy.sh)
- [Cron Setup Guide](setup_cron.sh)

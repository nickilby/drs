# vCenter DRS Compliance Dashboard

A comprehensive VMware vCenter compliance monitoring and optimization system that enforces VM placement rules based on affinity, anti-affinity, and dataset requirements, enhanced with **AI-powered VM placement optimization**.

## 🚀 Features

- **Real-time vCenter Integration**: Polls vCenter APIs for VM and Host metrics
- **Compliance Rules Engine**: Enforces complex VM placement rules
- **Dataset Affinity**: Ensures VMs are placed on appropriate storage datastores
- **Host Affinity/Anti-affinity**: Manages VM distribution across hosts
- **🤖 AI-Powered VM Placement**: Machine learning optimization for optimal VM placement
- **Web Dashboard**: Streamlit-based UI for compliance monitoring and AI optimization
- **Automated Cleanup**: Removes stale VM records automatically
- **Prometheus Metrics**: Exposes monitoring metrics for observability
- **Systemd Service**: Runs as a system service with auto-restart
- **Automated Data Collection**: Cron jobs for background data refresh

## 🤖 AI Optimizer Features

### **AI-Powered VM Placement**
- **Machine Learning Models**: Random Forest and Neural Network models for placement optimization
- **Performance-Based Recommendations**: Analyzes CPU, RAM, I/O, and ready time metrics
- **Intelligent Scoring**: Ranks host candidates based on optimization criteria
- **Trend Analysis**: Uses historical data to predict optimal placement
- **Configurable Optimization**: Adjustable parameters for different environments

### **AI Model Training**
- **Automated Training**: Train models on current infrastructure data
- **Prometheus Integration**: Uses real performance metrics for training
- **Configurable Parameters**: Adjust training episodes, learning rate, batch size, and exploration rate
- **Model Persistence**: Saves trained models for reuse
- **Performance Monitoring**: Tracks model accuracy and performance

### **AI Configuration Management**
- **Persistent Settings**: Configuration saved to JSON file
- **Real-time Updates**: Changes applied immediately after training
- **Time Window Analysis**: Configurable analysis periods (1-24 hours for CPU, 1-7 days for storage)
- **Optimization Parameters**: Adjustable ideal host usage ranges and priority weights

## 📋 Requirements

- Python 3.8+
- VMware vCenter Server
- MySQL Database
- Network access to vCenter APIs
- **Prometheus Server** (for AI optimization features)
- **PyTorch** (for neural network models)

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

4. **Configure AI Optimizer:**
   - Navigate to "AI Config" page
   - Set Prometheus URL and analysis parameters
   - Click "Save Configuration"
   - Click "Train Models" to initialize AI models

## 🤖 AI Optimizer Usage

### **AI Configuration Page**
Navigate to the "AI Config" page to configure AI optimization parameters:

#### **Prometheus Configuration**
- **Prometheus URL**: URL of your Prometheus server
- **Port**: Prometheus port (default: 9090)
- **Timeout**: Connection timeout in seconds
- **Retry Attempts**: Number of connection retries

#### **Analysis Windows**
- **CPU Trend (hours)**: Historical CPU analysis period (1-24 hours)
- **RAM Trend (hours)**: Historical RAM analysis period (1-24 hours)
- **I/O Trend (days)**: Historical I/O analysis period (1-7 days)
- **Storage Trend (days)**: Historical storage analysis period (1-7 days)
- **Ready Time Window (hours)**: CPU ready time analysis period (1-24 hours)

#### **Optimization Parameters**
- **Ideal Host Usage Min/Max (%)**: Target host utilization range (30-70% recommended)
- **Priority Weights**: CPU, RAM, Ready Time, and I/O priority weights
- **Max Recommendations**: Maximum number of placement recommendations

#### **ML Model Settings**
- **Training Episodes**: Number of training cycles (1000-10000)
- **Learning Rate**: Model learning speed (0.0001-0.1)
- **Batch Size**: Training batch size (8-128)
- **Exploration Rate**: Model exploration vs exploitation balance (0.01-0.5)

### **AI Optimizer Page**
Navigate to the "AI Optimizer" page to get VM placement recommendations:

1. **Select VM**: Choose a VM for placement analysis
2. **Set Cluster Filter**: Optionally filter by specific cluster
3. **Generate Recommendations**: Click to get AI-powered placement suggestions

#### **Recommendation Features**
- **Host Ranking**: AI-scored host candidates
- **Current Metrics**: Host performance before placement
- **Projected Metrics**: Predicted performance after placement
- **VM Metrics**: Detailed VM performance analysis
- **AI Reasoning**: Human-readable explanation of recommendations
- **Action Buttons**: Apply placement recommendations

### **Model Training**
Train AI models with current infrastructure data:

1. **Configure Settings**: Set analysis windows and optimization parameters
2. **Save Configuration**: Persist settings to JSON file
3. **Train Models**: Click "Train Models" to build AI models
4. **Monitor Progress**: Watch training progress and results
5. **Use Recommendations**: Generate placement recommendations with trained models

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
- **🤖 AI Optimizer**: Get AI-powered VM placement recommendations
- **🤖 AI Config**: Configure AI optimization parameters

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
  },
  {
    "type": "pool-anti-affinity",
    "role": ["WEB", "LB", "CACHE"],
    "pool_pattern": ["HQ", "L", "M"]
  }
]
```

### Rule Types
- **`dataset-affinity`**: Ensures VMs are on specific datastores
- **`dataset-anti-affinity`**: Prevents VMs from being on the same datastore
- **`pool-anti-affinity`**: Prevents VMs from being on the same ZFS pool (for resilience)
- **`affinity`**: Keeps VMs together on same host/cluster
- **`anti-affinity`**: Prevents VMs from being on same host/cluster

### AI Configuration
AI settings are automatically saved to `ai_optimizer/custom_config.json`:

```json
{
  "prometheus": {
    "url": "http://prometheus.zengenti.com",
    "port": 9090,
    "timeout": 30,
    "retry_attempts": 3
  },
  "analysis": {
    "cpu_trend_hours": 24,
    "ram_trend_hours": 6,
    "io_trend_days": 2,
    "storage_trend_days": 3,
    "ready_time_window": 1
  },
  "optimization": {
    "ideal_host_usage_min": 0.30,
    "ideal_host_usage_max": 0.70,
    "cpu_priority_weight": 0.6,
    "ram_priority_weight": 1.0,
    "ready_time_priority_weight": 0.8,
    "io_priority_weight": 0.4,
    "max_recommendations": 10
  },
  "ml": {
    "training_episodes": 1000,
    "learning_rate": 0.001,
    "batch_size": 32,
    "exploration_rate": 0.1
  }
}
```

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
├── ai_optimizer/           # AI optimization engine
│   ├── config.py          # AI configuration management
│   ├── data_collector.py  # Prometheus data collection
│   ├── ml_engine.py       # Machine learning models
│   ├── optimization_engine.py # Main optimization logic
│   └── exceptions.py      # AI-specific exceptions
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
- **🤖 AI Model Performance**: Training accuracy and prediction quality

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

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

### Quick Start for Contributors

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass (`pytest`)
6. Submit a pull request using our [PR template](.github/pull_request_template.md)

### Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

### Reporting Issues

Please use our [issue template](.github/ISSUE_TEMPLATE/issue_template.md) when reporting bugs or requesting features.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue using our [issue template](.github/ISSUE_TEMPLATE/issue_template.md)

### Security Issues

If you discover a security vulnerability, please see our [Security Policy](SECURITY.md) for reporting guidelines. **Do not** create a public issue for security vulnerabilities.

## 📚 Additional Documentation

- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community guidelines
- [Security Policy](SECURITY.md) - Security vulnerability reporting
- [Changelog](CHANGELOG.md) - Version history and changes
- [Prometheus Metrics Documentation](PROMETHEUS_METRICS.md)
- [Deployment Guide](deploy.sh)
- [Cron Setup Guide](setup_cron.sh)

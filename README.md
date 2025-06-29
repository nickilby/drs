# vCenter DRS Compliance Dashboard

A comprehensive VMware vCenter compliance monitoring and optimization system that enforces VM placement rules based on affinity, anti-affinity, and dataset requirements.

## 🚀 Features

- **Real-time vCenter Integration**: Polls vCenter APIs for VM and Host metrics
- **Compliance Rules Engine**: Enforces complex VM placement rules
- **Dataset Affinity**: Ensures VMs are placed on appropriate storage datastores
- **Host Affinity/Anti-affinity**: Manages VM distribution across hosts
- **Web Dashboard**: Streamlit-based UI for compliance monitoring
- **Automated Cleanup**: Removes stale VM records automatically

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
   pip install -r requirements.txt
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
   python -c "from db.metrics_db import MetricsDB; db = MetricsDB(); db.connect(); db.init_schema(); db.close()"
   ```

2. **Test vCenter connectivity:**
   ```bash
   python main.py check
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Usage

### Web Dashboard
- **Refresh Data**: Collects latest metrics from vCenter
- **Display Results**: Shows compliance violations by cluster
- **Cluster Filtering**: View violations for specific clusters

### Command Line
```bash
# Collect metrics
python main.py

# Check connectivity
python main.py check
```

## 🔧 Configuration

### Rules Configuration
Edit `rules/rules.json` to define compliance rules:

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
└── main.py                 # CLI entry point
```

## 🔍 Monitoring

The system tracks:
- **VM Metrics**: CPU, memory usage
- **Host Metrics**: Resource utilization
- **Compliance Violations**: Rule violations by cluster
- **Dataset Placement**: Storage datastore assignments

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

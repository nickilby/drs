# vCenter DRS Compliance Dashboard

A robust, extensible compliance and remediation platform for VMware vCenter environments.  
It enforces VM placement rules (affinity, anti-affinity, dataset/pool requirements), provides a Streamlit-based dashboard, exposes Prometheus metrics, and supports automated remediation and exception management.

---

## 🚀 Features

- **Real-time vCenter Integration:** Polls vCenter APIs for VM/Host metrics.
- **Compliance Rules Engine:** Enforces complex VM placement rules (host, storage, pool).
- **Automated Remediation:** Triggers playbooks for host/storage violations.
- **Exception & Rule Management:** UI for adding/removing exceptions and rules.
- **Prometheus Metrics:** Exposes service and compliance metrics.
- **Systemd & Cron Support:** Runs as a service, supports scheduled data collection.
- **Streamlit Dashboard:** Modern, interactive web UI.
- **CI/CD:** Automated testing and linting via GitHub Actions.

---

## 📋 Requirements

- Python 3.8+
- VMware vCenter Server (API access)
- MySQL Database
- Network access to vCenter and DB
- Linux (for systemd/cron deployment)
- [Optional] Prometheus for metrics scraping

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd vcenter_drs
   ```

2. **Create a virtual environment:**
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

---

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

---

## 🚀 Production Deployment

### Systemd Service

1. **Deploy as a systemd service:**
   ```bash
   ./deploy.sh
   ```

2. **Manage the service:**
   ```bash
   sudo systemctl start vcenter-drs
   sudo systemctl status vcenter-drs
   ```

### Automated Data Collection

Set up cron jobs:
```bash
./setup_cron.sh
```

---

## 📊 Usage

### Web Dashboard
- **Compliance Results:** View violations by cluster.
- **Remediation:** Trigger playbooks for violations.
- **Exception Management:** Add/remove exceptions.
- **Rule Management:** Add/delete rules.
- **VM Rule Validator:** Test rules for a given VM.

### Command Line
```bash
python vcenter_drs/main.py
python vcenter_drs/main.py check
```

### Prometheus Metrics
Metrics exposed on port 8081:
```bash
curl http://localhost:8081/metrics
```

---

## 🔧 Configuration

### Rules
Edit `vcenter_drs/rules/rules.json`:
```json
[
  {
    "type": "dataset-affinity",
    "level": "storage",
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
- `"level": "host"` → host playbook
- `"level": "storage"` → storage playbook

---

## 🏗️ Architecture

```
vcenter_drs/
├── api/
├── db/
├── rules/
├── streamlit_app.py
├── main.py
├── refresh_vcenter_data.py
├── check_compliance.py
└── vcenter-drs.service
```

---

## 🧪 Development & Testing

### Running Tests
```bash
pytest drs/vcenter_drs/tests/
```

### Linting & Type Checking
```bash
black vcenter_drs/
flake8 vcenter_drs/
mypy vcenter_drs/
```

### CI/CD
- GitHub Actions runs tests and linting on every push/PR.
- See `.github/workflows/ci.yml`.

---

## 💡 Coding Best Practices

- **Follow PEP8** for Python code style.
- **Write tests** for all new features and bugfixes.
- **Use type hints** and run `mypy` for static type checking.
- **Document** all public functions and modules.
- **Keep secrets out of version control** (`credentials.json` is in `.gitignore`).
- **Use environment variables** for sensitive or environment-specific config when possible.
- **Modularize code**: keep logic, UI, and data access separate.
- **Review PRs** before merging; use feature branches.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/expand tests
5. Submit a pull request

---

## 📝 License

MIT License

---

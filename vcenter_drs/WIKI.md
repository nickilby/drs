# vCenter DRS Compliance Dashboard – Wiki

## Overview

The vCenter DRS Compliance Dashboard is a modern, extensible platform for monitoring, enforcing, and remediating VM placement policies in VMware vCenter environments. It provides a web dashboard, automated remediation, and deep integration with vCenter and Prometheus.

---

## Key Features

- Real-time compliance monitoring
- Automated remediation (host/storage)
- Exception and rule management UI
- Prometheus metrics for observability
- Systemd/cron support for automation
- CI/CD with automated testing

---

## Getting Started

See the [README](README.md) for installation and setup instructions.

---

## Architecture

- **Streamlit App:** User interface for compliance, remediation, and management.
- **Rules Engine:** Evaluates VM placement against rules.
- **Database:** Stores VM/host/cluster data and exceptions.
- **API Integration:** Collects data from vCenter and triggers remediation.
- **Prometheus Metrics:** Exposes service and compliance metrics.

---

## Best Practices

- Use feature branches and PRs for all changes.
- Write and run tests for all new code.
- Keep credentials and secrets out of version control.
- Use the UI for rule and exception management.
- Monitor Prometheus metrics for system health.

---

## FAQ

**Q: How do I add a new rule?**  
A: Use the Rule Management tab in the dashboard or edit `rules.json` directly.

**Q: How do I trigger remediation?**  
A: Use the "Remediate/Fix" button on a violation in the dashboard.

**Q: How do I run tests?**  
A: `pytest drs/vcenter_drs/tests/`

**Q: How do I deploy in production?**  
A: Use the provided `deploy.sh` and systemd service.

---

## Support

- [GitHub Issues](../issues)
- [Prometheus Metrics Documentation](PROMETHEUS_METRICS.md)
- [Deployment Guide](deploy.sh)

--- 
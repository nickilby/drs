#!/bin/bash
set -e

# Copy service and timer files
sudo cp drs-refresh.service /etc/systemd/system/
sudo cp drs-refresh.timer /etc/systemd/system/
sudo cp drs-compliance.service /etc/systemd/system/
sudo cp drs-compliance.timer /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start timers
sudo systemctl enable drs-refresh.timer
sudo systemctl start drs-refresh.timer
sudo systemctl enable drs-compliance.timer
sudo systemctl start drs-compliance.timer

# Show status
systemctl list-timers | grep drs

echo "Systemd jobs for DRS have been installed and started." 
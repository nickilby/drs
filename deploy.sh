#!/bin/bash

# vCenter DRS Deployment Script
set -e

echo "🚀 Deploying vCenter DRS Compliance Dashboard..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root"
   exit 1
fi

# Variables
SERVICE_NAME="vcenter-drs"
SERVICE_FILE="vcenter-drs.service"
APP_DIR="/home/headhoncho/drs"
VENV_DIR="$APP_DIR/venv"
APP_SUBDIR="$APP_DIR/vcenter_drs"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "Please run: python -m venv $VENV_DIR"
    exit 1
fi

# Check if app directory exists
if [ ! -d "$APP_SUBDIR" ]; then
    echo "❌ App directory not found at $APP_SUBDIR"
    exit 1
fi

# Check if credentials file exists
if [ ! -f "$APP_SUBDIR/credentials.json" ]; then
    echo "⚠️  Warning: credentials.json not found"
    echo "Please create $APP_SUBDIR/credentials.json with your vCenter credentials"
fi

# Install systemd service
echo "📦 Installing systemd service..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start service
echo "🔧 Enabling and starting service..."
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl start "$SERVICE_NAME"

# Check service status
echo "📊 Service status:"
sudo systemctl status "$SERVICE_NAME" --no-pager -l

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Useful commands:"
echo "  Check status:    sudo systemctl status $SERVICE_NAME"
echo "  View logs:       sudo journalctl -u $SERVICE_NAME -f"
echo "  Restart service: sudo systemctl restart $SERVICE_NAME"
echo "  Stop service:    sudo systemctl stop $SERVICE_NAME"
echo ""
echo "🌐 Access the dashboard at: http://localhost:8080"
echo "" 
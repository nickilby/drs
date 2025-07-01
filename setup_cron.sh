#!/bin/bash

# Setup cron jobs for vCenter DRS data refresh
set -e

echo "Setting up cron jobs for vCenter DRS data refresh..."

# Get the absolute paths to the scripts
REFRESH_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/refresh_vcenter_data.py"
COMPLIANCE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_compliance.py"

# Create a temporary file with the cron jobs
TEMP_CRON=$(mktemp)

# Add the cron jobs
echo "# Refresh vCenter data every 15 minutes" > "$TEMP_CRON"
echo "*/15 * * * * $REFRESH_SCRIPT" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"
echo "# Check compliance and update metrics every 5 minutes" >> "$TEMP_CRON"
echo "*/5 * * * * $COMPLIANCE_SCRIPT" >> "$TEMP_CRON"

# Install the cron job for the current user
crontab "$TEMP_CRON"

# Clean up
rm "$TEMP_CRON"

echo "✅ Cron jobs installed successfully!"
echo ""
echo "📋 Current cron jobs:"
crontab -l
echo ""
echo "🔄 Data will be refreshed every 15 minutes"
echo "🔍 Compliance will be checked every 5 minutes"
echo "📝 Logs will be written to:"
echo "   - /var/log/vcenter-drs-refresh.log (data collection)"
echo "   - /var/log/vcenter-drs-compliance.log (compliance checks)"
echo ""
echo "💡 To modify the schedule, edit with: crontab -e"
echo "💡 To remove all cron jobs: crontab -r"
echo ""
echo "⏰ Common cron schedules:"
echo "  Every 5 minutes:  */5 * * * *"
echo "  Every 15 minutes: */15 * * * *"
echo "  Every hour:       0 * * * *"
echo "  Every 2 hours:    0 */2 * * *"
echo "  Daily at 2 AM:    0 2 * * *" 
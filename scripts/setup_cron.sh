#!/bin/bash
# scripts/setup_cron.sh
#
# Setup cron jobs for automated CryptoPulse data collection.
# Run this script once to install the cron jobs.
#

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH=$(which python3)

echo "Setting up CryptoPulse cron jobs..."
echo "Project directory: $PROJECT_DIR"
echo "Python path: $PYTHON_PATH"

# Create cron job entries
CRON_JOBS=$(cat << EOF
# CryptoPulse Data Collection & Processing Jobs

# Run data collection every 6 hours
0 */6 * * * cd $PROJECT_DIR && $PYTHON_PATH scripts/daily_collection.py >> logs/cron.log 2>&1

# Run metrics scoring 30 minutes after data collection
30 */6 * * * cd $PROJECT_DIR && $PYTHON_PATH scripts/daily_scoring.py >> logs/cron.log 2>&1

# Run price collection every hour (lightweight)
0 * * * * cd $PROJECT_DIR && $PYTHON_PATH -c "import sys; sys.path.append('src'); from price_collector import PriceCollector; PriceCollector().collect_latest_price()" >> logs/cron.log 2>&1

# Weekly backup and cleanup (Sundays at 3 AM)
0 3 * * 0 cd $PROJECT_DIR && cp db/cryptopulse.db db/backup_\$(date +\%Y\%m\%d).db >> logs/cron.log 2>&1
EOF
)

echo "Proposed cron jobs:"
echo "$CRON_JOBS"
echo ""

# Ask for confirmation
read -p "Do you want to install these cron jobs? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_JOBS") | crontab -
    echo "✅ Cron jobs installed successfully!"
    echo "📊 Your laptop will now collect and process data automatically:"
    echo "   - Every 6 hours: Full data collection (Reddit, Twitter, News)"
    echo "   - 30 min later: Process all new data with 5-metric scoring"
    echo "   - Every hour: ETH price updates"
    echo "   - Weekly: Database backup"
    echo ""
    echo "📝 Check logs in: $PROJECT_DIR/logs/"
    echo "🔧 View cron jobs: crontab -l"
    echo "❌ Remove cron jobs: crontab -e (delete CryptoPulse lines)"
else
    echo "❌ Cron jobs not installed."
    echo "💡 You can run manual collection with: python scripts/daily_collection.py"
fi

echo ""
echo "🔍 To monitor data collection:"
echo "   tail -f logs/cron.log"
echo "   tail -f logs/daily_collection_*.log"
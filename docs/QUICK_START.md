# CryptoPulse Quick Start Guide

## 🚀 For Continuing This Project

### If You Close Terminal/Restart:

1. **Navigate to project:**
   ```bash
   cd /home/zenitsu/Desktop/CryptoPulse
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Check system status:**
   ```bash
   # Check if automation is running
   crontab -l
   
   # Monitor logs
   tail -f logs/cron.log
   
   # Check data status
   sqlite3 db/cryptopulse.db "SELECT COUNT(*) FROM reddit_posts; SELECT COUNT(*) FROM text_metrics;"
   ```

### Quick Commands:

```bash
# Manual data collection
python scripts/daily_collection.py

# Manual metrics processing  
python scripts/daily_scoring.py

# Check database status
sqlite3 db/cryptopulse.db ".tables"

# View recent automation activity
tail -20 logs/cron.log
```

## 🎯 What To Do Next:

1. **Keep laptop running** (automation is active)
2. **Wait 30 days** for sufficient ML training data
3. **Proceed to Phase 4**: Price labeling & ML models
4. **Monitor logs** occasionally to ensure smooth operation

## 📊 Expected Automation Schedule:

- **Every hour**: ETH price updates
- **Every 6 hours**: Full data collection (Reddit, Twitter, News)
- **30 min after collection**: Process new data with 5-metric scoring
- **Weekly**: Database backup

## ✅ Success Indicators:

- New entries in `logs/cron.log` every hour
- Growing counts in database tables
- Processed metrics increasing daily
- No error messages in logs

## 🆘 If Something Breaks:

1. Read `PROJECT_STATUS.md` for full troubleshooting
2. Check cron is running: `systemctl is-active cron`
3. Test manually: `python scripts/daily_collection.py`
4. Reinstall automation: `./scripts/setup_cron.sh`

---
*Your system is fully automated - just keep it running!*
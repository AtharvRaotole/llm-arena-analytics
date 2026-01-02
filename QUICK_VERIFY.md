# ✅ Quick Verification Guide

## 🌐 Access Links

### Frontend Dashboard
**http://localhost:8501**

**What to check:**
- ✅ Page loads without errors
- ✅ **Leaderboard** tab shows 10-20 models with scores
- ✅ **Performance Trends** tab shows interactive line chart
- ✅ **Cost Intelligence** page shows calculator and value scores
- ✅ **Market Intelligence** page shows predictions and trends
- ✅ All pages have professional dark theme
- ✅ No blank pages or errors

### Backend API
**http://localhost:8000/docs**

**What to check:**
- ✅ Interactive API documentation loads
- ✅ Click **GET /health** → Try it out → Execute
  - Should return: `{"status": "healthy", "database": "connected", ...}`
- ✅ Click **GET /models/leaderboard** → Try it out → Execute
  - Should return JSON array with 10-20 models
- ✅ Click **GET /forecast/rank** → Try it out → Execute
  - Should return predicted rankings

## 📊 Database Check

Run this command to verify data:
```bash
docker-compose exec backend python <<'EOF'
from database.db_manager import DatabaseManager
db = DatabaseManager()

models = db.get_models()
print(f"✅ Models: {len(models)}")

query = "SELECT COUNT(*) as count FROM arena_rankings"
rankings = db.execute_query(query)[0]['count']
print(f"✅ Arena Rankings: {rankings}")

pricing = db.get_latest_pricing()
print(f"✅ Pricing Records: {len(pricing)}")

query = "SELECT COUNT(*) as count FROM market_sentiment"
sentiment = db.execute_query(query)[0]['count']
print(f"✅ Market Sentiment: {sentiment}")
EOF
```

**Expected:**
- Models: 10+
- Arena Rankings: 100+
- Pricing Records: 10+
- Market Sentiment: 1+ (if scraper was run)

## 🧪 Test Scrapers

### Test Arena Scraper
```bash
docker-compose exec backend python <<'EOF'
from scrapers.chatbot_arena_scraper import ChatbotArenaScraper
scraper = ChatbotArenaScraper()
models = scraper.scrape_leaderboard()
print(f"✅ Scraped {len(models)} models")
EOF
```

### Test Pricing Scraper
```bash
docker-compose exec backend python <<'EOF'
from scrapers.pricing_scraper import PricingScraper
scraper = PricingScraper()
pricing = scraper.scrape_openai_pricing()
print(f"✅ Scraped {len(pricing)} OpenAI models")
EOF
```

## ✅ Quick Checklist

- [ ] **Frontend** (http://localhost:8501) - All 4 pages load
- [ ] **Backend API** (http://localhost:8000/docs) - Health check works
- [ ] **Leaderboard** - Shows models with scores
- [ ] **Performance Trends** - Chart displays
- [ ] **Cost Intelligence** - Calculator works
- [ ] **Market Intelligence** - Predictions show
- [ ] **Database** - Has data (models, rankings, pricing)

## 🐛 If Something Doesn't Work

1. **Frontend blank/errors:**
   ```bash
   docker-compose restart frontend
   ```

2. **Backend API not responding:**
   ```bash
   docker-compose restart backend
   ```

3. **No data in dashboard:**
   ```bash
   # Seed historical data
   docker-compose exec backend bash -c "cd /app/scripts && python seed_historical_data.py"
   ```

4. **Check container status:**
   ```bash
   docker-compose ps
   ```

5. **View logs:**
   ```bash
   docker-compose logs backend
   docker-compose logs frontend
   ```


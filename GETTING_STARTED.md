# 🚀 Getting Started - Step-by-Step Guide

This guide will help you get LLM Arena Analytics running while waiting for Reddit API approval.

---

## ✅ Step 1: Prerequisites Check

### 1.1 Check Docker Installation

```bash
docker --version
docker-compose --version
```

**Expected:** Both should show version numbers (Docker 20.10+, Compose 2.0+)

**If not installed:**
- macOS: Download from https://www.docker.com/products/docker-desktop
- Linux: Follow Docker installation guide for your distribution

### 1.2 Verify Docker is Running

```bash
docker ps
```

**Expected:** Should show running containers (or empty list, not an error)

**If error:** Start Docker Desktop (macOS) or Docker service (Linux)

---

## ✅ Step 2: Create Environment File

### 2.1 Create .env File

```bash
cd llm-arena-analytics
cat > .env <<EOF
# Database Configuration
DB_NAME=llm_arena_analytics
DB_USER=postgres
DB_PASSWORD=your_secure_password_here
DB_PORT=5432

# API Configuration (optional)
API_BASE_URL=http://localhost:8000
EOF
```

**⚠️ Important:** Replace `your_secure_password_here` with a strong password!

**Example:**
```env
DB_PASSWORD=MySecurePass123!
```

### 2.2 Create Data Directories

```bash
mkdir -p data/models data/forecasts data/arena_leaderboard
```

---

## ✅ Step 3: Start Docker Services

### 3.1 Build and Start Containers

```bash
docker-compose up --build -d
```

**Expected output:**
```
Creating network "llm-arena-analytics_llm_arena_network" ...
Creating volume "llm-arena-analytics_postgres_data" ...
Creating llm_arena_postgres ...
Creating llm_arena_backend ...
Creating llm_arena_frontend ...
```

**Wait 2-3 minutes** for all services to start.

### 3.2 Check Service Status

```bash
docker-compose ps
```

**Expected:** All services should show "Up" status:
- ✅ `llm_arena_postgres` - Up
- ✅ `llm_arena_backend` - Up  
- ✅ `llm_arena_frontend` - Up

**If any service shows "Restarting" or "Exit":**
```bash
docker-compose logs [service_name]
# Example: docker-compose logs backend
```

---

## ✅ Step 4: Verify Database Setup

### 4.1 Check Database Connection

```bash
docker-compose exec postgres psql -U postgres -d llm_arena_analytics -c "SELECT version();"
```

**Expected:** PostgreSQL version information

### 4.2 Verify Tables Created

```bash
docker-compose exec postgres psql -U postgres -d llm_arena_analytics -c "\dt"
```

**Expected:** Should show tables:
- models
- arena_rankings
- pricing_data
- performance_metrics
- scraping_logs
- market_sentiment

---

## ✅ Step 5: Populate Initial Data

### 5.1 Insert Test Models

```bash
docker-compose exec backend python <<'EOF'
from database.db_manager import DatabaseManager

db = DatabaseManager()

# Insert test models
models = [
    ("GPT-4 Turbo", "OpenAI"),
    ("Claude 3.5 Sonnet", "Anthropic"),
    ("Gemini Pro", "Google"),
    ("Llama 3 70B", "Meta"),
    ("Mistral Large", "Mistral AI")
]

for name, provider in models:
    try:
        model_id = db.insert_model(name, provider)
        print(f"✅ Inserted {name} (ID: {model_id})")
    except Exception as e:
        print(f"⚠️  {name}: {str(e)}")

print("\n✅ Models inserted!")
EOF
```

**Expected output:**
```
✅ Inserted GPT-4 Turbo (ID: 1)
✅ Inserted Claude 3.5 Sonnet (ID: 2)
✅ Inserted Gemini Pro (ID: 3)
✅ Inserted Llama 3 70B (ID: 4)
✅ Inserted Mistral Large (ID: 5)

✅ Models inserted!
```

### 5.2 Insert Test Arena Scores (30 days of data)

```bash
docker-compose exec backend python <<'EOF'
from database.db_manager import DatabaseManager
from datetime import datetime, timedelta
import random

db = DatabaseManager()

# Get all models
models = db.get_models()
print(f"Found {len(models)} models")

# Insert scores for last 30 days
for model in models:
    base_score = 1250 - (models.index(model) * 10)
    inserted = 0
    
    for days_ago in range(30):
        date = datetime.now() - timedelta(days=days_ago)
        score = base_score + random.randint(-20, 20)
        rank = models.index(model) + 1
        
        try:
            db.insert_arena_score(
                model['id'],
                score,
                rank,
                "overall",
                date
            )
            inserted += 1
        except Exception as e:
            pass  # Skip duplicates
    
    print(f"✅ {model['name']}: {inserted} days of scores")

print("\n✅ Historical data created!")
EOF
```

**Expected output:**
```
Found 5 models
✅ GPT-4 Turbo: 30 days of scores
✅ Claude 3.5 Sonnet: 30 days of scores
✅ Gemini Pro: 30 days of scores
✅ Llama 3 70B: 30 days of scores
✅ Mistral Large: 30 days of scores

✅ Historical data created!
```

### 5.3 Insert Test Pricing Data

```bash
docker-compose exec backend python <<'EOF'
from database.db_manager import DatabaseManager
from datetime import datetime

db = DatabaseManager()

# Pricing data (per 1K tokens)
pricing = [
    ("GPT-4 Turbo", 0.01, 0.03),
    ("Claude 3.5 Sonnet", 0.003, 0.015),
    ("Gemini Pro", 0.00025, 0.001),
    ("Llama 3 70B", 0.0, 0.0),  # Open source
    ("Mistral Large", 0.002, 0.006)
]

models = db.get_models()
for model in models:
    for name, input_price, output_price in pricing:
        if model['name'] == name:
            try:
                db.insert_pricing(
                    model['id'],
                    "OpenAI" if "GPT" in name else 
                    "Anthropic" if "Claude" in name else
                    "Google" if "Gemini" in name else
                    "Meta" if "Llama" in name else "Mistral AI",
                    name,
                    input_price,
                    output_price,
                    datetime.now().date()
                )
                print(f"✅ {name}: ${input_price}/${output_price} per 1K tokens")
            except Exception as e:
                print(f"⚠️  {name}: {str(e)}")

print("\n✅ Pricing data inserted!")
EOF
```

---

## ✅ Step 6: Test Backend API

### 6.1 Health Check

```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-02T...",
  "database": "connected",
  "ml_models": {...}
}
```

### 6.2 Test Leaderboard Endpoint

```bash
curl http://localhost:8000/models/leaderboard
```

**Expected:** JSON array with model data

### 6.3 Test API Documentation

Open in browser: **http://localhost:8000/docs**

**Expected:** Swagger UI with all API endpoints

---

## ✅ Step 7: Test Frontend Dashboard

### 7.1 Access Dashboard

Open in browser: **http://localhost:8501**

**Expected:**
- ✅ "🏆 LLM Arena Analytics" title
- ✅ Sidebar with stats
- ✅ Leaderboard tab showing models
- ✅ Performance Trends tab

### 7.2 Test Leaderboard Page

- Should show 5 models with scores
- Models should be sortable
- Provider colors should display

### 7.3 Test Performance Trends

1. Select 2-3 models from dropdown
2. Set date range (last 30 days)
3. Chart should render with lines
4. Statistics table should show below

---

## ✅ Step 8: Test Scrapers (Optional - Real Data)

### 8.1 Scrape Chatbot Arena Leaderboard

```bash
docker-compose exec backend bash
cd /app/scrapers
python scraper_pipeline.py
exit
```

**Expected:**
```
[INFO] Starting scraper pipeline...
[INFO] Scraping Chatbot Arena leaderboard...
[INFO] Found 50+ models
[INFO] Inserted X new models
[INFO] Updated Y models with today's scores
[SUCCESS] Pipeline completed successfully
```

**Note:** This may take 30-60 seconds

### 8.2 Scrape Pricing Data

```bash
docker-compose exec backend bash
cd /app/scrapers
python pricing_scraper.py
exit
```

**Expected:** Pricing data for multiple providers

---

## ✅ Step 9: Generate More Historical Data (Optional)

### 9.1 Generate 90 Days of Data

```bash
docker-compose exec backend bash
cd /app/scripts

# Calculate dates (macOS)
START_DATE=$(date -v-90d +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)

python seed_historical_data.py --start-date $START_DATE --end-date $END_DATE
exit
```

**For Linux:**
```bash
START_DATE=$(date -d '90 days ago' +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)
```

**Expected:** 90 days of realistic historical data for all models

---

## ✅ Step 10: Train ML Models (Optional)

### 10.1 Train Performance Predictor

```bash
docker-compose exec backend bash
cd /app/models
python performance_predictor.py
exit
```

**Expected:**
```
[INFO] Loading data from database...
[INFO] Engineering features...
[INFO] Training model...
[INFO] Model saved to data/models/performance_predictor.pkl
[SUCCESS] Training completed!
```

**Note:** This may take 2-5 minutes

### 10.2 Train Trend Forecaster

```bash
docker-compose exec backend bash
cd /app/models
python trend_forecaster.py
exit
```

**Expected:** Forecast models and visualizations saved

---

## ✅ Step 11: Test Cost Intelligence (After Pricing Data)

### 11.1 Access Cost Intelligence Page

1. Open dashboard: http://localhost:8501
2. Navigate to "Cost Intelligence" page
3. Should show:
   - ✅ Interactive calculator
   - ✅ Value score leaderboard
   - ✅ Price history chart

### 11.2 Test Calculator

1. Set monthly tokens (e.g., 10M input, 5M output)
2. Set minimum score (e.g., 1200)
3. Click "Calculate Best Value"
4. Should recommend a model with cost estimate

---

## ✅ Step 12: Test Market Intelligence (After ML Models)

### 12.1 Access Market Intelligence Page

1. Navigate to "Market Intelligence" page
2. Should show:
   - ✅ Future rankings prediction
   - ✅ Trend analysis charts
   - ✅ Market events timeline
   - ✅ Insights & recommendations

**Note:** Requires trained ML models from Step 10

---

## ⏳ Step 13: Reddit Integration (After Approval)

### 13.1 Add Reddit Credentials

Once Reddit approves your API request:

```bash
# Option 1: Use helper script
./add_reddit_credentials.sh

# Option 2: Manually edit .env
nano .env
# Add:
# REDDIT_CLIENT_ID=your_client_id
# REDDIT_CLIENT_SECRET=your_secret
# REDDIT_USER_AGENT=llm-arena-analytics/1.0
```

### 13.2 Restart Services

```bash
docker-compose restart backend
```

### 13.3 Test Sentiment Scraper

```bash
docker-compose exec backend bash
cd /app/scrapers
python sentiment_scraper.py --days 7
exit
```

### 13.4 Analyze Sentiment

```bash
docker-compose exec backend bash
cd /app/models
python sentiment_analyzer.py --process-all
exit
```

---

## 🎯 Quick Verification Checklist

Run these commands to verify everything works:

```bash
# 1. Services running
docker-compose ps

# 2. Database has data
docker-compose exec postgres psql -U postgres -d llm_arena_analytics -c "SELECT COUNT(*) FROM models; SELECT COUNT(*) FROM arena_rankings;"

# 3. Backend responds
curl http://localhost:8000/health

# 4. Frontend responds
curl http://localhost:8501/_stcore/health

# 5. View logs if issues
docker-compose logs --tail=50
```

---

## 🐛 Troubleshooting

### Issue: Containers won't start

```bash
# Check logs
docker-compose logs

# Check specific service
docker-compose logs backend
docker-compose logs postgres
```

### Issue: "No data" in dashboard

```bash
# Check if data exists
docker-compose exec postgres psql -U postgres -d llm_arena_analytics -c "SELECT COUNT(*) FROM models;"

# If 0, run Step 5 again
```

### Issue: Port already in use

```bash
# Check what's using ports
lsof -i :8000  # Backend
lsof -i :8501  # Frontend
lsof -i :5432  # PostgreSQL

# Stop conflicting services or change ports in docker-compose.yml
```

### Issue: Database connection errors

```bash
# Verify database is running
docker-compose ps postgres

# Test connection
docker-compose exec postgres psql -U postgres -c "SELECT version();"

# Check environment variables
docker-compose exec backend env | grep DB_
```

---

## ✅ Success Criteria

Your application is fully working when:

- ✅ Dashboard loads at http://localhost:8501
- ✅ Leaderboard shows models with scores
- ✅ Performance trends chart renders
- ✅ Backend API responds at http://localhost:8000
- ✅ API docs accessible at http://localhost:8000/docs
- ✅ No errors in browser console (F12)
- ✅ Database has models and scores

---

## 📝 What Works Without Reddit

✅ **All of these work without Reddit:**
- Leaderboard dashboard
- Performance trends visualization
- Cost intelligence calculator
- Market intelligence (forecasting)
- ML model predictions
- Arena scraper
- Pricing scraper
- Historical data generation

❌ **Requires Reddit:**
- Sentiment scraper (Step 13)
- Sentiment analysis dashboard features

---

**You're all set! 🎉 Follow these steps and your application will be running.**


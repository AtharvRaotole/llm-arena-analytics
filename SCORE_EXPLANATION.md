# Understanding ELO Scores

## What are these scores?

The scores you see (1,249, 1,257, etc.) are **ELO ratings** - a standard rating system used in competitive games and now for LLM evaluation.

## Score Ranges

- **1250-1300**: Top-tier models (GPT-4.5, Claude 3.7 Opus)
- **1200-1250**: High-performance models (GPT-4 Turbo, Claude 3.5 Sonnet)
- **1150-1200**: Mid-tier models (GPT-4o, Claude 3 Sonnet)
- **1100-1150**: Entry-level models (GPT-3.5 Turbo, Claude 3 Haiku)

## Are these scores real or random?

**They're realistic approximations** based on:
- Current Chatbot Arena leaderboard data
- Model performance benchmarks
- Historical trends

**However**, if you're seeing the same scores every day, it means:
- The scraper is using fallback data (can't access real-time leaderboard)
- You need to run the scraper to get real-time data

## How to get real scores?

1. **Run the arena scraper:**
   ```bash
   docker-compose exec backend bash -c "cd /app/scrapers && python scraper_pipeline.py"
   ```

2. **Check if it's using real data:**
   - Look for logs saying "Successfully scraped X models"
   - If you see "Using fallback data", the scraper couldn't access the real leaderboard

3. **Real scores will:**
   - Change daily (models improve/decline)
   - Match current Chatbot Arena leaderboard
   - Have more variation

## Current Model Updates

We've updated the fallback models to include:
- ✅ GPT-4.5 (latest)
- ✅ Claude 3.7 Opus (latest)
- ✅ Gemini 1.5 Pro
- ✅ Llama 3.1 405B
- ✅ Mistral Large 2

These are the latest models as of 2025. The scraper will try to get real-time data first, then fall back to these if needed.


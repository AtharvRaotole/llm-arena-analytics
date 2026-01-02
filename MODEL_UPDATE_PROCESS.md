# Model Update Process

## Why We Use Fallback Models

The Chatbot Arena leaderboard is hosted on HuggingFace Spaces, which uses JavaScript to render content dynamically. This makes it difficult to scrape with traditional tools like BeautifulSoup.

## Current Approach

1. **First**: Try to scrape real-time data from multiple API endpoints
2. **Second**: Try Selenium for JavaScript rendering
3. **Third**: Use latest known models from web search

## Latest Models (Updated Jan 2026)

Based on web search, the latest models include:

- **GPT-5.2** (Dec 2025) - OpenAI's latest
- **Gemini 3 Pro/Flash** (Nov-Dec 2025) - Google's latest
- **Claude 3.7 Opus** - Anthropic's latest
- **Qwen3-Max** (Sep 2025) - Alibaba
- **DeepSeek V3.1** (Aug 2025)
- **Llama 4 Scout/Maverick** (Apr 2025) - Meta
- **Grok 4** (Jul 2025) - xAI
- **Mistral Medium 3** (May 2025)
- **Apertus 70B/8B** (Sep 2025) - Swiss National AI

## How to Get Real-Time Data

### Option 1: Run the Scraper
```bash
docker-compose exec backend bash -c "cd /app/scrapers && python scraper_pipeline.py"
```

### Option 2: Manual Update
If you find a working API endpoint, add it to `chatbot_arena_scraper.py` in the `alternative_urls` list.

### Option 3: Update Fallback Models
Edit `backend/scrapers/chatbot_arena_scraper.py` and update the `known_models` list in `_get_latest_known_models()` method.

## Future Improvements

1. **Automated Web Scraping**: Periodically search for latest model releases
2. **API Integration**: If LMSYS releases an official API
3. **Community Updates**: Allow users to submit model updates
4. **Real-time Monitoring**: Watch for model release announcements

## Note

The fallback models are updated based on web search results, but they may not reflect the exact current Chatbot Arena leaderboard. For the most accurate data, the scraper needs to successfully access the real leaderboard.


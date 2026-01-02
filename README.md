# 🏆 LLM Arena Analytics

> **Comprehensive analytics platform for tracking, comparing, and optimizing Large Language Model (LLM) performance, pricing, and market trends.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

LLM Arena Analytics is a full-stack analytics platform that scrapes, stores, and visualizes data from the LMSYS Chatbot Arena leaderboard. It provides real-time insights into LLM performance, pricing trends, cost optimization, and market intelligence.

### Key Capabilities

- **Real-time Leaderboard Tracking**: Automated scraping of Chatbot Arena rankings
- **Performance Analytics**: Historical trend analysis and forecasting
- **Cost Intelligence**: Price comparison and value scoring
- **Market Intelligence**: Trend predictions and anomaly detection
- **ML-Powered Recommendations**: Best model selection based on task requirements

---

## ✨ Features

### 📊 Dashboard Features

- **🏆 Leaderboard**: Current rankings with ELO scores, win rates, and provider information
- **📈 Performance Trends**: Time-series analysis with interactive charts
- **💰 Cost Intelligence**: Price comparison, value scoring, and cost optimization
- **🔮 Market Intelligence**: Future rankings prediction and trend analysis
- **📊 Arena Rankings**: Detailed model comparisons and benchmarks

### 🤖 Backend Features

- **Web Scraping**: Robust scrapers for Chatbot Arena and pricing data
- **ML Models**: Performance prediction and trend forecasting
- **REST API**: FastAPI endpoints for programmatic access
- **Database**: PostgreSQL with optimized schema and indexing
- **Automated Pipelines**: Daily data collection and processing

### 🎨 UI/UX

- **Modern Dark Theme**: Professional, easy-on-the-eyes interface
- **Interactive Visualizations**: Plotly charts with zoom, pan, and hover
- **Responsive Design**: Works on desktop and tablet
- **Real-time Updates**: Live data refresh capabilities

---

## 🛠 Tech Stack

### Backend
- **Python 3.11**: Core language
- **FastAPI**: REST API framework
- **PostgreSQL**: Relational database
- **SQLAlchemy/psycopg2**: Database ORM and driver
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn/XGBoost**: Machine learning
- **Prophet/Statsmodels**: Time series forecasting
- **BeautifulSoup/Selenium**: Web scraping

### Frontend
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Interactive visualizations
- **Custom CSS**: Professional styling

### Infrastructure
- **Docker & Docker Compose**: Containerization
- **Nginx**: Reverse proxy (optional)
- **GitHub Actions**: CI/CD (optional)

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- 4GB+ RAM available

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AtharvRaotole/llm-arena-analytics.git
   cd llm-arena-analytics
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Initialize database**
   ```bash
   docker-compose exec postgres psql -U postgres -d llm_arena_analytics -f /docker-entrypoint-initdb.d/schema.sql
   ```

5. **Populate initial data**
   ```bash
   # Run scraper to get latest models
   docker-compose exec backend bash -c "cd /app/scrapers && python scraper_pipeline.py"
   
   # Generate historical data (optional)
   docker-compose exec backend bash -c "cd /app/scripts && python seed_historical_data.py"
   ```

6. **Access the dashboard**
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

---

## 📁 Project Structure

```
llm-arena-analytics/
├── backend/
│   ├── api/              # FastAPI application
│   ├── database/         # Database schema and manager
│   ├── models/           # ML models (predictors, forecasters)
│   ├── scrapers/         # Web scrapers
│   ├── scripts/          # Utility scripts
│   └── tests/            # Unit tests
├── frontend/
│   ├── app.py            # Main Streamlit app
│   ├── pages/            # Dashboard pages
│   └── .streamlit/       # Streamlit config
├── data/                 # Scraped data storage
├── notebooks/            # Jupyter notebooks for analysis
├── deploy/               # Deployment scripts
├── docker-compose.yml    # Service orchestration
└── README.md             # This file
```

---

## 🔧 Key Components

### Database Schema

- **models**: LLM model metadata
- **arena_rankings**: Performance scores and rankings
- **pricing_data**: Cost information per token
- **market_sentiment**: Social media discussions
- **scraping_logs**: Scraper execution history

### Scrapers

- **Chatbot Arena Scraper**: Fetches leaderboard data with multiple fallback strategies
- **Pricing Scraper**: Collects pricing from OpenAI, Anthropic, Google AI
- **Sentiment Scraper**: Gathers social media discussions (Reddit, Hacker News)

### ML Models

- **Performance Predictor**: Recommends best model for given tasks
- **Trend Forecaster**: Predicts future rankings and detects anomalies
- **Cost Optimizer**: Calculates value scores and cost comparisons

### API Endpoints

- `GET /health`: System health check
- `GET /models/leaderboard`: Current top models
- `GET /models/{model_name}/history`: Historical performance
- `POST /predict/best-model`: ML-based recommendations
- `GET /forecast/rank`: Future rankings prediction
- `GET /cost/compare`: Cost comparison

---

## 📚 API Documentation

Full API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example API Call

```python
import requests

# Get current leaderboard
response = requests.get("http://localhost:8000/models/leaderboard")
top_models = response.json()

# Get model recommendations
response = requests.post("http://localhost:8000/predict/best-model", json={
    "task_type": "coding",
    "prompt_length": 1000,
    "complexity": 7,
    "max_cost_per_1m": 10.0
})
recommendation = response.json()
```

---

## 🚢 Deployment

### Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Cloud Deployment

See `deploy/` directory for:
- AWS EC2 + RDS setup scripts
- GCP Cloud Run + Cloud SQL scripts
- Migration and backup scripts

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Atharv Raotole**

- GitHub: [@AtharvRaotole](https://github.com/AtharvRaotole)
- Project: [llm-arena-analytics](https://github.com/AtharvRaotole/llm-arena-analytics)

---

## 🙏 Acknowledgments

- [LMSYS Chatbot Arena](https://chat.lmsys.org) for the leaderboard data
- [Hugging Face](https://huggingface.co) for hosting the leaderboard
- All the LLM providers for pushing the boundaries of AI

---

## 📊 Current Models Tracked

The system tracks **30+ models** including:

- **Latest Models (2025-2026)**: GPT-5.2, Gemini 3 Pro/Flash, Claude Opus 4.5, Claude Sonnet 4.5
- **Major Providers**: OpenAI, Anthropic, Google, Meta, Mistral AI, xAI, Alibaba, DeepSeek

See the dashboard for the complete list and real-time rankings!

---

**⭐ Star this repo if you find it useful!**

#!/bin/bash
# Quick Start Script for LLM Arena Analytics
# This script sets up everything needed to run the application

set -e

echo "🚀 LLM Arena Analytics - Quick Start"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose found${NC}"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env <<EOF
DB_NAME=llm_arena_analytics
DB_USER=postgres
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
DB_PORT=5432
EOF
    echo -e "${GREEN}✅ Created .env file with random password${NC}"
    echo -e "${YELLOW}⚠️  Database password: $(grep DB_PASSWORD .env | cut -d'=' -f2)${NC}"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Create data directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data/models data/forecasts data/arena_leaderboard
echo -e "${GREEN}✅ Data directories created${NC}"

# Build and start containers
echo -e "${YELLOW}Building and starting Docker containers...${NC}"
echo "This may take 3-5 minutes on first run..."
docker-compose up -d --build

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ Services are running${NC}"
else
    echo -e "${RED}❌ Some services failed to start. Check logs with: docker-compose logs${NC}"
    exit 1
fi

# Wait for database to be ready
echo -e "${YELLOW}Waiting for database to be ready...${NC}"
for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U postgres &>/dev/null; then
        echo -e "${GREEN}✅ Database is ready${NC}"
        break
    fi
    sleep 2
done

# Insert test data
echo -e "${YELLOW}Inserting test data...${NC}"
docker-compose exec -T backend python <<'PYTHON_SCRIPT'
from database.db_manager import DatabaseManager
from datetime import datetime, timedelta
import random

db = DatabaseManager()

# Insert test models
models = [
    ("GPT-4 Turbo", "OpenAI"),
    ("Claude 3.5 Sonnet", "Anthropic"),
    ("Gemini Pro", "Google"),
    ("Llama 3 70B", "Meta"),
    ("Mistral Large", "Mistral AI")
]

model_ids = []
for name, provider in models:
    try:
        model_id = db.insert_model(name, provider)
        model_ids.append((model_id, name))
        print(f"✅ Inserted {name}")
    except Exception as e:
        # Model might already exist
        models_list = db.get_models()
        for m in models_list:
            if m['name'] == name:
                model_ids.append((m['id'], name))
                print(f"✅ Found existing {name}")
                break

# Insert scores for last 30 days
for model_id, name in model_ids:
    base_score = 1250 - (model_ids.index((model_id, name)) * 10)
    inserted = 0
    for days_ago in range(30):
        date = datetime.now() - timedelta(days=days_ago)
        score = base_score + random.randint(-20, 20)
        rank = model_ids.index((model_id, name)) + 1
        try:
            db.insert_arena_score(model_id, score, rank, "overall", date)
            inserted += 1
        except:
            pass  # Skip duplicates
    print(f"✅ Inserted {inserted} days of scores for {name}")

print("\n🎉 Test data created successfully!")
PYTHON_SCRIPT

echo ""
echo "======================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "======================================"
echo ""
echo "🌐 Access the application:"
echo "   Frontend:  http://localhost:8501"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📊 To add more data:"
echo "   docker-compose exec backend bash"
echo "   cd /app/scrapers"
echo "   python scraper_pipeline.py"
echo ""
echo "🛑 To stop:"
echo "   docker-compose down"
echo ""


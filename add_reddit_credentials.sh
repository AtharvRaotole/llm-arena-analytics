#!/bin/bash
# Helper script to add Reddit credentials to .env file

echo "🔐 Adding Reddit API Credentials to .env"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env <<EOF
# Database Configuration
DB_NAME=llm_arena_analytics
DB_USER=postgres
DB_PASSWORD=postgres
DB_PORT=5432

# Reddit API Credentials
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=llm-arena-analytics/1.0
EOF
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "Please enter your Reddit API credentials:"
echo ""

# Get client ID
read -p "Enter Reddit Client ID: " CLIENT_ID
if [ -n "$CLIENT_ID" ]; then
    # Update or add REDDIT_CLIENT_ID
    if grep -q "REDDIT_CLIENT_ID" .env; then
        sed -i.bak "s/^REDDIT_CLIENT_ID=.*/REDDIT_CLIENT_ID=$CLIENT_ID/" .env
    else
        echo "REDDIT_CLIENT_ID=$CLIENT_ID" >> .env
    fi
    echo "✅ Client ID added"
fi

# Get client secret
read -p "Enter Reddit Client Secret: " CLIENT_SECRET
if [ -n "$CLIENT_SECRET" ]; then
    # Update or add REDDIT_CLIENT_SECRET
    if grep -q "REDDIT_CLIENT_SECRET" .env; then
        sed -i.bak "s/^REDDIT_CLIENT_SECRET=.*/REDDIT_CLIENT_SECRET=$CLIENT_SECRET/" .env
    else
        echo "REDDIT_CLIENT_SECRET=$CLIENT_SECRET" >> .env
    fi
    echo "✅ Client Secret added"
fi

# Ensure USER_AGENT is set
if ! grep -q "REDDIT_USER_AGENT" .env; then
    echo "REDDIT_USER_AGENT=llm-arena-analytics/1.0" >> .env
fi

echo ""
echo "✅ Reddit credentials added to .env file!"
echo ""
echo "To test, run:"
echo "  docker-compose exec backend bash"
echo "  cd /app/scrapers"
echo "  python sentiment_scraper.py --days 1"


-- Database schema for LLM Arena Analytics

-- Models table: Stores information about LLM models
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    provider VARCHAR(100),
    model_family VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table: Stores performance data for models
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    benchmark_type VARCHAR(100),
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, metric_name, benchmark_type, recorded_at)
);

-- Pricing data table: Stores pricing information
CREATE TABLE IF NOT EXISTS pricing_data (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
    provider VARCHAR(100) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    input_cost_per_token FLOAT,
    output_cost_per_token FLOAT,
    effective_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, effective_date)
);

-- Arena rankings table: Stores Chatbot Arena rankings
CREATE TABLE IF NOT EXISTS arena_rankings (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
    rank_position INTEGER NOT NULL,
    elo_rating FLOAT,
    win_rate FLOAT,
    total_battles INTEGER,
    category VARCHAR(100),
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Scraping logs table: Tracks scraping activities
CREATE TABLE IF NOT EXISTS scraping_logs (
    id SERIAL PRIMARY KEY,
    scraper_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    records_scraped INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- Market sentiment table: Stores social media discussions about LLMs
CREATE TABLE IF NOT EXISTS market_sentiment (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255),
    source VARCHAR(50) NOT NULL,  -- reddit, hackernews, twitter
    post_id VARCHAR(255) NOT NULL,  -- Unique ID from source
    title TEXT,
    content TEXT NOT NULL,
    author VARCHAR(255),
    score INTEGER DEFAULT 0,  -- Upvotes/likes
    url TEXT,
    posted_at TIMESTAMPTZ NOT NULL,
    scraped_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    sentiment VARCHAR(20),  -- positive, negative, neutral
    sentiment_score FLOAT,  -- -1.0 to 1.0
    confidence FLOAT,  -- 0.0 to 1.0
    topics TEXT[],  -- Array of extracted topics
    processed_at TIMESTAMPTZ,
    UNIQUE(source, post_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_models_provider ON models(provider);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_model_id ON performance_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_pricing_data_model_id ON pricing_data(model_id);
CREATE INDEX IF NOT EXISTS idx_pricing_data_provider ON pricing_data(provider);
CREATE INDEX IF NOT EXISTS idx_pricing_data_effective_date ON pricing_data(effective_date);
CREATE INDEX IF NOT EXISTS idx_arena_rankings_model_id ON arena_rankings(model_id);
CREATE INDEX IF NOT EXISTS idx_arena_rankings_recorded_at ON arena_rankings(recorded_at);
CREATE INDEX IF NOT EXISTS idx_arena_rankings_category ON arena_rankings(category);
CREATE INDEX IF NOT EXISTS idx_scraping_logs_started_at ON scraping_logs(started_at);
CREATE INDEX IF NOT EXISTS idx_scraping_logs_completed_at ON scraping_logs(completed_at);
CREATE INDEX IF NOT EXISTS idx_market_sentiment_model_name ON market_sentiment(model_name);
CREATE INDEX IF NOT EXISTS idx_market_sentiment_source ON market_sentiment(source);
CREATE INDEX IF NOT EXISTS idx_market_sentiment_posted_at ON market_sentiment(posted_at);
CREATE INDEX IF NOT EXISTS idx_market_sentiment_score ON market_sentiment(score);


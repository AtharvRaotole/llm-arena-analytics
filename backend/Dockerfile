FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including cmdstan build tools)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    build-essential \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install cmdstan for Prophet (optional - enables Prophet forecasting)
# Note: This takes several minutes but enables Prophet to work
RUN pip install cmdstanpy && \
    python -c "import cmdstanpy; cmdstanpy.install_cmdstan(verbose=False)" || echo "cmdstan installation skipped (Prophet will use fallback)"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/models /app/data/forecasts

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

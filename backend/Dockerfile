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
    wget \
    gnupg \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome for Selenium (using modern method without apt-key)
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
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

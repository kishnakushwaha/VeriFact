# Production Dockerfile for Fake News Detector API
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set NLTK data path
ENV NLTK_DATA=/usr/local/share/nltk_data

# Create directory
RUN mkdir -p /usr/local/share/nltk_data

# Download NLTK data to specific directory
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check - increased start period for lazy model loading on first request
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run with gunicorn: SINGLE WORKER to prevent fork memory duplication
# --preload loads app before fork (moot with 1 worker, but explicit)
# --threads 4 provides concurrency without memory duplication
# --timeout 300 for slow first-request model loading
# --max-requests 100 forces worker restart to clear memory leaks
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "--max-requests", "100", "--max-requests-jitter", "10", "app_flask:app"]

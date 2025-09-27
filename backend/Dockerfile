FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create non-root user (optional security)
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port (Railway will override this)
EXPOSE 8000

# Start application
CMD ["python", "bend.py"]

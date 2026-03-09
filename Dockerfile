FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    iverilog \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY chipmind/ chipmind/
COPY frontend/ frontend/
COPY Makefile .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["uvicorn", "chipmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

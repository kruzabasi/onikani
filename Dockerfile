# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system deps for common ML libs (light)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install python deps (use --no-cache-dir)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

# default: open a shell (override in docker-compose or with docker run)
CMD ["bash"]

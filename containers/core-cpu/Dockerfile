# Use an official Ubuntu image as a base.
FROM ubuntu:24.04

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    wget \
    fuse \
    libfuse2 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment, then install requirements
RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY scripts ./scripts

# Copy entrypoint script
COPY containers/core-cpu/core-cpu_entrypoint.sh .
RUN chmod +x core-cpu_entrypoint.sh

# Create a non-root user for security
RUN useradd -ms /bin/bash -u 1001 appuser
RUN chown -R appuser:appuser /app
USER appuser

ENV RHYTHMFORMHOME=/app
ENV PYTHONPATH=/app:$PYTHONPATH

WORKDIR /app/scripts

ENTRYPOINT ["/app/core-cpu_entrypoint.sh"]
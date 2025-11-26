# Use an official NVIDIA CUDA image as a base.
FROM ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including wget to download MuseScore and fuse which is required by AppImages.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    fuse \
    libfuse2 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Download and extract MuseScore AppImage
ARG MUSESCORE_VERSION=4.6.3
ARG GIT_HASH=252940956
RUN wget https://github.com/musescore/MuseScore/releases/download/v${MUSESCORE_VERSION}/MuseScore-Studio-${MUSESCORE_VERSION}.${GIT_HASH}-x86_64.AppImage -O /app/musescore.AppImage \
    && chmod +x /app/musescore.AppImage \
    && /app/musescore.AppImage --appimage-extract \
    && mv squashfs-root /app/musescore-extracted \
    && ln -s /app/musescore-extracted/usr/bin/mscore /app/mscore \
    && ln -s /app/musescore-extracted/usr/bin/mscore /app/musescore \
    && rm /app/musescore.AppImage

# Create a non-root user for security
RUN useradd -ms /bin/bash -u 1001 appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set environment variables for RHYTHMFORM
ENV RHYTHMFORMHOME=/app
ENV PYTHONPATH=/app:$PYTHONPATH

# Point MUSESCORE_PATH to the extracted directory
ENV MUSESCORE_PATH=/app/musescore-extracted
ENV PATH="/app:${PATH}"

RUN ln -s /usr/bin/pydoc3 /app/pydoc \
    && ln -s /usr/bin/python3 /app/python \
    && ln -s /usr/bin/python3-config /app/python-config

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY scripts ./scripts
COPY containers/core-cpu/core-cpu_entrypoint.sh .

# Add a healthcheck to monitor the training process
HEALTHCHECK --interval=60s --timeout=30s --start-period=5m \
  CMD pgrep -f "core-cpu_entrypoint.sh" || exit 1

# Specify the command to run on container startup.
ENTRYPOINT ["core-cpu_entrypoint.sh"]
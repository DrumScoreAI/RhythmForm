# Use an official NVIDIA CUDA image as a base.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including wget to download MuseScore and fuse which is required by AppImages.
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    wget
RUN apt-get install -y \
    libpipewire-0.3 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxcb-glx0 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-sync1 \
    libxcb-xfixes0
RUN apt-get install -y \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xkb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libgl1 \
    libegl1 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    libasound2 \
    libnss3 \
    libssl3 \
    libpulse0 \
    libsndfile1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libxrandr2 \
    libxinerama1 \
    libxss1 \
    libxv1 \
    libxxf86vm1 \
    libcups2 \
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    && rm -rf /var/lib/apt/lists/*

# Download and extract MuseScore AppImage
ARG MUSESCORE_VERSION=4.6.3
ARG GIT_HASH=252940956
RUN wget https://github.com/musescore/MuseScore/releases/download/v${MUSESCORE_VERSION}/MuseScore-Studio-${MUSESCORE_VERSION}.${GIT_HASH}-x86_64.AppImage -O /app/musescore.AppImage \
    && chmod +x /app/musescore.AppImage \
    && /app/musescore.AppImage --appimage-extract \
    && mv squashfs-root /app/musescore-extracted \
    && find /app/musescore-extracted -type d -exec chmod 777 {} \; \
    && find /app/musescore-extracted -type f -exec chmod 777 {} \; \
    && ln -s /app/musescore-extracted/usr/bin/mscore4portable /app/mscore \
    && ln -s /app/musescore-extracted/usr/bin/mscore4portable /app/musescore \
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
ENV PATH="/app:/home/appuser/.local/bin:${PATH}"

RUN ln -s /usr/bin/pydoc3 /app/pydoc \
    && ln -s /usr/bin/python3 /app/python \
    && ln -s /usr/bin/python3-config /app/python-config

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY scripts ./scripts
COPY containers/core/core_entrypoint.sh .

# Add a healthcheck to monitor the training process
HEALTHCHECK --interval=60s --timeout=30s --start-period=5m \
  CMD pgrep -f "core_entrypoint.sh" || exit 1

# Specify the command to run on container startup.
ENTRYPOINT ["core_entrypoint.sh"]
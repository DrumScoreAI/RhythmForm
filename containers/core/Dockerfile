# Use an official NVIDIA CUDA image as a base.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# Set the working directory in the container
WORKDIR /app

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=en_GB.UTF-8
ENV LANGUAGE=en_GB:en
ENV LC_ALL=en_GB.UTF-8
ENV QT_LOGGING_RULES="qt.qml.typeregistration=false"
ENV RHYTHMFORMHOME=/app
ENV PYTHONPATH=/app:$PYTHONPATH
ENV MUSESCORE_PATH=/app/musescore-extracted
ENV PATH="/app:/home/appuser/.local/bin:${PATH}"
ENV PATH="/app:/home/appuser/.local/bin:/app/venv/bin:${PATH}"

ARG MUSESCORE_VERSION=4.6.3
ARG GIT_HASH=252940956

# Install system dependencies, including wget to download MuseScore and fuse which is required by AppImages.
RUN apt-get update && apt-get install -y \
    locales \
    locales-all \
    # en_GB.UTF-8 \
    python3.12 \
    python3.12-venv \
    python3-pip \
    poppler-utils \
    git \
    wget \
    libpipewire-0.3.0t64 \
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
    libopengl0 \
    libxcb-xfixes0 \
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
    libasound2t64 \
    libnss3 \
    libssl3t64 \
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
    libcups2t64 \
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    libglib2.0-0t64 \
    libxcb-cursor0 \
    libxcb-util1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment, then install requirements
RUN python3.12 -m venv /app/venv

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir awscli

# Download and extract MuseScore AppImage
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

# Copy entrypoint script
COPY containers/core/core_entrypoint.sh .

# Add a healthcheck to monitor the training process
HEALTHCHECK --interval=60s --timeout=30s --start-period=5m \
  CMD pgrep -f "core_entrypoint.sh" || exit 1

# Specify the command to run on container startup.
ENTRYPOINT ["core_entrypoint.sh"]
# Use an official Ubuntu image as a base.
FROM ubuntu:24.04

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=en_GB.UTF-8
ENV LANGUAGE=en_GB:en
ENV LC_ALL=en_GB.UTF-8
ENV QT_LOGGING_RULES="qt.qml.typeregistration=false"
ENV RHYTHMFORMHOME=/app
ENV PYTHONPATH=/app:$PYTHONPATH
# Point MUSESCORE_PATH to the extracted directory
ENV MUSESCORE_PATH=/app/mscore

RUN mkdir /app && mkdir /app/venv

# Install system dependencies
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
    xcb-cursor0 \
    libxcb-cursor0 \
    libxcb-util1 \
    xvfb \
RUN apt-get install -y \
    libxcb-cursor \
    libxcb-icccm \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment, then install requirements
RUN python3.12 -m venv /app/venv
ENV PATH="/app:/home/appuser/.local/bin:/app/venv/bin:${PATH}"
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir awscli

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

# Copy the rest of your application's code
COPY . /app
WORKDIR /app

# Copy entrypoint script
COPY containers/core-cpu/core-cpu_entrypoint.sh .
RUN chmod +x core-cpu_entrypoint.sh

# Create a non-root user for security
RUN useradd -ms /bin/bash -u 1001 appuser
RUN chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["core-cpu_entrypoint.sh"]
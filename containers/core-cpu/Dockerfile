# Use an official Ubuntu image as a base.
FROM ubuntu:24.04

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=en_GB.UTF-8
ENV LANGUAGE=en_GB:en
ENV LC_ALL=en_GB.UTF-8
ENV QT_LOGGING_RULES="qt.qml.typeregistration=false"
ENV QT_QPA_PLATFORM="offscreen"
ENV RHYTHMFORMHOME=/app
ENV PYTHONPATH=/app:$PYTHONPATH
# Point MUSESCORE_PATH to the extracted directory
ENV MUSESCORE_PATH=/app/mscore

RUN mkdir /app && mkdir /app/venv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
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
ENV PATH="/app:/home/appuser/.local/bin:/app/venv/bin:${PATH}"

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

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install Python dependencies
RUN pip install --upgrade pip
# Install CPU-only version of PyTorch to save significant space (avoids CUDA binaries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Install remaining requirements (torch will be skipped as it's already satisfied)
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir awscli

# Install prmon for resource monitoring
RUN git clone --recurse-submodules https://github.com/HSF/prmon.git \
    && cd prmon \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/app/prmon -S .. -B . \
    && make -j$(nproc) \
    && make install

# Copy the rest of your application's code
COPY . /app

# Copy entrypoint script
COPY containers/core-cpu/core-cpu_entrypoint.sh .
RUN chmod +x core-cpu_entrypoint.sh

# Create a non-root user for security
RUN useradd -ms /bin/bash -u 1001 appuser
RUN mkdir -p /home/appuser/.aws
COPY s3/creds_template /home/appuser/.aws/credentials
RUN chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["core-cpu_entrypoint.sh"]
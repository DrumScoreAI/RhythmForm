# Use the official PyTorch runtime image which is lighter and verified to work with MIG.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=en_GB.UTF-8
ENV LANGUAGE=en_GB:en
ENV LC_ALL=en_GB.UTF-8
ENV QT_LOGGING_RULES="qt.qml.typeregistration=false"
ENV QT_QPA_PLATFORM="offscreen"
ENV RHYTHMFORMHOME=/app
ENV PYTHONPATH=/app:$PYTHONPATH
ENV MUSESCORE_PATH=/app/musescore-extracted
ENV PATH="/app:/home/appuser/.local/bin:${PATH}"

# --- PyTorch/CUDA Optimizations for MIG ---
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV NCCL_P2P_DISABLE=1
ENV NCCL_IB_DISABLE=1

# Install system dependencies.
# The base image is Ubuntu-based, so apt-get works.
RUN apt-get update && apt-get install -y \
    locales \
    locales-all \
    git \
    wget \
    poppler-utils \
    # MuseScore dependencies
    libpipewire-0.3-0 \
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
    libglib2.0-0 \
    libxcb-cursor0 \
    libxcb-util1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install python dependencies.
# We use the system python provided by the base image (which has the optimized torch installed).
# We filter out torch/torchvision from requirements.txt to avoid overwriting the optimized version.
RUN grep -vE "torch|torchvision" requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt && \
    pip install --no-cache-dir awscli

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

COPY . /app

# Create a non-root user for security
RUN useradd -ms /bin/bash -u 1001 appuser
RUN chown -R appuser:appuser /app
USER appuser

WORKDIR /app/scripts

# Copy entrypoint script
COPY containers/core/core_entrypoint.sh .

# Add a healthcheck to monitor the training process
HEALTHCHECK --interval=60s --timeout=30s --start-period=5m \
  CMD pgrep -f "core_entrypoint.sh" || exit 1

# Specify the command to run on container startup.
ENTRYPOINT ["/app/scripts/core_entrypoint.sh"]
# Use an official NVIDIA CUDA image as a base.
FROM ghcr.io/drumscoreai/rhythmform-core:latest

# Set the working directory in the container
WORKDIR /app/scripts

# Add a healthcheck to monitor the training process
HEALTHCHECK --interval=60s --timeout=30s --start-period=5m \
  CMD pgrep -f "python -m omr_model.fine_tune" || exit 1

# Specify the command to run on container startup.
CMD ["python", "-m", "omr_model.fine_tune", "--pretrained-checkpoint", "/app/checkpoints/.....", "--finetune-manifest", "/app/training_data/finetune_dataset.json"]
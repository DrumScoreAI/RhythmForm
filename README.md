# RhythmForm

An AI-powered utility to convert PDF drum scores into interactive MusicXML files. RhythmForm uses a transformer-based Optical Music Recognition (OMR) model specialized for percussion notation.

The goal of this project is to provide drummers and musicians with a tool to digitize their legally-owned sheet music for practice, editing, and analysis.

**Disclaimer:** This tool is for use with sheet music that you have the legal right to process. The user is solely responsible for ensuring they are not infringing on any copyrights.

**References:** This work was inspired by the [Sheet Music Transformer](https://github.com/antoniorv6/SMT) project [arXiv article](https://arxiv.org/abs/2405.12105).

## Features

- 🥁 **Drum-Focused OMR**: A transformer model trained specifically on drum and percussion notation for higher accuracy.
- 📄 **PDF to MusicXML**: Converts multi-page PDF drum scores into standard MusicXML files.
- 🎼 **Interactive Output**: The generated MusicXML can be imported into MuseScore, GarageBand, or any other notation software that supports the format.
- 🤖 **Synthetic Data Pipeline**: Includes a full suite of scripts for generating synthetic training data.

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch
- MuseScore Studio (for rendering synthetic data and viewing output)

### Install from source

```bash
git clone https://github.com/DrumScoreAI/RhythmForm.git
cd RhythmForm
pip install -r requirements.txt
pip install -e .
```

## Usage

**(Note: This describes the final intended usage. The model is currently under development.)**

```bash
rhythmform convert "/path/to/my/drum_score.pdf" --output "my_score.xml"
```

This command will:
1. Convert the PDF into a series of images.
2. Use the trained `RhythmForm` model to process each image into a symbolic music representation.
3. Convert the symbolic representation into a valid MusicXML file.
4. Save the file to the specified output path.

## Alternatively, use the pre-built Docker images

Pre-built Docker images are available on GitHub Container Registry (GHCR) for different stages of the RhythmForm workflow:

- **`ghcr.io/drumscoreai/rhythmform-core:latest`**  
  The full package image with all dependencies installed.  
  *No default command is set.*  
  Example:
  ```sh
  docker run -it --rm ghcr.io/drumscoreai/rhythmform-core:latest bash
  ```

  **`ghcr.io/drumscoreai/rhythmform-core-cpu:latest`**  
  The full package image without GPU dependencies.  
  *No default command is set.*
  Use this container for data synthesis.
  Do not use this image for training as it will not recognise GPUs.
  Examples:
  ```sh
  docker run -it --rm ghcr.io/drumscoreai/rhythmform-core-cpu:latest bash
  ```
  ```sh
  docker run -it --rm -v /training_data:/app/training_data ghcr.io/drumscoreai/rhythmform-core-cpu:latest /app/scripts/synthesized.sh -s 100 -n 4
  ```
  where `-s` defines the number of scores to be generated and `-n` defines the maximum number of CPUs to be used for any parallelised tasks. Use `-h` for additional options.

**Note:**  
- All images expect data to be mounted at `/app/training_data` for reading and writing datasets, models, and outputs.

## How It Works

RhythmForm uses a three-stage process:

### 1. Data Preparation
A dataset of paired `(image, music_data)` is created. This project includes a powerful synthetic data generator (`scripts/generate_synthetic_score.py`) that creates random but musically plausible drum scores in MusicXML, which are then rendered to PDF/PNG to create a perfectly aligned dataset.

#### Latest dataset
The latest synthesized dataset can be found at:

| S3 URI | Format | File Count |
|--------|--------|------------|
| - | zip | 0 |
| - | tar.gz | 0 |

### 2. Model Training
A transformer-based sequence-to-sequence model is trained on the dataset to learn the mapping from score images to a symbolic text (ST) representation.

### 3. Inference and Conversion
- The `convert` command takes a user's PDF.
- It runs the trained model on the PDF images to generate an ST string.
- A separate script parses this ST string and uses the `music21` library to programmatically construct a MusicXML score, which is then saved to a file.

#### Latest models
The latest models can be found at:

| S3 URI | Training strategy | Size |
|--------|-------------------|------|
| - | Synthetic | 0 |
| - | Fine-tuned | 0 |

## Project Structure

```
RhythmForm/
├── containers/
│   ├── core/
│   │   ├── Dockerfile
│   │   └── core_entrypoint.sh
│   ├── synthesizer/
│   │   └── Dockerfile
│   ├── trainer/
│   │   └── Dockerfile
│   └── predicter/
│       └── Dockerfile
├── scripts/
│   ├── generate_synthetic_score.py      # Synthetic MusicXML generator
│   ├── prepare_dataset.py               # Converts MusicXML to images and creates dataset manifest
│   └── omr_model/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── tokenizer.py
│       ├── train.py
│       ├── model.py
│       └── ...
├── training_data/
│   ├── musicxml/
│   ├── images/
│   ├── dataset.json
│   └── tokenizer_vocab.json
├── requirements.txt
├── .dockerignore
├── .gitignore
└── .github/
    └── workflows/
        ├── docker-build-core.yaml
        ├── docker-build-synthesizer.yaml
        ├── docker-build-trainer.yaml
        └── docker-build-predicter.yaml
```

## Typical Workflow

1. **Generate synthetic scores:**  
   `python scripts/generate_synthetic_score.py`
2. **Prepare dataset (convert XML to images, create manifest):**  
   `python scripts/prepare_dataset.py`
3. **Build tokenizer vocabulary:**  
   `python -m scripts.omr_model.tokenizer`
4. **Train the model:**  
   `python -m scripts.omr_model.train`

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

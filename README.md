# RhythmForm

An AI-powered utility to convert PDF drum scores into interactive MusicXML files. RhythmForm uses a transformer-based Optical Music Recognition (OMR) model specialized for percussion notation.

The goal of this project is to provide drummers and musicians with a tool to digitize their legally-owned sheet music for practice, editing, and analysis.

**Disclaimer:** This tool is for use with sheet music that you have the legal right to process. The user is solely responsible for ensuring they are not infringing on any copyrights.

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

## How It Works

RhythmForm uses a three-stage process:

### 1. Data Preparation
A dataset of paired `(image, music_data)` is created. This project includes a powerful synthetic data generator (`scripts/generate_synthetic_score.py`) that creates random but musically plausible drum scores in MusicXML, which are then rendered to PDF/PNG to create a perfectly aligned dataset.

### 2. Model Training
A transformer-based sequence-to-sequence model is trained on the dataset to learn the mapping from score images to a symbolic text representation (SMT).

### 3. Inference and Conversion
- The `convert` command takes a user's PDF.
- It runs the trained model on the PDF images to generate an SMT string.
- A separate script parses this SMT string and uses the `music21` library to programmatically construct a MusicXML score, which is then saved to a file.

## Project Structure

```
RhythmForm/
├── omr_model/            # Main model and training package
│   ├── __init__.py
│   ├── train.py         # Model training script
│   ├── model.py         # Transformer model definition
│   └── dataset.py       # PyTorch dataset and dataloader
├── scripts/              # Helper and processing scripts
│   ├── prepare_dataset.py # Processes raw data into a dataset.json
│   ├── generate_synthetic_score.py # Creates synthetic MusicXML scores
│   ├── predict.py         # (Future) Inference script for PDF-to-SMT
│   └── smt_to_musicxml.py # (Future) SMT-to-MusicXML converter
├── training_data/        # Directory for training data
│   ├── pdfs/
│   ├── musicxml/
│   └── dataset.json
├── setup.py              # Package setup
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
```# RhythmForm

An AI-powered utility to convert PDF drum scores into interactive MusicXML files. RhythmForm uses a transformer-based Optical Music Recognition (OMR) model specialized for percussion notation.

The goal of this project is to provide drummers and musicians with a tool to digitize their legally-owned sheet music for practice, editing, and analysis.

**Disclaimer:** This tool is for use with sheet music that you have the legal right to process. The user is solely responsible for ensuring they are not infringing on any copyrights.

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

## How It Works

RhythmForm uses a three-stage process:

### 1. Data Preparation
A dataset of paired `(image, music_data)` is created. This project includes a powerful synthetic data generator (`scripts/generate_synthetic_score.py`) that creates random but musically plausible drum scores in MusicXML, which are then rendered to PDF/PNG to create a perfectly aligned dataset.

### 2. Model Training
A transformer-based sequence-to-sequence model is trained on the dataset to learn the mapping from score images to a symbolic text representation (SMT).

### 3. Inference and Conversion
- The `convert` command takes a user's PDF.
- It runs the trained model on the PDF images to generate an SMT string.
- A separate script parses this SMT string and uses the `music21` library to programmatically construct a MusicXML score, which is then saved to a file.

## Project Structure

```
RhythmForm/
├── omr_model/            # Main model and training package
│   ├── __init__.py
│   ├── train.py         # Model training script
│   ├── model.py         # Transformer model definition
│   └── dataset.py       # PyTorch dataset and dataloader
├── scripts/              # Helper and processing scripts
│   ├── prepare_dataset.py # Processes raw data into a dataset.json
│   ├── generate_synthetic_score.py # Creates synthetic MusicXML scores
│   ├── predict.py         # (Future) Inference script for PDF-to-SMT
│   └── smt_to_musicxml.py # (Future) SMT-to-MusicXML converter
├── training_data/        # Directory for training data
│   ├── pdfs/
│   ├── musicxml/
│   └── dataset.json
├── setup.py              # Package setup
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

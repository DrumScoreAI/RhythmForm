# RhythmForm Installation Guide

This guide will walk you through setting up the `RhythmForm` project and its dependencies on a Linux or macOS system.

## Option 1 - Install from source

### 1. Prerequisites

Before you begin, ensure you have the following installed:

-   **Python**: Version 3.10 or higher.
-   **Git**: For cloning the repository.
-   **MuseScore Studio**: This is required for the data generation pipeline (to render synthetic MusicXML scores into images) and is highly recommended for viewing the final MusicXML output. You can download it from [musescore.org](https://musescore.org/).

### 2. Installation Steps

#### Step 1: Clone the Repository
Open your terminal and clone the `RhythmForm` repository from GitHub:
```bash
git clone https://github.com/DrumScoreAI/RhythmForm.git
cd RhythmForm
```

#### Step 2: Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```
Your terminal prompt should now be prefixed with `(venv)`.

#### Step 3: Install Dependencies
Install all required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

#### Step 4: Install RhythmForm
Install the `RhythmForm` package itself in "editable" mode. This allows you to run the command-line tool while still being able to edit the source code.
```bash
pip install -e .
```

### 3. Verifying the Installation

You can verify that the package is installed correctly by checking its version or help menu:
```bash
rhythmform --version
```
or
```bash
rhythmform --help
```
This should display the version number or the list of available commands without any errors.

## Option 2 - use pre-build Docker images

### 1. Prerequisites

Before you begin, ensure you have the following installed:

-   **Docker Engine**: Either Docker Engine, or some other way to run Docker images.

### 2. Docker `run` Commands

To interactively explore the package, or to run any script directly:

```sh
docker run -it --rm ghcr.io/drumscoreai/rhythmform-core:latest bash
```

To synthesis training data for RhythmForm (pairs of drum score PDF files and symbolic text representations):

```sh
docker run --rm -v /path/to/training_data:/app/training_data ghcr.io/drumscoreai/rhythmform-synthesizer:latest
```

To run training:

```sh
docker run --rm -v /path/to/training_data:/app/training_data ghcr.io/drumscoreai/rhythmform-trainer:latest
```

For more info, see [README.md](README.md)

### 4. Next Steps

The installation is now complete.

-   To understand the project's goals and structure, read the main [README.md](README.md).
-   To learn how to use the tool for converting PDFs, see the [USAGE.md](USAGE.md) file.
-   To train the model, you will first need to generate a dataset. See the documentation on the data generation scripts.
"""
A command-line script to test the SMT OMR model using the author's
custom code from the official GitHub repository.
"""
import logging
from pathlib import Path
import sys
import os

import click
from PIL import Image
from pdf2image import convert_from_path
import torch
import numpy as np

# --- Add the SMT repository to the Python path ---
# Use a direct path to the non-nested repository location
SMT_PATH = Path("/home/dave/personal/SMT")
if SMT_PATH.is_dir():
    sys.path.append(str(SMT_PATH))
else:
    print(f"Error: SMT repository not found at {SMT_PATH}")
    print("Please clone it first: git clone https://github.com/antoniorv6/SMT.git")
    sys.exit(1)
# ---------------------------------------------------------

# --- Import from the SMT repository ---
try:
    from smt_model import SMTModelForCausalLM
    from data_augmentation.data_augmentation import convert_img_to_tensor
except ImportError as e:
    print(f"Error importing from SMT repository: {e}")
    sys.exit(1)
# --------------------------------------

# --- Configuration ---
MODEL_NAME = "antoniorv6/smt-grandstaff"
# ---------------------

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_image(image: Image.Image, model, device) -> str:
    """Uses the SMT model to transcribe a single image."""
    try:
        # --- Image Pre-processing with Padding ---
        # The model expects a fixed input size. To avoid distortion, we pad the image.
        target_height = 128
        target_width = 1024
        
        # Resize image while maintaining aspect ratio
        original_width, original_height = image.size
        new_width = int(target_height * original_width / original_height)
        
        # Ensure the new width does not exceed the target width
        if new_width > target_width:
            new_width = target_width
            
        resized_image = image.resize((new_width, target_height), Image.Resampling.LANCZOS)

        # Create a new blank image with the target size and paste the resized image
        new_image = Image.new("RGB", (target_width, target_height), "white")
        new_image.paste(resized_image, (0, 0))
        # --- End of Pre-processing ---

        # Convert PIL Image to NumPy array (as cv2.imread would produce)
        image_np = np.array(new_image)
        image_bgr = image_np[:, :, ::-1] # Convert RGB to BGR

        # Use the author's custom function. It will handle the final conversion.
        image_tensor = convert_img_to_tensor(image_bgr).unsqueeze(0).to(device)
        
        predictions, _ = model.predict(image_tensor, convert_to_str=True)

        # Use the author's custom formatting for the output
        output_str = "".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')
        return output_str

    except Exception as e:
        logging.error(f"Error during OMR transcription: {e}", exc_info=True)
        return ""

@click.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(file_path: Path):
    """
    Transcribes a single image or the first page of a PDF file using the
    antoniorv6/smt-grandstaff model and prints the raw symbolic text output.
    """
    click.echo(f"Processing file: {file_path}")

    image = None
    try:
        if file_path.suffix.lower() == '.pdf':
            click.echo("File is a PDF, converting first page to image...")
            images = convert_from_path(file_path, first_page=1, last_page=1, dpi=300)
            if images:
                image = images[0]
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            click.echo("File is an image, loading...")
            image = Image.open(file_path).convert("RGB")
        else:
            click.secho(f"Error: Unsupported file type '{file_path.suffix}'. Please provide a PDF or image file.", fg='red')
            return

        if image is None:
            click.secho("Error: Could not load or convert the file to an image.", fg='red')
            return

    except Exception as e:
        click.secho(f"Error processing file: {e}", fg='red')
        logging.error(f"Failed to load/convert {file_path}", exc_info=True)
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        click.echo(f"Using device: {device}")
        click.echo(f"Loading OMR model ({MODEL_NAME})... This may take a moment.")
        model = SMTModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        model.eval() # Set model to evaluation mode
        click.echo("Model loaded successfully.")
    except Exception as e:
        click.secho(f"Fatal: Could not load the OMR model. {e}", fg='red')
        logging.error("Failed to load model", exc_info=True)
        return

    click.echo("Transcribing image...")
    symbolic_text = transcribe_image(image, model, device)

    click.echo("\n" + "="*20 + " RAW SYMBOLIC TEXT OUTPUT " + "="*20)
    click.echo(symbolic_text)
    click.echo("="*66)


if __name__ == '__main__':
    main()
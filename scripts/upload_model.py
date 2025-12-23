import os
import s3fs
import datetime
import logging
import argparse
import json
from glob import glob
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# The directory where model checkpoints are saved.
SOURCE_DIR = "/app/checkpoints"

# S3 Configuration from environment variables
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

def upload_to_s3(file_path, bucket, object_name=None):
    """Upload a file to an S3-compatible bucket using s3fs."""
    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        # Create S3 filesystem object
        s3 = s3fs.S3FileSystem(
            key=AWS_ACCESS_KEY_ID,
            secret=AWS_SECRET_ACCESS_KEY,
            client_kwargs={'endpoint_url': S3_ENDPOINT_URL}
        )

        # Define the full S3 path
        s3_path = f"{bucket}/{object_name}"

        logging.info(f"Uploading {file_path} to {s3_path}...")
        # Use s3.put to upload the file with a progress bar
        with open(file_path, 'rb') as f:
            with tqdm.wrapattr(f, "write", total=os.path.getsize(file_path), desc=f"Uploading {os.path.basename(file_path)}", leave=False) as f_wrapped:
                s3.put_file(f_wrapped, s3_path)
        logging.info(f"Successfully uploaded {os.path.basename(file_path)}.")
    except FileNotFoundError:
        logging.error(f"The file was not found: {file_path}")
        return False
    except Exception as e:
        logging.error(f"An error occurred during S3 upload for {file_path}: {e}")
        return False
    return True

def main():
    """Main function to find and upload model checkpoints."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Upload model checkpoints (.pth files) to an S3 bucket.")
    parser.add_argument(
        "--note",
        type=str,
        default="No metadata note provided.",
        help="A metadata note to include (e.g., 'Models from training run on A100')."
    )
    parser.add_argument(
        "--bucket-name",
        "-b",
        type=str,
        required=False,
        help="The name of the S3 bucket to upload the models to."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=SOURCE_DIR,
        help=f"The directory to search for .pth files (default: {SOURCE_DIR})."
    )
    args = parser.parse_args()

    # Determine the bucket name
    bucket_name = args.bucket_name or S3_BUCKET_NAME

    # Validate environment variables
    if not all([S3_ENDPOINT_URL, bucket_name, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        logging.error("One or more required S3 environment variables are not set.")
        logging.error("Please set: S3_ENDPOINT_URL, --bucket-name <bucket_name> or S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        return

    # --- Find Model Files ---
    logging.info(f"Searching for .pth model files in {args.source_dir}...")
    model_files = glob(os.path.join(args.source_dir, "*.pth"))

    if not model_files:
        logging.warning(f"No .pth files found in {args.source_dir}. Nothing to upload.")
        return

    logging.info(f"Found {len(model_files)} model files to upload.")

    # --- Metadata Creation ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_filename = f"models_metadata_{timestamp}.json"
    
    uploaded_models_info = []
    for f in model_files:
        uploaded_models_info.append({
            "filename": os.path.basename(f),
            "size_bytes": os.path.getsize(f)
        })

    metadata = {
        "creation_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": args.note,
        "uploaded_models": uploaded_models_info,
        "model_count": len(model_files)
    }
    
    # Save metadata to a temporary local file
    local_metadata_path = f"/tmp/{metadata_filename}"
    with open(local_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Created metadata file for this upload: {metadata_filename}")

    # --- Upload Models and Metadata ---
    for model_path in tqdm(model_files, desc="Uploading all models", unit="file"):
        upload_to_s3(model_path, bucket_name)
    
    logging.info("Uploading metadata file...")
    upload_to_s3(local_metadata_path, bucket_name, object_name=metadata_filename)
    
    # Clean up local metadata file
    os.remove(local_metadata_path)

    logging.info("Script finished.")

if __name__ == "__main__":
    main()

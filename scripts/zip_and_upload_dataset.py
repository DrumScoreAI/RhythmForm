import os
import zipfile
import tarfile
import s3fs
import datetime
import logging
import argparse
import json
from tqdm import tqdm
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
SOURCE_DIR = "/app/training_data"
ARCHIVE_SUBDIR = "zips"
ARCHIVE_DIR = os.path.join(SOURCE_DIR, ARCHIVE_SUBDIR)

# S3 Configuration from environment variables
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

def create_zip_archive(source_dir, archive_path, exclude_dir):
    """Creates a zip archive of a directory, excluding a specific subdirectory."""
    logging.info(f"Creating zip archive: {archive_path}")
    
    # First, gather a list of all files to be archived for an accurate total
    file_paths = []
    for root, _, files in os.walk(source_dir):
        if root == exclude_dir:
            continue
        for file in files:
            file_paths.append(os.path.join(root, file))

    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate over the list of files with a tqdm progress bar
        for file_path in tqdm(file_paths, desc="Zipping files", unit="file", leave=False):
            arcname = os.path.relpath(file_path, source_dir)
            try:
                zipf.write(file_path, arcname)
            except Exception as e:
                logging.error(f"Error adding file {file_path} to zip: {e}")
                continue
            
    logging.info("Zip archive created successfully.")
    return(len(file_paths))

def create_tar_gz_archive(source_dir, archive_path, exclude_dir):
    """Creates a .tar.gz archive of a directory, excluding a specific subdirectory."""
    logging.info(f"Creating tar.gz archive: {archive_path}")
    
    # First, gather a list of all files to be archived for an accurate total
    file_paths = []
    for root, _, files in os.walk(source_dir):
        if root == exclude_dir:
            continue
        for file in files:
            file_paths.append(os.path.join(root, file))

    with tarfile.open(archive_path, "w:gz") as tar:
        # Iterate over the list of files with a tqdm progress bar
        for file_path in tqdm(file_paths, desc="Taring files", unit="file", leave=False):
            arcname = os.path.relpath(file_path, source_dir)
            try:
                tar.add(file_path, arcname=arcname)
            except Exception as e:
                logging.error(f"Error adding file {file_path} to tar.gz: {e}")
                continue
            
    logging.info("tar.gz archive created successfully.")
    return(len(file_paths))

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
        # Use s3.put to upload the file
        s3.put(file_path, s3_path)
        logging.info("Upload successful.")
    except FileNotFoundError:
        logging.error(f"The file was not found: {file_path}")
        return False
    except Exception as e:
        logging.error(f"An error occurred during S3 upload: {e}")
        return False
    return True

def main():
    """Main function to orchestrate archiving and uploading."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Archive and upload the training dataset to S3.")
    parser.add_argument(
        "--note",
        type=str,
        default="No metadata note provided.",
        help="A metadata note to include in the archive (e.g., '1st generation synthetic data')."
    )
    parser.add_argument(
        "--bucket-name",
        "-b",
        type=str,
        required=False,
        help="The name of the S3 bucket to upload the archives to."
    )
    parser.add_argument(
        "--tar-only",
        action='store_true',
        help="If set, only create and upload the tar.gz archive."
    )
    parser.add_argument(
        "--zip-only",
        action='store_true',
        help="If set, only create and upload the zip archive."
    )
    args = parser.parse_args()

    # Validate environment variables
    if not all([S3_ENDPOINT_URL, args.bucket_name or S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        logging.error("One or more required S3 environment variables are not set.")
        logging.error("Please set: S3_ENDPOINT_URL, --bucket-name <bucket_name> or S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        return

    # Ensure the archive directory exists
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # --- Metadata Creation ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate a unique filename based on the current timestamp
    base_filename = f"training_data_{timestamp}"
    
    zip_path = os.path.join(ARCHIVE_DIR, f"{base_filename}.zip")
    tar_path = os.path.join(ARCHIVE_DIR, f"{base_filename}.tar.gz")

    # Create archives (which will now include metadata.json)
    file_count_t = 0
    file_count_z = 0
    if not args.tar_only:
        logging.info("Creating zip archive...")
        file_count_z = create_zip_archive(SOURCE_DIR, zip_path, ARCHIVE_DIR)
    if not args.zip_only:
        logging.info("Creating tar.gz archive...")
        file_count_t = create_tar_gz_archive(SOURCE_DIR, tar_path, ARCHIVE_DIR)
    # If both archives were created, ensure file counts match
    # Verify file counts match
    if not args.tar_only and not args.zip_only:
        try:
            assert file_count_z == file_count_t, "File counts in zip and tar.gz do not match!"
        except AssertionError as e:
            logging.error(f"Assertion Error: {e}")
            exit(1)
    
    metadata = {
        "creation_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": args.note,
        "file_count": max(file_count_t, file_count_z)
    }
    metadata_path = os.path.join(SOURCE_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Created metadata file at {metadata_path}")

    # --- Cleanup metadata file ---
    # try:
    #     os.remove(metadata_path)
    #     logging.info(f"Cleaned up temporary metadata file: {metadata_path}")
    # except OSError as e:
    #     logging.error(f"Error removing metadata file: {e}")

    # Upload archives
    if args.bucket_name:
        bucket_name = args.bucket_name
    else:
        bucket_name = S3_BUCKET_NAME
    if not args.zip_only:
        logging.info("Uploading tar.gz archive to S3...")
        upload_to_s3(tar_path, bucket_name)
    if not args.tar_only:
        logging.info("Uploading zip archive to S3...")
        upload_to_s3(zip_path, bucket_name)
    upload_to_s3(metadata_path, bucket_name)

    logging.info("Script finished.")

if __name__ == "__main__":
    main()
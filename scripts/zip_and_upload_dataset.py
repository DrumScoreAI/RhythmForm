import os
import zipfile
import tarfile
import s3fs
import datetime
import logging

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
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Exclude the archive directory itself from being walked
            if root == exclude_dir:
                dirs[:] = []  # This modifies the list in-place
                files[:] = []
                continue

            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    logging.info("Zip archive created successfully.")

def create_tar_gz_archive(source_dir, archive_path, exclude_dir):
    """Creates a .tar.gz archive of a directory, excluding a specific subdirectory."""
    logging.info(f"Creating tar.gz archive: {archive_path}")
    
    def exclude_filter(tarinfo):
        """Filter function for tarfile.add to exclude the archive directory."""
        path_to_check = os.path.abspath(tarinfo.name)
        if path_to_check.startswith(os.path.abspath(exclude_dir)):
            logging.debug(f"Excluding from tar: {tarinfo.name}")
            return None  # Exclude this item
        logging.debug(f"Including in tar: {tarinfo.name}")
        return tarinfo # Include this item

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir), filter=exclude_filter)
    logging.info("tar.gz archive created successfully.")

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
    # Validate environment variables
    if not all([S3_ENDPOINT_URL, S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        logging.error("One or more required S3 environment variables are not set.")
        logging.error("Please set: S3_ENDPOINT_URL, S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        return

    # Ensure the archive directory exists
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # Generate a unique filename based on the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"training_data_{timestamp}"
    
    zip_path = os.path.join(ARCHIVE_DIR, f"{base_filename}.zip")
    tar_path = os.path.join(ARCHIVE_DIR, f"{base_filename}.tar.gz")

    # Create archives
    create_zip_archive(SOURCE_DIR, zip_path, ARCHIVE_DIR)
    create_tar_gz_archive(SOURCE_DIR, tar_path, ARCHIVE_DIR)

    # Upload archives
    upload_to_s3(zip_path, S3_BUCKET_NAME)
    upload_to_s3(tar_path, S3_BUCKET_NAME)

    logging.info("Script finished.")

if __name__ == "__main__":
    main()
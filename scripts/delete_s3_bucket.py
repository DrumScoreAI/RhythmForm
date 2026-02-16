import os
import s3fs
import logging
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# S3 Configuration from environment variables
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

def delete_archives_from_s3(bucket_name, dry_run=False, assume_yes=False):
    """Deletes .zip and .tar.gz archives from an S3-compatible bucket."""
    try:
        # Create S3 filesystem object
        s3 = s3fs.S3FileSystem(
            key=AWS_ACCESS_KEY_ID,
            secret=AWS_SECRET_ACCESS_KEY,
            client_kwargs={'endpoint_url': S3_ENDPOINT_URL}
        )

        logging.info(f"Scanning bucket '{bucket_name}' for archives...")
        
        # List all objects in the bucket
        all_objects = s3.ls(bucket_name, detail=False)
        
        # Filter for .zip and .tar.gz files
        archives_to_delete = [
            obj for obj in all_objects 
            if obj.endswith('.zip') or obj.endswith('.tar.gz')
        ]

        if not archives_to_delete:
            logging.info("No .zip or .tar.gz archives found to delete.")
            return

        logging.info(f"Found {len(archives_to_delete)} archives to delete:")
        for archive in archives_to_delete:
            print(f" - {os.path.basename(archive)}")

        if dry_run:
            logging.info("Dry run complete. No files were deleted.")
            return

        if not assume_yes:
            confirm = input("Are you sure you want to delete these files? (yes/no): ")
            if confirm.lower() != 'yes':
                logging.info("Deletion cancelled by user.")
                return

        logging.info("Deleting archives...")
        for archive in archives_to_delete:
            try:
                s3.rm(archive)
                logging.info(f"Deleted {os.path.basename(archive)}")
            except Exception as e:
                logging.error(f"Failed to delete {os.path.basename(archive)}: {e}")
        
        logging.info("Deletion process completed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

def main():
    """Main function to orchestrate the deletion process."""
    parser = argparse.ArgumentParser(
        description="Delete all .zip and .tar.gz archives from an S3 bucket.",
        epilog="This script is destructive. Use with caution. It is recommended to run with --dry-run first."
    )
    parser.add_argument(
        "--bucket-name",
        "-b",
        type=str,
        required=False,
        help="The name of the S3 bucket. Overrides the S3_BUCKET_NAME environment variable."
    )
    parser.add_argument(
        "--dry-run",
        action='store_true',
        help="List the files that would be deleted without actually deleting them."
    )
    parser.add_argument(
        "--yes",
        "-y",
        action='store_true',
        help="Assume 'yes' to all prompts and run non-interactively."
    )
    args = parser.parse_args()

    # Determine bucket name
    bucket_name = args.bucket_name or S3_BUCKET_NAME

    # Validate environment variables and bucket name
    if not all([S3_ENDPOINT_URL, bucket_name, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        logging.error("One or more required S3 configuration variables are not set.")
        logging.error("Please set: S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and either S3_BUCKET_NAME or --bucket-name.")
        sys.exit(1)

    delete_archives_from_s3(bucket_name, args.dry_run, args.yes)

    logging.info("Script finished.")

if __name__ == "__main__":
    main()

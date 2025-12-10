
import os
import re
import s3fs
import sys

def get_latest_from_s3(s3, bucket_name, extension):
    """Find the lexicographically latest file with a given extension in an S3 bucket."""
    try:
        all_files = s3.ls(bucket_name, detail=False)
        target_files = [f for f in all_files if f.endswith(extension)]
        if not target_files:
            return None
        # The timestamp in the name means the alphabetically last file is the latest
        latest_file = sorted(target_files)[-1]
        return os.path.basename(latest_file)
    except Exception as e:
        print(f"Error listing S3 bucket '{bucket_name}': {e}")
        return None

def get_current_from_readme(readme_content, extension):
    """Extract the current dataset filename from the README content."""
    # Regex to find 'training_data_YYYY_MM_DD_HH_MM_SS.zip' or .tar.gz
    pattern = re.compile(f"training_data_[0-9_]*\\{extension}")
    match = pattern.search(readme_content)
    if not match:
        print(f"Warning: Could not find file with extension '{extension}' in README.md")
        return None
    return match.group(0)

def main():
    """
    Checks an S3 bucket for the latest datasets, compares them with the links
    in README.md, and updates the README if newer datasets are found.
    """
    # --- 1. Configuration ---
    # Get config from environment variables, with fallbacks for local testing
    endpoint_url = os.getenv("S3_ENDPOINT_URL", "https://s3.eidf.ac.uk")
    bucket_name = os.getenv("S3_BUCKET_NAME", "rhythmformdatasets")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # The script should be run from the repository root
    readme_path = "README.md"

    if not (aws_access_key_id and aws_secret_access_key):
        print("Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are not set.")
        sys.exit(1)

    print(f"Checking for latest datasets in s3://{bucket_name}...")

    # --- 2. Connect to S3 and find latest datasets ---
    try:
        s3 = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": endpoint_url},
            key=aws_access_key_id,
            secret=aws_secret_access_key,
        )
        latest_zip = get_latest_from_s3(s3, bucket_name, ".zip")
        latest_tar = get_latest_from_s3(s3, bucket_name, ".tar.gz")
    except Exception as e:
        print(f"Error connecting to S3: {e}")
        sys.exit(1)

    if not latest_zip or not latest_tar:
        print("Could not find both a .zip and .tar.gz file in the bucket. Exiting.")
        # Exit successfully because this is not a failure condition
        sys.exit(0)

    print(f"Latest .zip on S3: {latest_zip}")
    print(f"Latest .tar.gz on S3: {latest_tar}")

    # --- 3. Get current dataset names from README ---
    try:
        with open(readme_path, "r") as f:
            readme_content = f.read()
    except FileNotFoundError:
        print(f"Error: {readme_path} not found. This script should be run from the repository root.")
        sys.exit(1)

    current_zip = get_current_from_readme(readme_content, ".zip")
    current_tar = get_current_from_readme(readme_content, ".tar.gz")

    if not current_zip or not current_tar:
        print("Could not determine current dataset files from README.md. Exiting.")
        sys.exit(1)

    print(f"Current .zip in README: {current_zip}")
    print(f"Current .tar.gz in README: {current_tar}")

    # --- 4. Compare and update if necessary ---
    if latest_zip != current_zip or latest_tar != current_tar:
        print("Newer dataset found. Updating README.md...")
        
        new_readme_content = readme_content.replace(current_zip, latest_zip)
        new_readme_content = new_readme_content.replace(current_tar, latest_tar)

        try:
            with open(readme_path, "w") as f:
                f.write(new_readme_content)
            print("README.md updated successfully.")
            # Signal that a change was made for the subsequent workflow step
            print("::set-output name=updated::true")
        except Exception as e:
            print(f"Error writing to {readme_path}: {e}")
            sys.exit(1)
    else:
        print("README.md is already up to date. No changes needed.")
        print("::set-output name=updated::false")

if __name__ == "__main__":
    main()

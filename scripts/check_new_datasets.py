import os
import re
import s3fs
import sys
import pathlib

repository_root = pathlib.Path(os.environ.get("RHYTHMFORMHOME")) or pathlib.Path(__file__).parent.parent.resolve()

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

def get_metadata_from_s3(s3, bucket_name):
    """Retrieve and return the metadata JSON content for a given file in S3."""
    import json
    metadata_filename = 'metadata.json'
    s3_path = f"{bucket_name}/{metadata_filename}"
    try:
        with s3.open(s3_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error retrieving metadata file from S3.': {e}")
        return None

def get_current_from_readme(readme_content, extension):
    """
    Extract the current dataset filename and its line number from the README content.
    Returns a tuple of (filename, line_number).
    """
    # Regex to find lines like:
    # | https://s3.eidf.ac.uk/RhythmFormDatasets/training_data_20251201_000000.zip | zip | 0 |
    # and capture just the filename.
    pattern = re.compile(f"\\|.*(training_data_[0-9_]*\\{extension}).*\\|")
    for i, line in enumerate(readme_content.splitlines()):
        match = pattern.search(line)
        if match:
            # Return the captured filename and the 1-based line number
            return (match.group(1), i + 1)
    
    print(f"Warning: Could not find file with extension '{extension}' in the README.md dataset table.")
    return None

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
    readme_path = repository_root /"README.md"

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
        print(f"Error: {readme_path} not found. Check RHYTHMFORMHOME environment variable.")
        sys.exit(1)

    current_zip_info = get_current_from_readme(readme_content, ".zip")
    current_tar_info = get_current_from_readme(readme_content, ".tar.gz")

    if not current_zip_info or not current_tar_info:
        print("Could not determine current dataset files from README.md. Exiting.")
        sys.exit(1)

    current_zip, line_zip = current_zip_info
    current_tar, line_tar = current_tar_info

    print(f"Current .zip in README: {current_zip}")
    print(f"Current .tar.gz in README: {current_tar}")

    # --- 4. Compare and update if necessary ---
    if latest_zip != current_zip or latest_tar != current_tar:
        print("Newer dataset found. Updating README.md...")

        metadata = get_metadata_from_s3(s3, bucket_name)
        if metadata:
            print("Latest dataset metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        # --- 4a. Construct new content ---
        base_s3_url = f"{endpoint_url}/{bucket_name}"
        file_count = metadata.get('file_count', 0) if metadata else 0

        new_zip_uri = f"{base_s3_url}/{latest_zip}"
        new_tar_uri = f"{base_s3_url}/{latest_tar}"

        new_zip_line = f"| {new_zip_uri} | zip | {file_count} |"
        new_tar_line = f"| {new_tar_uri} | tar.gz | {file_count} |"

        # --- 4b. Replace lines in README ---
        readme_lines = readme_content.splitlines()
        # Replace the old lines with the newly constructed ones
        readme_lines[line_zip - 1] = new_zip_line
        readme_lines[line_tar - 1] = new_tar_line
        
        # Join the lines back together, ensuring a trailing newline
        new_readme_content = "\n".join(readme_lines) + "\n"

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

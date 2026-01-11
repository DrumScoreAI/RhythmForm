import argparse
from pathlib import Path
import subprocess
import sys
from multiprocessing import Pool, cpu_count

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parents[1]

def convert_file(smt_path, output_dir, python_executable):
    """
    Calls the smt_to_musicxml.py script on a single file.
    """
    smt_path = Path(smt_path)
    output_xml_path = output_dir / f"{smt_path.stem}.xml"
    
    script_path = get_project_root() / "scripts" / "smt_to_musicxml.py"
    
    command = [
        str(python_executable),
        str(script_path),
        "--input-smt",
        str(smt_path),
        "--output-xml",
        str(output_xml_path),
    ]
    
    try:
        # We use DEVNULL to hide the per-file print statements for a cleaner log
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully converted: {smt_path.name} -> {output_xml_path.name}")
        return None # Success
    except subprocess.CalledProcessError as e:
        error_message = f"Failed to convert {smt_path.name}.\n"
        error_message += f"  Return Code: {e.returncode}\n"
        error_message += f"  Stdout: {e.stdout.strip()}\n"
        error_message += f"  Stderr: {e.stderr.strip()}\n"
        return error_message # Return error message on failure

def main():
    project_root = get_project_root()
    
    parser = argparse.ArgumentParser(description="Batch convert .smt files to .musicxml using multiple processes.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "training_data" / "fine_tuning" / "smt",
        help="Directory containing the .smt files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "training_data" / "fine_tuning" / "musicxml",
        help="Directory to save the .musicxml files."
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=cpu_count(), 
        help='Number of worker processes to use.'
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .smt files
    smt_files = list(args.input_dir.glob("*.smt"))
    if not smt_files:
        print(f"No .smt files found in {args.input_dir}")
        return

    print(f"Found {len(smt_files)} .smt files to convert.")
    print(f"Using {args.workers} worker processes.")

    # Get the python executable from the current environment
    python_executable = sys.executable

    # Use a process pool to convert files in parallel
    with Pool(processes=args.workers) as pool:
        # Create a list of arguments for each task
        tasks = [(path, args.output_dir, python_executable) for path in smt_files]
        
        # map_async is used to apply the function to each task
        results = pool.starmap(convert_file, tasks)

    # Filter out successful (None) results and print errors
    errors = [res for res in results if res is not None]
    if errors:
        print("\n" + "="*30)
        print(f"Encountered {len(errors)} errors during conversion:")
        print("="*30)
        for error in errors:
            print(error)
    else:
        print("\nAll files converted successfully!")

if __name__ == "__main__":
    main()

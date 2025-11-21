"""
A command-line script to test the SymbolicToMusicXMLConverter.

This script reads a raw symbolic text file (as produced by the SMT model),
runs it through the parser, and saves the output as a MusicXML file.
"""
import logging
from pathlib import Path
import sys

import click

# Add project root to path to allow importing from the 'scorefinder' package
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from scorefinder.symbolic import SymbolicToMusicXMLConverter
except ImportError as e:
    print(f"Error: Could not import SymbolicToMusicXMLConverter. {e}")
    print(f"Please ensure you are running this script from the project's root directory.")
    sys.exit(1)

# Set up basic logging to see errors from the parser
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output', '-o', 'output_file', type=click.Path(dir_okay=False, writable=True, path_type=Path), help="Path to save the output MusicXML file.")
def main(input_file: Path, output_file: Path):
    """
    Parses a raw symbolic text file (from the SMT model) and converts it
    into a MusicXML file.
    """
    if not output_file:
        # Default output file name if not provided
        output_file = input_file.with_suffix('.musicxml')

    click.echo(f"Reading symbolic text from: {input_file}")
    try:
        symbolic_text = input_file.read_text()
    except Exception as e:
        click.secho(f"Error reading file: {e}", fg='red')
        return

    click.echo("Initializing parser...")
    parser = SymbolicToMusicXMLConverter()

    click.echo("Converting to MusicXML...")
    try:
        musicxml_output = parser.convert(symbolic_text)
        click.echo("Conversion successful.")
    except Exception as e:
        click.secho(f"An error occurred during parsing: {e}", fg='red')
        import traceback
        traceback.print_exc()
        return

    click.echo(f"Saving MusicXML to: {output_file}")
    try:
        output_file.write_text(musicxml_output)
        click.secho(f"Success! You can now open '{output_file}' in a score editor.", fg='green')
    except Exception as e:
        click.secho(f"Error writing to output file: {e}", fg='red')


if __name__ == '__main__':
    main()

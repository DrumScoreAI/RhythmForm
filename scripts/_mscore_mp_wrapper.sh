#!/bin/bash
# Script to batch convert MusicXML files to PDF using MuseScore
# For use as a wrapper called by synthesizer.sh with xargs for parallel processing

if [ $# -ne 1 ]; then
  echo "Usage: $0 <abs_path_to_musicxml_file>"
  exit 1
fi

if [ -z "$MUSESCORE_PATH" ]; then
  echo "Error: MUSESCORE_PATH environment variable is not set."
  exit 1
fi

xml_file=$1

filename="${xml_file##*/}"
xml_dir="${xml_file%/*}"
pdf_dir="${xml_dir}/../pdfs"
# echo "Converted $filename to ${filename%.xml}.pdf"

# Source - https://stackoverflow.com/a/30336424
# Posted by Charles Duffy, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-14, License - CC BY-SA 3.0
# allow settings to be updated via environment
: "${xvfb_lockdir:=$HOME/.xvfb-locks}"
: "${xvfb_display_min:=99}"
: "${xvfb_display_max:=59999}"

# assuming only one user will use this, let's put the locks in our own home directory
# avoids vulnerability to symlink attacks.
mkdir -p -- "$xvfb_lockdir" || exit

i=$xvfb_display_min     # minimum display number
while (( i < xvfb_display_max )); do
  if [ -f "/tmp/.X$i-lock" ]; then                # still avoid an obvious open display
    (( ++i )); continue
  fi
  exec 5>"$xvfb_lockdir/$i" || continue           # open a lockfile
  if flock -x -n 5; then                          # try to lock it
    exec xvfb-run --server-num="$i" "$MUSESCORE_PATH" -o "${pdf_dir}/${filename%.xml}.pdf" -r 300 "${xml_file}" 2>&1 | grep -v -E "pw.context|pw.conf|libOpenGL|libjack|libnss3|libpipewire"  # if locked, run xvfb-run
  fi
  (( i++ ))
done
echo "Error: No available X displays in range $xvfb_display_min to $xvfb_display_max" >&2
exit 1
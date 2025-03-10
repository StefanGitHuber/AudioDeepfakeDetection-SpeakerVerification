#!/bin/bash

<<COMMENT
Bash script to convert
from any speech recording (raw = whatever audio format it is)
to signed 16 Bit 16 kHz wave (since most speech models are trained on uncompressed waveforms).

Please install the awesome SoX (Sound eXchange) command line tool http://sox.sourceforge.net/
to perform simple sample rate conversion https://pysox.readthedocs.io/en/latest/_modules/sox/transform.html
COMMENT

# Set input and output folders
input_folder="/your/path/to/data_raw"
output_folder="/your/path/to/data_orig"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Loop through all .wav files in the input folder
for input_file in "$input_folder"/*.wav; do
  if [[ -f "$input_file" ]]; then
    filename=$(basename "$input_file")
    filename_no_ext="${filename%.*}"
    output_file="$output_folder/${filename_no_ext}_mono_16k.wav"

    sox "$input_file" -c 1 -b 16 -e signed -r 16000 "$output_file"

    if [ $? -eq 0 ]; then
      echo "********************************************************"
      soxi "$input_file"
      soxi "$output_file"
    else
      echo "Error converting $filename!"
    fi
  fi
done

echo "********************************************************"
echo "Conversion of $input_folder complete."

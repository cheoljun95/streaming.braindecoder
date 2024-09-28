#!/bin/bash

# Loop through numbers 0 to 33, formatted with leading zeros (6 digits)
for n in $(seq -f "%06g" 0 33); do
    # Define the filename
    filename="d_${n}.tar.gz"
    
    # Check if the file exists
    if [ -f "$filename" ]; then
        echo "Extracting $filename..."
        tar -xzvf "$filename"
    else
        echo "$filename not found."
    fi
done
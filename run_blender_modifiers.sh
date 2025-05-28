#!/bin/bash

# Path to Blender executable
BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"

# Check if Blender exists at the specified path
if [ ! -f "$BLENDER_PATH" ]; then
    echo "Error: Blender not found at $BLENDER_PATH"
    echo "Please modify the BLENDER_PATH in this script to point to your Blender installation"
    exit 1
fi

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [subdivision_levels] [noise_scale] [noise_strength]"
    echo "Example: $0 test_highpolygon_obj modified_noise_meshes 2 0.1 0.1"
    exit 1
fi

# Set default values if not provided
SUBDIVISION_LEVELS=${3:-2}
NOISE_SCALE=${4:-0.1}
NOISE_STRENGTH=${5:-0.1}

# Create a temporary Python script with the arguments
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << EOF
import sys
import os
import bpy

# Set up the arguments
sys.argv = [
    "blender",
    "$1",
    "$2",
    "--subdivision-levels",
    "$SUBDIVISION_LEVELS",
    "--noise-scale",
    "$NOISE_SCALE",
    "--noise-strength",
    "$NOISE_STRENGTH"
]

# Get the absolute path to the script
script_path = os.path.abspath("src/models/apply_blender_modifiers.py")

# Execute the script
with open(script_path, 'r') as f:
    exec(f.read())
EOF

# Run Blender with the temporary script
"$BLENDER_PATH" --background --python "$TEMP_SCRIPT"

# Clean up
rm "$TEMP_SCRIPT" 
#!/bin/bash

# Navigate to the 'example' directory
cd ./examples || { echo "Directory not found"; exit 1; }

# Loop through all Python files and execute them
for script in *.py; do
    # Check if there are no Python files
    if [ "$script" = "*.py" ]; then
        echo "No Python scripts found in the directory."
        break
    fi

    # Execute the Python script
    echo "Running $script..."
    python "$script"
done

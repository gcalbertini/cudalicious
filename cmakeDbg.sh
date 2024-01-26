#!/bin/bash

BUILD_DIR="build"
DEBUG_DIR="${BUILD_DIR}/Debug"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

# Check if Debug directory exists
if [ ! -d "$DEBUG_DIR" ]; then
    echo "Running CMake configuration (Debug build)..."
    cd "$BUILD_DIR" || exit
    cmake -DCMAKE_BUILD_TYPE=Debug ..
fi

# Build the project
echo "Building the project..."
cmake --build "$BUILD_DIR" --config Debug

# Navigate to Debug folder
cd "$DEBUG_DIR" || exit

# Check if a script name is provided as a command line argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_name>"
    exit 1
fi

SCRIPT_NAME="$1"

# Run the specified script
echo "Running ${SCRIPT_NAME}..."
./"${SCRIPT_NAME}"

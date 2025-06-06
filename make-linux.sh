#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [debug|release]"
    exit 1
fi

# Set the build type based on the argument
BUILD_TYPE=""
if [ "$1" == "debug" ]; then
    BUILD_TYPE=Debug
elif [ "$1" == "release" ]; then
    BUILD_TYPE=Release
else
    echo "Invalid argument: $1"
    echo "Usage: $0 [debug|release]"
    exit 1
fi

# Define the build directory
BUILD_DIR=bin/linux_$BUILD_TYPE

# Create the build directory if it doesn't exist
mkdir -p $BUILD_DIR
cd $BUILD_DIR || exit 1

# Run CMake with the specified build type
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ../..

# Build the project
cmake --build .

# Return to the original directory
cd ../..

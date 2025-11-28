#!/usr/bin/env bash

# This script is designed to be run from the ROOT directory.

# 1. Install Python dependencies
echo "Installing Python dependencies..."
# requirements.txt is in the root
pip install -r requirements.txt

# 2. Build the React frontend
echo "Building React frontend..."
# package.json is in the root (no --prefix needed)
npm install
npm run build 
# This will create a 'dist' directory in the root

echo "Build process completed successfully."
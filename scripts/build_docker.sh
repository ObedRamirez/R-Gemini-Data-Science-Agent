#!/bin/bash
echo "Building R Docker image..."
cd docker
docker build -f Dockerfile.consolidated -t r_data_science:v2.0 .
echo "✓ Build complete!"

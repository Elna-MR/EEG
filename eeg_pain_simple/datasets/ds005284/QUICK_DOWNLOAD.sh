#!/bin/bash
# Quick download script - uses pip-installed AWS CLI

set -e

echo "=========================================="
echo "ds005284 Dataset Download"
echo "=========================================="
echo ""

# Add pip-installed AWS CLI to PATH
export PATH="$HOME/Library/Python/3.10/bin:$PATH"

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI not found!"
    echo ""
    echo "Installing AWS CLI via pip..."
    pip3 install awscli --user
    export PATH="$HOME/Library/Python/3.10/bin:$PATH"
    
    if ! command -v aws &> /dev/null; then
        echo "ERROR: Installation failed. Please install manually:"
        echo "  pip3 install awscli --user"
        exit 1
    fi
fi

echo "✓ AWS CLI version: $(aws --version 2>&1 | head -1)"
echo ""

# Navigate to download directory
DOWNLOAD_DIR="/Users/elnaz/Desktop/repo/EEG/eeg_pain_simple/datasets/downloads"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "[INFO] Starting download from S3..."
echo "[INFO] This will download ~16-20 GB (may take 30-60 minutes)"
echo ""

# Download from S3
aws s3 sync --no-sign-request s3://openneuro.org/ds005284 ds005284-download/

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""

# Count downloaded BDF files
BDF_COUNT=$(find ds005284-download -name "*.bdf" 2>/dev/null | wc -l | tr -d ' ')
echo "[INFO] Found $BDF_COUNT BDF files"

if [ "$BDF_COUNT" -gt 0 ]; then
    echo ""
    echo "[INFO] Copying BDF files to project structure..."
    
    cd ds005284-download
    COPIED=0
    
    for sub_dir in sub-*/eeg/*.bdf; do
        if [ -f "$sub_dir" ]; then
            sub=$(echo "$sub_dir" | cut -d'/' -f1)
            target_dir="../ds005284/data/$sub/eeg"
            mkdir -p "$target_dir"
            cp "$sub_dir" "$target_dir/"
            echo "  ✓ Copied: $sub"
            ((COPIED++))
        fi
    done
    
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo "Copied $COPIED BDF files"
    echo ""
    echo "Next: Run the pipeline"
    echo "  cd ../ds005284"
    echo "  ./run_pipeline.sh"
else
    echo "[WARN] No BDF files found"
fi


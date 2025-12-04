#!/bin/bash
# Pipeline script for osf-data preprocessing, feature extraction, and DANN training
# Same pipeline as ds005284 for comparison

set -e

echo "=========================================="
echo "osf-data Pipeline"
echo "=========================================="

cd "$(dirname "$0")"

# Step 1: Preprocess
echo ""
echo "Step 1: Preprocessing..."
python scripts/preprocess.py

# Step 2: Extract features
echo ""
echo "Step 2: Extracting features..."
python scripts/extract_features.py \
    --npz packed/osf-data_pain.npz \
    --out packed/features_osf-data.npz

if [ ! -f "packed/features_osf-data.npz" ]; then
    echo "ERROR: Feature extraction failed!"
    exit 1
fi

# Step 3: Train DANN
echo ""
echo "Step 3: Training DANN..."
python scripts/train_dann.py \
    --features packed/features_osf-data.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results saved to reports/ directory"
echo "=========================================="



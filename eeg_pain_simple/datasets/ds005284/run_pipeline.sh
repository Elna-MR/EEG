#!/bin/bash
# Complete pipeline for ds005284 dataset
# Run from datasets/ds005284/ directory

set -e

echo "=========================================="
echo "ds005284 Complete Pipeline"
echo "=========================================="
echo ""

# Step 1: Preprocess
echo "[1/3] Preprocessing ds005284 data..."
python3 scripts/preprocess.py

if [ ! -f "packed/ds005284_pain.npz" ]; then
    echo "ERROR: Preprocessing failed!"
    exit 1
fi

echo ""
echo "[2/3] Extracting Riemannian features..."
python3 scripts/extract_features.py \
    --npz packed/ds005284_pain.npz \
    --out packed/features_ds005284.npz

if [ ! -f "packed/features_ds005284.npz" ]; then
    echo "ERROR: Feature extraction failed!"
    exit 1
fi

echo ""
echo "[3/3] Training DANN model..."
python3 scripts/train_dann.py \
    --features packed/features_ds005284.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - Preprocessed: packed/ds005284_pain.npz"
echo "  - Features: packed/features_ds005284.npz"
echo "  - Reports: reports/"


#!/bin/bash
# Complete pipeline for cpCGX_BIDS dataset
# Run from datasets/cpcgx/ directory

set -e

echo "=========================================="
echo "cpCGX_BIDS Complete Pipeline"
echo "=========================================="
echo ""

# Step 1: Preprocess
echo "[1/3] Preprocessing cpCGX_BIDS data..."
python scripts/preprocess_cpcgx.py \
    --root data \
    --out packed/cpcgx_pain.npz

if [ ! -f "packed/cpcgx_pain.npz" ]; then
    echo "ERROR: Preprocessing failed!"
    exit 1
fi

echo ""
echo "[2/3] Extracting Riemannian features..."
python scripts/extract_features.py \
    --npz packed/cpcgx_pain.npz \
    --out packed/features_cpcgx.npz

if [ ! -f "packed/features_cpcgx.npz" ]; then
    echo "ERROR: Feature extraction failed!"
    exit 1
fi

echo ""
echo "[3/3] Training DANN model..."
python scripts/train_dann_cpcgx.py \
    --features packed/features_cpcgx.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - Preprocessed: packed/cpcgx_pain.npz"
echo "  - Features: packed/features_cpcgx.npz"
echo "  - Reports: reports/"


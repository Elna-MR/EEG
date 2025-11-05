#!/bin/bash
# Script to process a new EEG pain dataset through the full pipeline

set -e

DATASET_NAME=${1:-"biovid"}
DATA_DIR=${2:-"data/${DATASET_NAME}"}

echo "=========================================="
echo "Processing ${DATASET_NAME} Dataset"
echo "=========================================="
echo ""

# Step 1: Preprocess
echo "[1/3] Preprocessing..."
python src/preprocess_multidataset.py \
    --dataset ${DATASET_NAME} \
    --data-dir ${DATA_DIR} \
    --output ${DATASET_NAME}_pain.npz

# Step 2: Extract Features
echo ""
echo "[2/3] Extracting Riemannian features..."
python src/extract_features.py \
    --npz packed/${DATASET_NAME}_pain.npz \
    --out packed/features_${DATASET_NAME}.npz

# Step 3: Train DANN
echo ""
echo "[3/3] Training DANN..."
python src/train_dann.py \
    --features packed/features_${DATASET_NAME}.npz \
    --epochs 30 \
    --batch 128 \
    --lr 1e-3 \
    --report-dir reports/${DATASET_NAME}

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Results saved to: reports/${DATASET_NAME}/"


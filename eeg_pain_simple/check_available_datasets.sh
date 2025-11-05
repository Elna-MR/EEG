#!/bin/bash
echo "Checking for available datasets..."
echo ""
for dataset in biovid painmonit seed_pain; do
    if [ -d "data/${dataset}" ]; then
        echo "✅ Found ${dataset} in data/${dataset}/"
        find data/${dataset} -name "*.bdf" -o -name "*.edf" -o -name "*.mat" | head -3
    else
        echo "❌ ${dataset} not found in data/${dataset}/"
    fi
    echo ""
done

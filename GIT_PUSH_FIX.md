# Git Push Fix - Large Files Removed

## Problem
Git push failed with HTTP 400 error trying to push 1.93 GB of large data files.

## Solution Applied

### 1. Removed Large Files from Tracking
- Removed 149 large files (.eeg, .gdf, .npz, .bdf)
- Files remain on disk but won't be pushed to git

### 2. Updated .gitignore
Added patterns to ignore:
- `*.npz` - Preprocessed data
- `*.bdf` - Raw BDF files
- `*.eeg` - Raw EEG data
- `*.vhdr`, `*.vmrk` - BrainVision files
- `*.gdf` - GDF files
- `eeg_pain_simple/datasets/*/packed/` - All packed folders
- `eeg_pain_simple/datasets/*/data/**/*.bdf` - Raw data files

### 3. Increased Git Buffer
```bash
git config http.postBuffer 524288000
git config http.maxRequestBuffer 100M
```

## Next Steps

```bash
# Commit the changes
git add .gitignore
git commit -m "Remove large data files from tracking"

# Try pushing
git push
```

## If Push Still Fails

The files may still be in git history. Use one of these:

### Option 1: Force Push (if you're the only contributor)
```bash
git push --force
```

### Option 2: Remove from History (recommended)
```bash
# Install BFG Repo-Cleaner (easier than filter-branch)
# Or use git filter-branch:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch eeg_pain_simple/datasets/*/data/**/*.eeg" \
  --prune-empty --tag-name-filter cat -- --all
```

### Option 3: Use Git LFS for Large Files
```bash
git lfs install
git lfs track "*.npz"
git lfs track "*.eeg"
git lfs track "*.bdf"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## Files Removed
- 144 .eeg files (cpCGX raw data)
- 5 .gdf files (pain-eeg-figures)
- All .npz files (preprocessed data)
- All .bdf files (ds005284 raw data)

**Note:** These files remain on your local disk - they're just not tracked by git anymore.


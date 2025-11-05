# SEED-IV Download Status

## Testing Results

✅ **Download Scripts Created**:
- `src/download_seed_iv.py` - Direct URL download attempt
- `src/download_seed_iv_torcheeg.py` - TorchEEG library approach
- `src/prepare_seed_iv.py` - Dataset preparation and conversion

## Test Results

### 1. Direct Download Attempt
- ❌ **Failed**: SEED-IV requires manual registration and authentication
- URLs tested: Not publicly accessible
- **Reason**: Dataset requires license agreement and email approval

### 2. TorchEEG Library
- ❌ **Failed**: Installation dependency issues (requires gfortran)
- **Reason**: TorchEEG depends on scipy which needs Fortran compiler

### 3. Current Status
- ❌ **Dataset not found**: `data/seed_iv/` directory is empty
- ✅ **Scripts ready**: All preparation scripts are ready to use

## Manual Download Required

SEED-IV **cannot be downloaded automatically** due to:
1. License agreement requirement
2. Manual approval process
3. Authentication needed

### Steps to Download SEED-IV:

1. **Visit the official page**:
   - http://bcmi.sjtu.edu.cn/~seed/seed-iv.html

2. **Download license agreement**:
   - https://bcmi.sjtu.edu.cn/~seed/resource/license/SEED-IV%20license.pdf

3. **Fill out and sign** the agreement with your information

4. **Email signed PDF** to:
   - weilonglive@gmail.com

5. **Wait for approval** - you'll receive:
   - Download link
   - Password for access

6. **Download and extract** to:
   ```
   eeg_pain_simple/data/seed_iv/
   ```

7. **Check structure**:
   ```bash
   python src/prepare_seed_iv.py --data-dir data/seed_iv/
   ```

8. **Convert MAT to BDF** (if needed):
   ```bash
   python src/prepare_seed_iv.py --data-dir data/seed_iv/ --convert
   ```

9. **Run preprocessing**:
   ```bash
   python src/preprocess_multidataset.py --dataset seed_pain --data-dir data/seed_iv/
   ```

## Alternative: Use BioVid or PainMonit

Since SEED-IV requires manual registration, consider:

### BioVid (Recommended - 87 subjects)
- Larger dataset
- May be more accessible
- Thermal pain stimulation

### PainMonit (56 subjects)
- Electrical stimulation
- May require registration but potentially faster

## Next Steps

1. **If you have SEED-IV access**: Run `prepare_seed_iv.py` to check structure
2. **If downloading**: Follow manual steps above
3. **Alternative**: Try BioVid or PainMonit datasets which may be easier to access


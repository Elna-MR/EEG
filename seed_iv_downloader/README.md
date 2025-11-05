# SEED / SEED-IV Local Downloader & Packager

This repo lets **you** download SEED / SEED-IV EEG archives (after you obtain access links)
and repackage them into a clean structure ready for ML pipelines.

> Note: This script cannot bypass access controls. Paste your **authorized** download links
> (OneDrive/Baidu/HTTP) into `links.txt` and run locally.

## Quick start

1) Create and activate a Python env (3.9+ recommended), then install deps:
```
pip install -r requirements.txt
```

2) Put your download URLs into `links.txt` (one per line). For OneDrive links,
add `?download=1` if needed to force direct download.

3) Download all files to `raw/`:
```
python bin/downloader.py --links links.txt --out raw
```

4) Unpack archives and normalize into `out/SEED_IV/` with a manifest:
```
python bin/package.py --raw raw --out out
```

5) Create a single distributable zip:
```
python bin/package.py --raw raw --out out --zip seed_iv_package.zip
```

You’ll end up with:
```
out/
  SEED_IV/
    subject_01/
      session_01/
        trial_001.mat
        ...
    subject_02/...
  manifest.csv
seed_iv_package.zip  # (optional final package)
```

### Manifest columns
- `relpath` : path relative to `out/`
- `subject` : subject id inferred from folder/file name
- `session` : session id (if detectable)
- `trial`   : trial id (if detectable)
- `md5`     : checksum of the file

## Tips
- If links require cookies/auth, download the archives with your browser into `raw/` and
  run **only** the packaging step (#4–5).
- If filenames don't encode subject/session/trial, the packager will still include them in
  `manifest.csv` but with `unknown` fields. You can edit the CSV later if you have metadata.

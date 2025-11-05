#!/usr/bin/env python3
import argparse, os, re, zipfile, hashlib, csv, shutil
from pathlib import Path

def md5sum(path, buf=1024*1024):
    m = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            m.update(chunk)
    return m.hexdigest()

def safe_extract(archive_path, extract_to):
    p = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    if p.suffix.lower() in [".zip"]:
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(extract_to / p.stem)
        return extract_to / p.stem
    # fall back: if it's not an archive we know, just copy it
    dst = extract_to / p.name
    shutil.copy2(p, dst)
    return dst

def infer_tags(name):
    s = re.search(r"(S|subj|subject)[ _-]?(\d+)", name, re.IGNORECASE)
    subject = f"{int(s.group(2)):02d}" if s else "unknown"
    s2 = re.search(r"(Sess|session)[ _-]?(\d+)", name, re.IGNORECASE)
    session = f"{int(s2.group(2)):02d}" if s2 else "01"
    s3 = re.search(r"(T|trial)[ _-]?(\d+)", name, re.IGNORECASE)
    trial  = f"{int(s3.group(2)):03d}" if s3 else "unknown"
    return subject, session, trial

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir containing downloaded archives/files")
    ap.add_argument("--out", required=True, help="output dir for normalized package")
    ap.add_argument("--zip", default="", help="optional: name of final zip to create at project root")
    args = ap.parse_args()

    raw = Path(args.raw); out = Path(args.out) / "SEED_IV"
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    unpack_root = Path(args.out) / "_unpacked"
    unpack_root.mkdir(parents=True, exist_ok=True)

    for p in raw.iterdir():
        if p.is_file():
            try:
                extracted = safe_extract(p, unpack_root)
            except Exception as e:
                print(f"[WARN] could not extract {p}: {e}")
        else:
            shutil.copytree(p, unpack_root / p.name, dirs_exist_ok=True)

    for f in unpack_root.rglob("*"):
        if not f.is_file(): 
            continue
        if f.suffix.lower() not in [".mat", ".edf", ".set", ".fif"]:
            continue
        subject, session, trial = infer_tags(str(f))
        target = out / f"subject_{subject}" / f"session_{session}"
        target.mkdir(parents=True, exist_ok=True)
        newname = f"trial_{trial}{f.suffix.lower()}"
        dst = target / newname
        shutil.copy2(f, dst)

        rel = Path("SEED_IV") / f"subject_{subject}" / f"session_{session}" / newname
        manifest_rows.append({
            "relpath": str(rel),
            "subject": subject,
            "session": session,
            "trial": trial,
            "md5": md5sum(dst)
        })

    manifest_path = Path(args.out) / "manifest.csv"
    import csv
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relpath","subject","session","trial","md5"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote manifest: {manifest_path} ({len(manifest_rows)} files)")

    if args.zip:
        zpath = Path(args.zip)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in (Path(args.out) / "SEED_IV").rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(Path(args.out)))
            z.write(manifest_path, manifest_path.relative_to(Path(args.out)))
        print(f"Created package: {zpath}")

if __name__ == "__main__":
    main()

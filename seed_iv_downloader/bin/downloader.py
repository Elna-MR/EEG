#!/usr/bin/env python3
import argparse, os, sys, time
import requests
from tqdm import tqdm
from urllib.parse import urlparse

def download(url, outdir):
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(urlparse(url).path) or f"file_{int(time.time()*1000)}"
    dest = os.path.join(outdir, name)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", required=True, help="text file with URLs, one per line")
    ap.add_argument("--out", required=True, help="directory to write downloaded files")
    args = ap.parse_args()

    with open(args.links, "r") as fh:
        for line in fh:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            try:
                dest = download(url, args.out)
                print(f"Downloaded: {dest}")
            except Exception as e:
                print(f"[WARN] Failed: {url} -> {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

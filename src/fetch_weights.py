"""Robust downloader for torchvision pretrained weights (resume + retry + hash check).

The network here truncates large downloads, so urllib's default retrieval fails the
hash check. This script resumes partial files with HTTP Range requests and verifies
the hash prefix before placing the file in the torch hub cache.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")

# filename on the pytorch CDN -> sha256 prefix (same check torchvision does)
WEIGHTS = {
    "resnet18-f37072fd.pth": "f37072fd",
    "resnet50-11ad3fa6.pth": "11ad3fa6",
    "efficientnet_b0_rwightman-7f5810bc.pth": "7f5810bc",
    "convnext_tiny-983f1562.pth": "983f1562",
}

URLS = {
    "resnet18-f37072fd.pth": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet50-11ad3fa6.pth": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "efficientnet_b0_rwightman-7f5810bc.pth": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "convnext_tiny-983f1562.pth": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, retries: int = 15) -> str:
    os.makedirs(CACHE, exist_ok=True)
    dest = os.path.join(CACHE, name)
    url = URLS[name]
    expected = WEIGHTS[name]
    for attempt in range(1, retries + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = None
                if resp.status == 206:
                    cr = resp.headers.get("Content-Range", "")
                    if "/" in cr:
                        total = int(cr.rsplit("/", 1)[1])
                else:
                    total = int(resp.headers.get("Content-Length") or 0) + have
                    if have and resp.status == 200:
                        have = 0  # server ignored Range, restart from zero
                mode = "ab" if have else "wb"
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                final = os.path.getsize(dest)
                print(f"  {name}: {final // 1024 // 1024} MB "
                      f"(total {None if total is None else total // 1024 // 1024} MB, attempt {attempt})",
                      flush=True)
                if total and final < total:
                    continue
        except urllib.error.HTTPError as e:
            if e.code == 416:  # already complete
                pass
            else:
                print(f"  HTTP {e.code} on attempt {attempt}, retrying...", flush=True)
                time.sleep(2)
                continue
        except Exception as e:  # noqa: BLE001
            print(f"  {type(e).__name__} on attempt {attempt}, retrying...", flush=True)
            time.sleep(2)
            continue

        digest = sha256(dest)
        if digest.startswith(expected):
            print(f"  OK {name} sha256={digest[:12]}", flush=True)
            return dest
        print(f"  hash mismatch {digest[:12]} != {expected}, restarting", flush=True)
        os.remove(dest)

    raise RuntimeError(f"could not download {name}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(URLS)
    for n in names:
        print(f"fetching {n}", flush=True)
        fetch(n)
    print("all weights ready in", CACHE)

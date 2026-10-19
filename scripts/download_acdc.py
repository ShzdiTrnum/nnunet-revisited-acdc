#!/usr/bin/env python3
"""Download ACDC from the official HumanHeart Project (CREATIS) Girder server.

- Resumable: files already downloaded completely are skipped; an interrupted file restarts.
- Downloads only what the reproduction needs: the ED/ES frames + ground truth of the
  100 training patients (+ Info.cfg). Skips the large 4D cine volumes and, by default,
  the test set (the paper reports 5-fold cross-validation on the 200 training cases).
- Standard library only.

Usage:
  python download_acdc.py --out ~/nnunet_repro/downloads/ACDC            # download
  python download_acdc.py --out ~/nnunet_repro/downloads/ACDC --list-only  # just show what is there
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
COLLECTION = "637218c173e9f0047faa00fb"   # ACDC collection linked from the official challenge page
RETRIES = 8


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def get_json(path):
    url = API + path
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(f"Access denied (HTTP {e.code}) for {url}\n-> send this message to Claude.")
            err = e
        except Exception as e:  # network hiccup: retry with backoff
            err = e
        wait = min(120, 2 ** attempt)
        log(f"  retry {attempt + 1}/{RETRIES} after: {err} (waiting {wait}s)")
        time.sleep(wait)
    sys.exit(f"Giving up on {url}")


def walk(parent_type, parent_id, rel=""):
    """Yield (relative_path, file_id, size) for every file under a Girder collection/folder."""
    for folder in get_json(f"/folder?parentType={parent_type}&parentId={parent_id}&limit=0"):
        sub = os.path.join(rel, folder["name"])
        for item in get_json(f"/item?folderId={folder['_id']}&limit=0"):
            for f in get_json(f"/item/{item['_id']}/files?limit=0"):
                yield os.path.join(sub, f["name"]), f["_id"], int(f["size"])
        yield from walk("folder", folder["_id"], sub)


def wanted(rel_path, with_testing):
    parts = rel_path.replace("\\", "/").split("/")
    name = parts[-1]
    if not (name.endswith(".nii.gz") or name == "Info.cfg"):
        return False
    if "_4d" in name:                      # 4D cine volume, not used by nnU-Net / the paper
        return False
    if "testing" in parts and not with_testing:
        return False
    return True


def download(file_id, dest, size):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(f"{API}/file/{file_id}/download", timeout=300) as r, open(tmp, "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            got = os.path.getsize(tmp)
            if got != size:
                raise IOError(f"size mismatch: got {got}, expected {size}")
            os.replace(tmp, dest)
            return "ok"
        except Exception as e:
            wait = min(120, 2 ** attempt)
            log(f"  retry {attempt + 1}/{RETRIES} for {os.path.basename(dest)}: {e} (waiting {wait}s)")
            time.sleep(wait)
    raise SystemExit(f"Giving up on {dest}. Re-run the same command later to resume.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--with-testing", action="store_true", help="also download the 50 test patients")
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    log("Listing files on the ACDC server (takes a few minutes: one request per patient folder)...")
    all_files = list(walk("collection", COLLECTION))
    todo = [f for f in all_files if wanted(f[0], args.with_testing)]
    skipped = [f for f in all_files if not wanted(f[0], args.with_testing)]
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump([{"path": p, "id": i, "size": s} for p, i, s in all_files], fh, indent=1)

    gb = lambda n: f"{n / 1024 ** 3:.2f} GB"
    log(f"Server has {len(all_files)} files ({gb(sum(f[2] for f in all_files))}).")
    log(f"Will download {len(todo)} files ({gb(sum(f[2] for f in todo))}); skipping {len(skipped)} "
        f"({gb(sum(f[2] for f in skipped))}: 4D volumes{'' if args.with_testing else ', test set'}, other files).")
    if not todo:
        log("No matching .nii.gz files found. Other files on the server:")
        for p, _, s in skipped[:30]:
            print(f"   {p}  ({s / 1024 ** 2:.1f} MB)")
        sys.exit("-> send this output to Claude.")
    if args.list_only:
        for p, _, s in todo[:10]:
            print(f"   {p}  ({s / 1024 ** 2:.1f} MB)")
        print("   ...")
        return

    done_bytes, t0 = 0, time.time()
    for n, (rel, fid, size) in enumerate(todo, 1):
        status = download(fid, os.path.join(out, rel), size)
        done_bytes += size if status == "ok" else 0
        if status == "ok" or n % 50 == 0 or n == len(todo):
            rate = done_bytes / max(1.0, time.time() - t0) / 1024
            log(f"[{n}/{len(todo)}] {status:4s} {rel}  ({rate:.0f} kB/s this run)")

    # The nnU-Net converter expects <root>/training and <root>/testing side by side.
    roots = sorted({os.path.normpath(os.path.join(out, ("/" + p.replace("\\", "/")).split("/training/")[0].lstrip("/")))
                    for p, _, _ in todo if "/training/" in "/" + p.replace("\\", "/")})
    for root in roots:
        os.makedirs(os.path.join(root, "testing"), exist_ok=True)
        n_pat = len([d for d in os.listdir(os.path.join(root, "training")) if d.startswith("patient")])
        log(f"ACDC root for conversion: {root}  ({n_pat} training patients)")
    log("DOWNLOAD COMPLETE")


if __name__ == "__main__":
    main()

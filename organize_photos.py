#!/usr/bin/env python3
"""
Organize images by EXIF/filename date into:
  <DEST>/<YYYY>/<MM>/<MAKE_MODEL>/<DD>/  or  <DEST>/undated/<MAKE_MODEL>/

Rules:
- MAKE_MODEL is one component (default separator "_"). If BOTH make & model unknown ⇒ "UNKNOWN".
- Date priority: EXIF DateTimeOriginal -> DateTime -> filename patterns.
- If no date found ⇒ goes to <DEST>/undated/<MAKE_MODEL>/.
- Ignores small files by --min-bytes (default 10_000).
- Deduplicates by SHA-256 (across names), with optional pre-scan of DEST.
- CSV logging with date_source.

Usage:
  python organize_photos.py --src /path/to/source --dest /path/to/base [--move] [--dry-run]
                            [--log photos_log.csv] [--min-bytes 10000] [--scan-dest]
                            [--mm-sep _]
Requires:
  pip install pillow pillow-heif
"""

import argparse
import csv
import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

# Optional HEIC/HEIF support
try:
    from pillow_heif import register_heif  # type: ignore
    register_heif()
except Exception:
    pass

from PIL import Image, UnidentifiedImageError

IMG_EXTS = {
    ".jpg", ".jpeg", ".tif", ".tiff", ".png", ".gif", ".heic", ".heif", ".webp"
}

EXIF_DATETIME_ORIGINAL = 36867  # DateTimeOriginal
EXIF_DATETIME = 306             # DateTime
EXIF_MAKE = 271                 # Make
EXIF_MODEL = 272                # Model

CSV_FIELDS = [
    "ts", "action", "reason", "date_source",
    "src_path", "dst_path", "bytes", "sha256",
    "exif_datetime", "exif_make", "exif_model"
]

def parse_args():
    p = argparse.ArgumentParser(description="Copy/move images into YYYY/MM/MAKE_MODEL/DD or undated/, with logging & dedup.")
    p.add_argument("--src", required=True, type=Path, help="Source folder (recursive).")
    p.add_argument("--dest", required=True, type=Path, help="Destination base folder.")
    p.add_argument("--move", action="store_true", help="Move instead of copy.")
    p.add_argument("--dry-run", action="store_true", help="Show actions, do not write.")
    p.add_argument("--log", type=Path, default=Path("photos_log.csv"), help="CSV log path.")
    p.add_argument("--min-bytes", type=int, default=10_000, help="Ignore files smaller than this many bytes.")
    p.add_argument("--scan-dest", action="store_true", help="Pre-index destination to avoid importing dupes already there.")
    p.add_argument("--mm-sep", type=str, default="_", help="Separator between make and model in the folder name.")
    return p.parse_args()

def sanitize_component(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", "_", s).lower()
    s = re.sub(r"[^a-z0-9._-]", "_", s)
    return s

def combine_make_model(make: str, model: str, sep: str) -> str:
    # Decide unknowns first (case-insensitive)
    mk_unknown = (not make) or make.strip().lower() in {"", "unknown", "na", "n/a", "null"}
    md_unknown = (not model) or model.strip().lower() in {"", "unknown", "na", "n/a", "null"}
    if mk_unknown and md_unknown:
        return "UNKNOWN"  # exact uppercase per request

    ms = sanitize_component(make or "")
    mdl = sanitize_component(model or "")
    # Fall back if one side empty
    if ms and mdl:
        combo = f"{ms}{sep}{mdl}"
    else:
        combo = ms or mdl or "UNKNOWN"
    # Collapse repeated separators and trim
    combo = re.sub(rf"{re.escape(sep)}+", sep, combo).strip(sep)
    return combo if combo else "UNKNOWN"

def get_exif(p: Path) -> dict:
    try:
        with Image.open(p) as im:
            exif = im.getexif()
            return dict(exif) if exif else {}
    except (UnidentifiedImageError, OSError):
        return {}

def parse_exif_dt(s: str) -> Optional[datetime]:
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

# ---- Filename date extraction ----
# Tries common patterns in camera/phone file names
FN_PATTERNS = [
    # IMG_20210305_142233, 20210305_142233, 2021-03-05, 2021_03_05, 2021.03.05
    (re.compile(r"(?P<y>20\d{2}|19\d{2})[-_.]?(?P<m>0[1-9]|1[0-2])[-_.]?(?P<d>0[1-9]|[12]\d|3[01])"), "%Y%m%d"),
    # 05-03-2021 or 05032021 (DDMMYYYY)
    (re.compile(r"(?P<d>0[1-9]|[12]\d|3[01])[-_.]?(?P<m>0[1-9]|1[0-2])[-_.]?(?P<y>20\d{2}|19\d{2})"), "%d%m%Y"),
    # 2021 03 05 anywhere with spaces
    (re.compile(r"(?P<y>20\d{2}|19\d{2})\s+(?P<m>0[1-9]|1[0-2])\s+(?P<d>0[1-9]|[12]\d|3[01])"), "%Y %m %d"),
    # YYYYMMDD as 8 consecutive digits
    (re.compile(r"(?P<y>20\d{2}|19\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])"), "%Y%m%d"),
    # DDMMYY or YYMMDD (ambiguous short years) – assume 20xx for YY<70 else 19xx
    (re.compile(r"(?P<d>[0-3]\d)[-_.]?(?P<m>0[1-9]|1[0-2])[-_.]?(?P<y>\d{2})(?!\d)"), "%d%m%y"),
    (re.compile(r"(?P<y>\d{2})[-_.]?(?P<m>0[1-9]|1[0-2])[-_.]?(?P<d>[0-3]\d)(?!\d)"), "%y%m%d"),
]

def parse_filename_date(name: str) -> Optional[datetime]:
    stem = Path(name).stem
    for rx, fmt in FN_PATTERNS:
        m = rx.search(stem)
        if not m:
            continue
        gd = m.groupdict()
        try:
            y = gd["y"]; mth = gd["m"]; d = gd["d"]
        except KeyError:
            continue
        # Normalize 2-digit year if present
        if len(y) == 2:
            yy = int(y)
            y = f"{2000+yy:04d}" if yy < 70 else f"{1900+yy:04d}"
        try:
            return datetime(int(y), int(mth), int(d))
        except ValueError:
            continue
    return None

def extract_meta_and_date_source(p: Path) -> Tuple[Optional[datetime], str, str, str, Optional[str]]:
    """
    Returns: (dt or None, date_source, make, model, raw_exif_dt_str)
    date_source in {"exif","filename","undated"}.
    """
    exif = get_exif(p)

    raw_dt = exif.get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
    dt_exif: Optional[datetime] = None
    if isinstance(raw_dt, bytes):
        raw_dt = raw_dt.decode(errors="ignore")
    if isinstance(raw_dt, str):
        dt_exif = parse_exif_dt(raw_dt)

    raw_make = exif.get(EXIF_MAKE)
    if isinstance(raw_make, bytes):
        raw_make = raw_make.decode(errors="ignore")
    make = raw_make if isinstance(raw_make, str) else ""

    raw_model = exif.get(EXIF_MODEL)
    if isinstance(raw_model, bytes):
        raw_model = raw_model.decode(errors="ignore")
    model = raw_model if isinstance(raw_model, str) else ""

    if dt_exif:
        return dt_exif, "exif", make or "Unknown", model or "Unknown", raw_dt

    # try filename date
    dt_fn = parse_filename_date(p.name)
    if dt_fn:
        return dt_fn, "filename", make or "Unknown", model or "Unknown", raw_dt if isinstance(raw_dt, str) else None

    # undated (do NOT fall back to mtime per request)
    return None, "undated", make or "Unknown", model or "Unknown", raw_dt if isinstance(raw_dt, str) else None

def sha256_file(p: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def ensure_log(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

def log_row(path: Path, row: Dict):
    row = {k: row.get(k, "") for k in CSV_FIELDS}
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

def unique_target(dst_dir: Path, src_name: str) -> Path:
    base = Path(src_name).stem
    ext = Path(src_name).suffix
    candidate = dst_dir / (base + ext)
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{base}_{i}{ext}"
        i += 1
    return candidate

def scan_hashes(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            try:
                h = sha256_file(p)
                index.setdefault(h, p)
            except Exception:
                pass
    return index

def main():
    a = parse_args()
    src = a.src.expanduser().resolve()
    dest = a.dest.expanduser().resolve()

    if not src.is_dir():
        raise SystemExit(f"Source is not a directory: {src}")

    ensure_log(a.log)

    # Hash index: previously-seen (optionally from dest) + this run
    hash_index: Dict[str, Path] = {}
    if a.scan_dest and dest.exists():
        print(f"[INFO] Pre-scanning destination for duplicates: {dest}")
        hash_index.update(scan_hashes(dest))
        print(f"[INFO] Found {len(hash_index)} existing hashed files in destination.")

    total = 0
    processed = 0

    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue

        size = p.stat().st_size
        if size < a.min_bytes:
            total += 1
            log_row(a.log, {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "SKIP",
                "reason": f"too_small<{a.min_bytes}",
                "date_source": "",
                "src_path": str(p),
                "dst_path": "",
                "bytes": size,
                "sha256": "",
                "exif_datetime": "",
                "exif_make": "",
                "exif_model": "",
            })
            continue

        total += 1

        # Metadata + date
        dt, date_source, make, model, raw_exif_dt = extract_meta_and_date_source(p)
        make_model = combine_make_model(make, model, a.mm_sep)

        # Hash for dedup
        try:
            h = sha256_file(p)
        except Exception as e:
            log_row(a.log, {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "ERROR",
                "reason": f"hash_failed:{e}",
                "date_source": date_source,
                "src_path": str(p),
                "dst_path": "",
                "bytes": size,
                "sha256": "",
                "exif_datetime": raw_exif_dt or "",
                "exif_make": make,
                "exif_model": model,
            })
            continue

        if h in hash_index:
            existing = hash_index[h]
            log_row(a.log, {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "SKIP_DUPLICATE",
                "reason": f"duplicate_of:{existing}",
                "date_source": date_source,
                "src_path": str(p),
                "dst_path": str(existing),
                "bytes": size,
                "sha256": h,
                "exif_datetime": raw_exif_dt or "",
                "exif_make": make,
                "exif_model": model,
            })
            print(f"SKIP_DUPLICATE: {p} == {existing}")
            continue

        # Destination path (dated vs undated)
        if dt is not None:
            year = f"{dt.year:04d}"
            month = f"{dt.month:02d}"
            day = f"{dt.day:02d}"
            dst_dir = dest / year / month / make_model / day
        else:
            dst_dir = dest / "undated" / make_model

        target = unique_target(dst_dir, p.name)
        action = "MOVE" if a.move else "COPY"

        if a.dry_run:
            print(f"[DRY] {action} {p} -> {target}")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                if a.move:
                    shutil.move(str(p), str(target))
                else:
                    shutil.copy2(str(p), str(target))
            except Exception as e:
                log_row(a.log, {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "action": "ERROR",
                    "reason": f"io:{e}",
                    "date_source": date_source,
                    "src_path": str(p),
                    "dst_path": str(target),
                    "bytes": size,
                    "sha256": h,
                    "exif_datetime": raw_exif_dt or "",
                    "exif_make": make,
                    "exif_model": model,
                })
                print(f"[ERROR] {p} -> {target}: {e}")
                continue

        hash_index[h] = target
        processed += 1
        log_row(a.log, {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "action": action if not a.dry_run else f"DRY_{action}",
            "reason": "",
            "date_source": date_source,
            "src_path": str(p),
            "dst_path": str(target),
            "bytes": size,
            "sha256": h,
            "exif_datetime": raw_exif_dt or "",
            "exif_make": make,
            "exif_model": model,
        })

    print(f"Done. {processed}/{total} image files processed (>= {a.min_bytes} bytes). Log: {a.log}")

if __name__ == "__main__":
    main()

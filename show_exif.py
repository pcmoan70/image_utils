#!/usr/bin/env python3
"""
dump_exif.py — dump EXIF for images (JPEG/TIFF/RAW, etc.)
Requires: pip install exifread

Examples:
  python dump_exif.py IMG_1234.JPG
  python dump_exif.py /path/to/folder -r
  python dump_exif.py *.jpg --json > exif.json
  python dump_exif.py photo.jpg --gps-decimal
"""
import argparse, os, sys, json, glob
from collections import OrderedDict
from typing import Dict, Any, List

def dms_to_decimal(dms, ref):
    # dms is a list of exifread.utils.Ratio (deg, min, sec)
    try:
        deg = float(dms[0])
        minutes = float(dms[1])
        secs = float(dms[2])
        sign = -1 if ref in ("S", "W") else 1
        return sign * (deg + minutes/60.0 + secs/3600.0)
    except Exception:
        return None

def read_exif(path: str) -> Dict[str, Any]:
    import exifread  # lazy import
    data = OrderedDict()
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=True, debug=False)
    # Convert values to printable strings (truncate long binaries)
    for k in sorted(tags.keys()):
        v = tags[k]
        s = str(v)
        if len(s) > 500:
            s = s[:500] + " ...[truncated]"
        data[k] = s
    # Optional: GPS as decimal if present
    lat = lon = None
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon_ref = tags.get("GPS GPSLongitudeRef")
    lat_vals = tags.get("GPS GPSLatitude")
    lon_vals = tags.get("GPS GPSLongitude")
    if lat_ref and lon_ref and lat_vals and lon_vals:
        lat = dms_to_decimal(list(lat_vals.values) if hasattr(lat_vals, "values") else lat_vals, str(lat_ref))
        lon = dms_to_decimal(list(lon_vals.values) if hasattr(lon_vals, "values") else lon_vals, str(lon_ref))
        if lat is not None and lon is not None:
            data["_GPS.LatitudeDecimal"] = lat
            data["_GPS.LongitudeDecimal"] = lon
    return data

def iter_inputs(inputs: List[str], recursive: bool) -> List[str]:
    files = []
    for p in inputs:
        if any(ch in p for ch in "*?[]"):
            files.extend(glob.glob(p, recursive=recursive))
        elif os.path.isdir(p):
            if recursive:
                for root, _, fnames in os.walk(p):
                    for fn in fnames:
                        files.append(os.path.join(root, fn))
            else:
                for fn in os.listdir(p):
                    files.append(os.path.join(p, fn))
        else:
            files.append(p)
    # Filter to likely image files (exifread handles many RAWs too)
    exts = {".jpg",".jpeg",".tif",".tiff",".png",".heic",".arw",".nef",".cr2",".cr3",".rw2",".orf",".raf",".dng"}
    files = [f for f in files if os.path.splitext(f)[1].lower() in exts and os.path.isfile(f)]
    return sorted(set(files))

def main():
    ap = argparse.ArgumentParser(description="Dump EXIF metadata from image files.")
    ap.add_argument("paths", nargs="+", help="Image file(s), folder(s), or globs")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into folders/globs")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args()

    files = iter_inputs(args.paths, args.recursive)
    if not files:
        print("No matching image files.", file=sys.stderr); sys.exit(1)

    if args.json:
        out = OrderedDict()
        for f in files:
            try:
                out[f] = read_exif(f)
            except Exception as e:
                out[f] = {"_error": str(e)}
        if args.pretty:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
        return

    # Human-readable text
    for f in files:
        print(f"\n=== {f} ===")
        try:
            ex = read_exif(f)
            if not ex:
                print("(no EXIF found)")
                continue
            for k, v in ex.items():
                print(f"{k}: {v}")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()

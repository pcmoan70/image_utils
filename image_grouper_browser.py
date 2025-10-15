#!/usr/bin/env python3
"""
Image Grouper & Fast Browser
---------------------------------
Browse a folder of images grouped by visual similarity. Quickly "store" (move)
keepers and delete rejects using single-key shortcuts. Designed for speed and
minimal dependencies.

New: Fuzzy grouping with a local CLIP model (no internet needed after first weights download).

Features
- Two grouping engines:
  1) **pHash** (fast perceptual hash, Hamming distance)
  2) **CLIP** (local AI embeddings, cosine similarity)
- Fast keyboard workflow over OpenCV window
- Two views: single-image and contact sheet of current group
- Actions: Store (move to keep folder), Delete (send to OS Trash), Skip, Undo
- Group and image navigation
- Persistent thumbnail cache in memory for responsiveness

Controls (active while the OpenCV window is focused)
  Help           : h
  Next image     : Right Arrow / l
  Prev image     : Left Arrow  / h
  Next group     : Down Arrow  / j
  Prev group     : Up Arrow    : k
  Store image    : s   (moves file to --store-dir)
  Delete image   : d   (Send to Trash via send2trash)
  Toggle view    : t   (single <-> contact sheet)
  Open in viewer : o   (open file with OS default)
  Undo last      : u   (undo last store/delete)
  Quit           : q or ESC

Usage
  python image_grouper_browser.py --root "/path/to/images" --mode clip --sim-thresh 0.87 --store-dir "./_store"

Options
  --root           Root folder with images (recursively scanned)
  --exts           Comma-separated extensions (default: jpg,jpeg,png,webp,bmp,tiff)
  --mode           Grouping engine: phash | clip  (default: phash)
  --hash-thresh    Hamming threshold for phash (default: 10)
  --sim-thresh     Cosine similarity threshold for CLIP (0..1, default: 0.87)
  --clip-model     open_clip model name (default: ViT-B-32)
  --clip-pretrained Pretrained tag (default: openai)
  --batch-size     Batch size for CLIP inference (default: 32)
  --min-group      Minimum images required to form a group (default: 1)
  --store-dir      Folder to move stored images (default: ./_store under root)
  --tile-width     Width of contact-sheet window in pixels (default: 1280)
  --max-rows       Max rows in contact sheet (default: 4)

Dependencies
  pip install pillow imagehash opencv-python send2trash torch torchvision open_clip_torch

Sharpness indicator (optional)
- Uses OpenCV Haar cascades to detect faces/eyes (fully local), then computes
  **variance of Laplacian** on eye regions. If no eyes detected, falls back to
  whole-image sharpness.
- Overlays a compact status badge on each preview: `EYES SHARP` / `SOFT`, with
  numeric scores.

Notes
- CLIP runs locally on CPU or GPU (CUDA if available). The first run will download weights.
- Deletes are safe (sent to Trash). You can restore from the OS Trash/Recycle Bin.
- "Undo" only undoes the most recent store (stack-based). For deletes, restore from Trash.
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2  # type: ignore
import numpy as np
from PIL import Image
import imagehash  # type: ignore
from send2trash import send2trash  # type: ignore

# Optional CLIP imports (lazy)
try:
    import torch  # type: ignore
    import open_clip  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    open_clip = None

# -------------------------- Utility ------------------------------------
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def is_image(path: str, exts: Optional[set] = None) -> bool:
    exts = exts or SUPPORTED
    return os.path.splitext(path)[1].lower() in exts

@dataclass
class Item:
    path: str
    phash: Optional[int] = None
    emb: Optional[np.ndarray] = None  # L2-normalized CLIP embedding

@dataclass
class Action:
    kind: str  # 'store' | 'delete'
    src: str
    dest: Optional[str] = None

@dataclass
class Group:
    rep_idx: int  # index into items list for representative
    indices: List[int]  # indices of items in this group

# ------------------------ Perceptual Hashing ----------------------------

def phash64(path: str) -> int:
    with Image.open(path) as im:
        im = im.convert("RGB")
        h = imagehash.phash(im)
    return int(str(h), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# ------------------------ CLIP Embeddings -------------------------------
class CLIPEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        if open_clip is None:
            raise RuntimeError("open_clip_torch is not installed. pip install open_clip_torch torch torchvision")
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        self.model.to(self.device)
        # use image tower only
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    @torch.no_grad()  # type: ignore
    def encode_paths(self, paths: List[str], batch_size: int = 32) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = []
            for p in batch:
                with Image.open(p) as im:
                    imgs.append(self.preprocess(im.convert("RGB")))
            x = torch.stack(imgs).to(self.device, dtype=self.dtype)
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.detach().cpu().float().numpy())
        return np.vstack(embs)

# ------------------------ Grouping --------------------------------------

def greedy_group_phash(items: List[Item], thresh: int, min_group: int) -> List[Group]:
    reps: List[int] = []  # store index of representative item
    groups: List[Group] = []
    for idx, it in enumerate(items):
        if it.phash is None:
            continue
        best_g = -1
        best_d = 1e9
        for gi, rep_idx in enumerate(reps):
            d = hamming(it.phash, items[rep_idx].phash)  # type: ignore
            if d < best_d:
                best_d = d
                best_g = gi
        if best_g >= 0 and best_d <= thresh:
            groups[best_g].indices.append(idx)
        else:
            reps.append(idx)
            groups.append(Group(rep_idx=idx, indices=[idx]))
    if min_group > 1:
        groups = [g for g in groups if len(g.indices) >= min_group]
    groups.sort(key=lambda g: -len(g.indices))
    return groups


def greedy_group_clip(items: List[Item], sim_thresh: float, min_group: int) -> List[Group]:
    reps: List[int] = []
    groups: List[Group] = []
    for idx, it in enumerate(items):
        if it.emb is None:
            continue
        best_g = -1
        best_s = -1.0
        v = it.emb
        for gi, rep_idx in enumerate(reps):
            s = float(np.dot(v, items[rep_idx].emb))  # type: ignore
            if s > best_s:
                best_s = s
                best_g = gi
        if best_g >= 0 and best_s >= sim_thresh:
            groups[best_g].indices.append(idx)
        else:
            reps.append(idx)
            groups.append(Group(rep_idx=idx, indices=[idx]))
    if min_group > 1:
        groups = [g for g in groups if len(g.indices) >= min_group]
    groups.sort(key=lambda g: -len(g.indices))
    return groups

# ------------------------ Rendering -------------------------------------

def load_thumb(path: str, max_side: int = 512) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        cv2.putText(img, "(unreadable)", (10, max_side // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return img
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def overlay_text(img: np.ndarray, text: str, margin: int = 8) -> np.ndarray:
    out = img.copy()
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = margin, margin + th
    box = out.copy()
    cv2.rectangle(box, (margin - 4, margin - 4), (margin + tw + 4, margin + th + 4), (0, 0, 0), -1)
    cv2.addWeighted(box, 0.4, out, 0.6, 0, out)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

# --------- Sharpness & Eye Detection ------------------------------------
class EyeSharpness:
    def __init__(self, eye_thresh: float = 120.0, global_thresh: float = 60.0):
        base = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(base, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(base, 'haarcascade_eye.xml'))
        self.eye_thresh = eye_thresh
        self.global_thresh = global_thresh
        self.cache: Dict[str, Dict] = {}

    def score_path(self, path: str) -> Dict:
        if path in self.cache:
            return self.cache[path]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            res = {"status":"unknown","global":0.0,"eyes":[],"used":"none"}
            self.cache[path] = res
            return res
        scale = 640.0 / max(img.shape)
        if scale < 1:
            small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), cv2.INTER_AREA)
        else:
            small = img
        faces = self.face_cascade.detectMultiScale(small, 1.2, 5)
        eyes_scores = []
        used = 'eyes'
        if len(faces) == 0:
            eyes = self.eye_cascade.detectMultiScale(small, 1.2, 8)
            for (ex,ey,ew,eh) in eyes:
                roi = small[ey:ey+eh, ex:ex+ew]
                eyes_scores.append(self._lapvar(roi))
        else:
            (x,y,w,h) = max(faces, key=lambda r: r[2]*r[3])
            face = small[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face, 1.15, 8)
            for (ex,ey,ew,eh) in eyes:
                roi = face[ey:ey+eh, ex:ex+ew]
                eyes_scores.append(self._lapvar(roi))
        if not eyes_scores:
            used = 'global'
            gscore = self._lapvar(small)
            status = 'sharp' if gscore >= self.global_thresh else 'soft'
            res = {"status":status,"global":float(gscore),"eyes":[],"used":used}
            self.cache[path] = res
            return res
        es = float(np.median(eyes_scores))
        status = 'sharp' if es >= self.eye_thresh else 'soft'
        res = {"status":status,"global":float(self._lapvar(small)),"eyes":[float(s) for s in eyes_scores],"used":used}
        self.cache[path] = res
        return res

    @staticmethod
    def _lapvar(gray: np.ndarray) -> float:
        v = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(v)


def make_contact_sheet(paths: List[str], thumbs: Dict[str, np.ndarray], tile_width: int = 1280, max_rows: int = 4) -> np.ndarray:
    n = len(paths)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    rows = min(rows, max_rows)
    cols = int(np.ceil(n / rows))
    pad = 8
    cell_w = (tile_width - (cols + 1) * pad) // cols
    cell_h = cell_w
    canvas_h = rows * cell_h + (rows + 1) * pad
    canvas = np.zeros((canvas_h, tile_width, 3), dtype=np.uint8)
    y = pad
    i = 0
    for r in range(rows):
        x = pad
        for c in range(cols):
            if i >= n: break
            p = paths[i]
            th = thumbs[p]
            th = fit_into(th, (cell_w, cell_h))
            th = add_border(th, 2, (50, 50, 50))
            th = overlay_text(th, f"{i+1}: {os.path.basename(p)}")
            h, w = th.shape[:2]
            canvas[y:y+h, x:x+w] = th
            x += cell_w + pad
            i += 1
        y += cell_h + pad
    return canvas


def fit_into(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    W, H = size
    h, w = img.shape[:2]
    scale = min(W / w, H / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    y = (H - new_h) // 2
    x = (W - new_w) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas


def add_border(img: np.ndarray, thickness: int, color: Tuple[int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros((h + 2*thickness, w + 2*thickness, 3), dtype=np.uint8)
    out[:, :] = color
    out[thickness:thickness+h, thickness:thickness+w] = img
    return out

# ------------------------ Browser ---------------------------------------
class Browser:
    def __init__(self, paths: List[str], groups: List[Group], store_dir: str, tile_width: int, max_rows: int, sharp: Optional[EyeSharpness] = None):
        self.paths = paths  # canonical order aligned with items
        self.groups = groups
        self.gi = 0
        self.ii = 0
        self.view = 'single'
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.cache: Dict[str, np.ndarray] = {}
        self.actions: List[Action] = []
        self.tile_width = tile_width
        self.max_rows = max_rows
        self.sharp = sharp
        cv2.namedWindow("Image Grouper", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image Grouper", tile_width, int(tile_width * 0.75))

    def current_group(self) -> Group:
        return self.groups[self.gi]

    def current_index(self) -> int:
        return self.current_group().indices[self.ii]

    def ensure_thumb(self, path: str) -> np.ndarray:
        if path not in self.cache:
            self.cache[path] = load_thumb(path)
        return self.cache[path]

    def open_current(self) -> None:
        path = self.paths[self.current_index()]
        if sys.platform.startswith('darwin'):
            subprocess.call(['open', path])
        elif os.name == 'nt':
            os.startfile(path)  # type: ignore
        else:
            subprocess.call(['xdg-open', path])

    def render(self) -> None:
        g = self.current_group()
        if self.view == 'grid':
            paths = [self.paths[i] for i in g.indices]
            for p in paths:
                self.ensure_thumb(p)
            sheet = make_contact_sheet(paths, self.cache, self.tile_width, self.max_rows)
            header = f"Group {self.gi+1}/{len(self.groups)}  |  {len(g.indices)} images  |  t:toggle view  s:store  d:delete  arrows:nav  h:help"
            sheet = overlay_text(sheet, header)
            cv2.imshow("Image Grouper", sheet)
        else:
            idx = self.current_index()
            p = self.paths[idx]
            th = self.ensure_thumb(p)
            label = f"Group {self.gi+1}/{len(self.groups)}  Img {self.ii+1}/{len(g.indices)}  [{os.path.basename(p)}]  s:store d:delete t:toggle arrows:nav h:help"
            img = overlay_text(th, label)
            if self.sharp is not None:
                info = self.sharp.score_path(p)
                eye = (np.median(info['eyes']) if info['eyes'] else 0.0)
                badge = f"EYES: {eye:.0f}  GLOBAL: {info['global']:.0f}  {('EYES' if info['used']=='eyes' else 'GLOBAL')} -> {'SHARP' if info['status']=='sharp' else 'SOFT'}"
                img = overlay_text(img, badge, margin=40)
            cv2.imshow("Image Grouper", img)

    def next_img(self) -> None:
        g = self.current_group()
        self.ii = (self.ii + 1) % len(g.indices)

    def prev_img(self) -> None:
        g = self.current_group()
        self.ii = (self.ii - 1) % len(g.indices)

    def next_group(self) -> None:
        self.gi = (self.gi + 1) % len(self.groups)
        self.ii = 0

    def prev_group(self) -> None:
        self.gi = (self.gi - 1) % len(self.groups)
        self.ii = 0

    def toggle_view(self) -> None:
        self.view = 'grid' if self.view == 'single' else 'single'

    def store_current(self) -> None:
        idx = self.current_index()
        src = self.paths[idx]
        base = os.path.basename(src)
        dest = unique_path(os.path.join(self.store_dir, base))
        try:
            os.replace(src, dest)
            self.actions.append(Action('store', src=dest))
            self.remove_index_from_groups(idx)
        except Exception as e:
            print(f"Store failed: {e}")

    def delete_current(self) -> None:
        idx = self.current_index()
        src = self.paths[idx]
        try:
            send2trash(src)
            self.actions.append(Action('delete', src=src))
            self.remove_index_from_groups(idx)
        except Exception as e:
            print(f"Delete failed: {e}")

    def remove_index_from_groups(self, idx: int) -> None:
        g = self.current_group()
        try:
            pos = g.indices.index(idx)
            del g.indices[pos]
        except ValueError:
            return
        if not g.indices:
            del self.groups[self.gi]
            if not self.groups:
                cv2.destroyAllWindows()
                sys.exit(0)
            self.gi %= len(self.groups)
            self.ii = 0
        else:
            self.ii %= len(g.indices)

    def undo(self) -> None:
        if not self.actions:
            print("Nothing to undo")
            return
        act = self.actions.pop()
        if act.kind == 'store':
            stored_path = act.src
            try:
                parent = os.path.dirname(os.path.dirname(stored_path))
                base = os.path.basename(stored_path)
                dest = unique_path(os.path.join(parent, base))
                os.replace(stored_path, dest)
                print(f"Restored to {dest}")
            except Exception as e:
                print(f"Undo store failed: {e}")
        elif act.kind == 'delete':
            print("Undo delete is not supported (item sent to OS Trash). Restore via your Trash/Recycle Bin.")


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while True:
        p = f"{root}_{i}{ext}"
        if not os.path.exists(p):
            return p
        i += 1

# ------------------------ Scanning & Main -------------------------------

def scan_images(root: str, exts: set) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if is_image(p, exts):
                paths.append(p)
    paths.sort()
    return paths


def build_items_phash(paths: List[str]) -> List[Item]:
    items: List[Item] = []
    for p in paths:
        try:
            h = phash64(p)
            items.append(Item(path=p, phash=h))
        except Exception as e:
            print(f"Hash failed for {p}: {e}")
    return items


def build_items_clip(paths: List[str], model_name: str, pretrained: str, batch_size: int) -> List[Item]:
    embedder = CLIPEmbedder(model_name, pretrained)
    embs = embedder.encode_paths(paths, batch_size=batch_size)
    items: List[Item] = []
    for p, e in zip(paths, embs):
        items.append(Item(path=p, emb=e.astype(np.float32)))
    return items


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Browse grouped similar images quickly.")
    ap.add_argument('--root', required=True, help='Root folder of images')
    ap.add_argument('--exts', default='jpg,jpeg,png,webp,bmp,tif,tiff', help='Comma-separated extensions')
    ap.add_argument('--mode', choices=['phash', 'clip'], default='phash', help='Grouping engine')
    ap.add_argument('--hash-thresh', type=int, default=10, help='Hamming distance threshold for phash')
    ap.add_argument('--sim-thresh', type=float, default=0.87, help='Cosine similarity threshold for CLIP (0..1)')
    ap.add_argument('--clip-model', default='ViT-B-32', help='open_clip model name')
    ap.add_argument('--clip-pretrained', default='openai', help='open_clip pretrained tag')
    ap.add_argument('--batch-size', type=int, default=32, help='Batch size for CLIP inference')
    ap.add_argument('--min-group', type=int, default=1, help='Minimum group size to keep')
    ap.add_argument('--store-dir', default=None, help='Where to move stored images (default: ROOT/_store)')
    ap.add_argument('--tile-width', type=int, default=1280, help='Contact sheet width in pixels')
    ap.add_argument('--max-rows', type=int, default=4, help='Max rows in contact sheet')
    ap.add_argument('--sharp-indicator', action='store_true', help='Compute and overlay sharpness/eyes indicator')
    ap.add_argument('--eye-thresh', type=float, default=120.0, help='Threshold for eye-region Laplacian variance')
    ap.add_argument('--global-thresh', type=float, default=60.0, help='Threshold for image-wide Laplacian variance')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root)
    exts = {'.' + e.strip().lower().lstrip('.') for e in args.exts.split(',')}
    store_dir = args.store_dir or os.path.join(root, '_store')

    print("Scanning images...")
    paths = scan_images(root, exts)
    if not paths:
        print("No images found.")
        return
    print(f"Found {len(paths)} images.")

    if args.mode == 'phash':
        print("Computing perceptual hashes...")
        items = build_items_phash(paths)
        print(f"Hashed {len(items)} images. Grouping (Hamming <= {args.hash_thresh})...")
        groups = greedy_group_phash(items, thresh=args.hash_thresh, min_group=args.min_group)
    else:
        if open_clip is None:
            print("open_clip_torch is required for --mode clip. Install dependencies and retry.")
            return
        print(f"Loading CLIP ({args.clip_model}, {args.clip_pretrained}) and embedding images...")
        items = build_items_clip(paths, model_name=args.clip_model, pretrained=args.clip_pretrained, batch_size=args.batch_size)
        print(f"Embedded {len(items)} images. Grouping (cosine >= {args.sim_thresh})...")
        groups = greedy_group_clip(items, sim_thresh=args.sim_thresh, min_group=args.min_group)

    if not groups:
        print("No groups formed. Adjust thresholds or mode.")
        return
    print(f"Formed {len(groups)} groups. Launching browser...")

    sharp = EyeSharpness(args.eye_thresh, args.global_thresh) if args.sharp_indicator else None
    br = Browser(paths, groups, store_dir=store_dir, tile_width=args.tile_width, max_rows=args.max_rows, sharp=sharp)
    br.render()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 255:
            continue
        if key in (ord('q'), 27):
            break
        elif key in (ord('h'), ):
            print(__doc__)
        elif key in (ord('t'), ):
            br.toggle_view()
        elif key in (ord('s'), ):
            br.store_current()
        elif key in (ord('d'), ):
            br.delete_current()
        elif key in (ord('u'), ):
            br.undo()
        elif key in (ord('o'), ):
            br.open_current()
        elif key in (81, ord('h')):  # left or vim h
            br.prev_img()
        elif key in (83, ord('l')):  # right or vim l
            br.next_img()
        elif key in (84, ord('j')):  # down or vim j
            br.next_group()
        elif key in (82, ord('k')):  # up or vim k
            br.prev_group()
        br.render()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

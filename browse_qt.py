#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Grouper Browser (PyQt5)
Patched to:
- Remove PIL.ImageQt dependency (works on Python 3.13 / Pillow builds lacking ImageQt)
- Correct PIL -> QImage conversion with deep copy (proper stride, formats)
- Use QThread correctly (no GUI blocking), queued signals, abort() support
- Throttle re-rendering on mouse move / wheel (prevents freezes)
- Reuse thumbnail widgets + simple in-memory cache
"""
import os
import sys
from datetime import datetime

from PyQt5.QtCore import Qt, QObject, QThread, QSize, pyqtSignal, QTimer, QEvent
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QPushButton,
    QScrollArea,
)

from PIL import Image
import piexif

# ----------------------------
# Utilities
# ----------------------------

def is_image(path: str) -> bool:
    b = os.path.basename(path)
    if b.startswith('._'):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def get_image_paths(folder: str):
    paths = []
    stack = [folder]
    while stack:
        root = stack.pop()
        try:
            with os.scandir(root) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False) and is_image(e.path):
                        paths.append(e.path)
        except PermissionError:
            pass
    return sorted(paths)


def get_exif_datetime(path: str):
    try:
        exif_dict = piexif.load(path)
        dt_bytes = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
        if dt_bytes:
            s = dt_bytes.decode() if isinstance(dt_bytes, (bytes, bytearray)) else dt_bytes
            return datetime.strptime(s, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    return None


def group_images_by_time(image_paths, max_gap_seconds=2):
    if not image_paths:
        return [], []
    dt_list = [get_exif_datetime(p) for p in image_paths]
    items = sorted(zip(image_paths, dt_list), key=lambda x: (x[1] or datetime.min, x[0]))
    groups, cur, last = [], [], None
    for p, dt in items:
        if not cur:
            cur, last = [p], dt
            continue
        if dt and last and (dt - last).total_seconds() > max_gap_seconds:
            groups.append(cur)
            cur, last = [p], dt
        else:
            cur.append(p)
            last = dt or last
    if cur:
        groups.append(cur)
    idx_map = []
    for gi, g in enumerate(groups):
        idx_map.extend([gi] * len(g))
    return groups, idx_map


def pil_to_qimage(im: Image.Image) -> QImage:
    """Robust PIL -> QImage conversion without ImageQt.
    Deep-copies buffer so Qt owns memory; safe for cross-thread signals.
    """
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        fmt = QImage.Format_RGBA8888
        channels = 4
    else:
        im = im.convert("RGB")
        fmt = QImage.Format_RGB888
        channels = 3
    w, h = im.size
    buf = im.tobytes()
    qimg = QImage(buf, w, h, channels * w, fmt)
    return qimg.copy()  # detach


def pil_to_qpixmap(im: Image.Image) -> QPixmap:
    return QPixmap.fromImage(pil_to_qimage(im))


# ----------------------------
# Worker (QThread)
# ----------------------------
class ThumbWorker(QObject):
    thumb_ready = pyqtSignal(int, QImage)   # idx, image
    progress = pyqtSignal(int, int)         # done, total
    finished = pyqtSignal()

    def __init__(self, image_paths, thumb_size=160):
        super().__init__()
        self.image_paths = image_paths
        self.thumb_size = int(thumb_size)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        total = len(self.image_paths)
        for i, p in enumerate(self.image_paths):
            if self._abort:
                break
            try:
                thumb_data = None
                try:
                    exif_dict = piexif.load(p)
                    thumb_data = exif_dict.get('thumbnail') if exif_dict else None
                except Exception:
                    pass
                if thumb_data:
                    from io import BytesIO
                    t = Image.open(BytesIO(thumb_data)).convert('RGB')
                else:
                    t = Image.open(p).convert('RGB')
                t.thumbnail((self.thumb_size, self.thumb_size), Image.LANCZOS)
                qimg = pil_to_qimage(t)
                self.thumb_ready.emit(i, qimg)
            except Exception:
                pass
            self.progress.emit(i + 1, total)
        self.finished.emit()


# ----------------------------
# UI Widgets
# ----------------------------
class ClickableLabel(QLabel):
    clicked = pyqtSignal(int)

    def __init__(self, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.setFocusPolicy(Qt.NoFocus)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class ImageGrouperBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Grouper Browser (PyQt5)')
        self.resize(1600, 1000)

        # State
        self.folder = ''
        self.images = []
        self.groups = []
        self.group_indices = []
        self.current_group_idx = 0
        self.current_idx_in_group = 0
        self.current_idx = 0
        self._current_image: Image.Image | None = None

        self.zoom_factor = 1.0
        self.zoom_center = None
        self.panning = False
        self.pan_last_pos = None

        self.thumbnail_labels = []
        self.active_thumbnail_label = None
        self._thumb_cache = {}  # path -> QPixmap (scaled)

        self.thumb_thread: QThread | None = None
        self.thumb_worker: ThumbWorker | None = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.status_label = QLabel('Ready')
        self.status_label.setFont(QFont('Arial', 12))
        layout.addWidget(self.status_label)

        self.info_label = QLabel('No folder selected')
        self.info_label.setFont(QFont('Arial', 14))
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.main_image_scroll_area = QScrollArea()
        self.main_image_label = QLabel()
        self.main_image_label.setFocusPolicy(Qt.StrongFocus)
        self.main_image_scroll_area.setWidget(self.main_image_label)
        self.main_image_scroll_area.setWidgetResizable(False)
        self.main_image_scroll_area.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.main_image_scroll_area, stretch=2)

        # Prevent scroll areas and buttons from stealing arrow keys
        self.main_image_scroll_area.setFocusPolicy(Qt.NoFocus)

        # Forward Ctrl+Wheel from the viewport so zoom works reliably
        self.main_image_scroll_area.viewport().installEventFilter(self)

        # Install application-level filter so we always get arrow keys
        QApplication.instance().installEventFilter(self)

        self.thumb_scroll_area = QScrollArea()
        self.thumb_widget = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_widget)
        self.thumb_scroll_area.setFocusPolicy(Qt.NoFocus)
        self.thumb_layout.setContentsMargins(6, 6, 6, 6)
        self.thumb_layout.setSpacing(6)
        self.thumb_scroll_area.setWidget(self.thumb_widget)
        self.thumb_scroll_area.setWidgetResizable(True)
        self.thumb_scroll_area.setFixedHeight(200)
        self.thumb_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.thumb_scroll_area)

        ctrl = QHBoxLayout()
        self.folder_btn = QPushButton('Select Folder')
        self.folder_btn.clicked.connect(self._select_folder)
        self.folder_btn.setFocusPolicy(Qt.NoFocus)
        ctrl.addWidget(self.folder_btn)
        layout.addLayout(ctrl)

        self.setLayout(layout)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.main_image_label.setMouseTracking(True)

        # Throttle rendering to ~60 FPS during pan/zoom
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self.show_zoomed_image)
        self._render_interval_ms = 16

    # -------- Folder / data
    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Image Folder', os.path.expanduser('~'))
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder: str):
        self.status_label.setText(f'Loading images from: {folder}')
        QApplication.processEvents()

        self.folder = folder
        self.images = get_image_paths(folder)
        self.groups, self.group_indices = group_images_by_time(self.images, max_gap_seconds=2)

        self.current_group_idx = 0
        self.current_idx_in_group = 0
        self._thumb_cache.clear()

        self.status_label.setText(f'Loaded {len(self.images)} images in {len(self.groups)} groups')
        self.update_view()
        self.setFocus(Qt.ActiveWindowFocusReason)
        QTimer.singleShot(0, self.show_zoomed_image)  # ensure render after layout

        # Ensure we own the keyboard after loading
        self.setFocus(Qt.ActiveWindowFocusReason)

    # -------- View update
    def update_view(self):
        if not self.groups:
            self._set_empty()
            return

        self.current_group_idx = max(0, min(self.current_group_idx, len(self.groups) - 1))
        group = self.groups[self.current_group_idx]
        if not group:
            self._set_empty()
            return
        self.current_idx_in_group = max(0, min(self.current_idx_in_group, len(group) - 1))

        img_path = group[self.current_idx_in_group]
        self.current_idx = self.images.index(img_path)

        self.status_label.setText(
            '←/→ image  ↑/↓ group  Ctrl+Wheel/+/- zoom  0 reset  Q quit\n'
            f'Group {self.current_group_idx+1}/{len(self.groups)}  '
            f'Image {self.current_idx_in_group+1}/{len(group)}  {os.path.basename(img_path)}'
        )

        try:
            with Image.open(img_path) as im:
                self._current_image = im.convert('RGB').copy()
            self.zoom_factor = max(self.zoom_factor, 1.0)
            self.show_zoomed_image()
            self.info_label.setText(f'Path: {img_path}')
        except Exception as e:
            self._current_image = None
            self.main_image_label.setText(f"[Unreadable] {os.path.basename(img_path)}\n{e}")

        self._build_or_reuse_thumb_labels(group)
        self._start_thumb_thread(group)

    def _set_empty(self):
        self.status_label.setText('No images found.')
        self.info_label.setText('No images found in folder.')
        self.main_image_label.clear()
        self._clear_thumb_labels()

    # -------- Thumbnails
    def _clear_thumb_labels(self):
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.thumbnail_labels.clear()
        self.active_thumbnail_label = None

    def _build_or_reuse_thumb_labels(self, group):
        if len(self.thumbnail_labels) != len(group):
            self._clear_thumb_labels()
            for i, _ in enumerate(group):
                lbl = ClickableLabel(i)
                lbl.setFixedSize(160, 120)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setText(str(i + 1))
                lbl.clicked.connect(self._on_thumb_clicked)
                self.thumb_layout.addWidget(lbl)
                self.thumbnail_labels.append(lbl)
        h = max(60, self.thumb_scroll_area.height() - 20)
        for i, p in enumerate(group):
            if p in self._thumb_cache:
                self.thumbnail_labels[i].setPixmap(self._thumb_cache[p])
            else:
                self.thumbnail_labels[i].setText(str(i + 1))
        self._highlight_active_thumbnail()

    def _on_thumb_clicked(self, idx: int):
        self.current_idx_in_group = idx
        self.update_view()

    def _start_thumb_thread(self, image_list):
        if self.thumb_worker:
            self.thumb_worker.abort()
        if self.thumb_thread:
            self.thumb_thread.quit()
            self.thumb_thread.deleteLater()

        self.thumb_thread = QThread()
        self.thumb_worker = ThumbWorker(image_list, thumb_size=max(60, self.thumb_scroll_area.height() - 20))
        self.thumb_worker.moveToThread(self.thumb_thread)

        self.thumb_thread.started.connect(self.thumb_worker.run)
        self.thumb_worker.thumb_ready.connect(self._on_thumb_ready, type=Qt.QueuedConnection)
        self.thumb_worker.progress.connect(self._on_thumb_progress, type=Qt.QueuedConnection)
        self.thumb_worker.finished.connect(self.thumb_thread.quit)
        self.thumb_worker.finished.connect(self._highlight_active_thumbnail)

        self.thumb_thread.start()

    def _on_thumb_ready(self, idx: int, qimage: QImage):
        group = self.groups[self.current_group_idx]
        if idx >= len(group) or idx >= len(self.thumbnail_labels):
            return
        h = max(60, self.thumb_scroll_area.height() - 20)
        pix = QPixmap.fromImage(qimage).scaledToHeight(h, Qt.SmoothTransformation)
        p = group[idx]
        self._thumb_cache[p] = pix
        self.thumbnail_labels[idx].setPixmap(pix)

    def _on_thumb_progress(self, done: int, total: int):
        self.status_label.setText(f'Generating thumbnails: {done}/{total}')

    def _highlight_active_thumbnail(self):
        if self.active_thumbnail_label:
            self.active_thumbnail_label.setStyleSheet("")
        if not self.groups:
            return
        group = self.groups[self.current_group_idx]
        if not group:
            return
        idx = self.current_idx_in_group
        if idx < len(self.thumbnail_labels):
            self.active_thumbnail_label = self.thumbnail_labels[idx]
            self.active_thumbnail_label.setStyleSheet("border: 3px solid #2b6cb0;")

    # -------- Image rendering / zoom
    def show_zoomed_image(self):
        """
        Zoom by scaling the whole image; never shrink the view window.
        - zoom_factor <= 1.0: fit-down only (no upscaling above 1:1)
        - zoom_factor  > 1.0: enlarge pixels; label grows; scrollbars handle navigation
        """
        if self._current_image is None:
            self.main_image_label.clear()
            return

        im = self._current_image
        w, h = im.size

        # Viewport size can be (0, 0) before layout; fall back to native
        vp = self.main_image_scroll_area.viewport()
        Lw = vp.width() if vp.width() > 1 else w
        Lh = vp.height() if vp.height() > 1 else h

        if self.zoom_factor <= 1.0:
            # Fit-down only (never upscale above 1:1)
            scale = min(1.0, min(Lw / w, Lh / h))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        else:
            # Pixel zoom: scale entire image by zoom_factor (no cropping)
            new_w, new_h = max(1, int(w * self.zoom_factor)), max(1, int(h * self.zoom_factor))

        # High-quality resize from the original each time (no cumulative blur)
        pix = pil_to_qpixmap(im.resize((new_w, new_h), Image.LANCZOS))

        # Important: with widgetResizable(False) we must size the label to the pixmap
        self.main_image_label.setPixmap(pix)
        self.main_image_label.setFixedSize(new_w, new_h)
        self.main_image_label.setScaledContents(False)
        self.main_image_label.setAlignment(Qt.AlignCenter)

    # -------- Navigation / events
    def eventFilter(self, obj, event: QEvent) -> bool:
        # Intercept Ctrl+Wheel on the scroll area's viewport for zooming
        if obj is self.main_image_scroll_area.viewport() and event.type() == QEvent.Wheel:
            self.wheelEvent(event)
            return True
        # Intercept key presses at the app level to ensure they are not missed
        if event.type() == QEvent.KeyPress:
            self.keyPressEvent(event)
            return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        if not self.groups or self._current_image is None:
            return
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom_in(event.pos())
            elif delta < 0:
                self._zoom_out(event.pos())
            event.accept()

    def mousePressEvent(self, event):
        if not self.groups or self._current_image is None:
            return
        if event.button() == Qt.LeftButton and self.zoom_factor > 1.0:
            self.panning = True
            self.pan_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning and self.pan_last_pos is not None:
            delta = event.pos() - self.pan_last_pos
            self.pan_last_pos = event.pos()
            h_bar = self.main_image_scroll_area.horizontalScrollBar()
            v_bar = self.main_image_scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.panning:
            self.panning = False
            self.pan_last_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        if key == Qt.Key_Q:
            self.close()
            return

        if not self.groups:
            return

        # Navigation
        if key == Qt.Key_Right:
            self.current_idx_in_group += 1
            if self.current_idx_in_group >= len(self.groups[self.current_group_idx]):
                self.current_idx_in_group = 0
            self.update_view()
        elif key == Qt.Key_Left:
            self.current_idx_in_group -= 1
            if self.current_idx_in_group < 0:
                self.current_idx_in_group = len(self.groups[self.current_group_idx]) - 1
            self.update_view()
        elif key == Qt.Key_Down:
            self.current_group_idx += 1
            if self.current_group_idx >= len(self.groups):
                self.current_group_idx = 0
            self.current_idx_in_group = 0
            self.update_view()
        elif key == Qt.Key_Up:
            self.current_group_idx -= 1
            if self.current_group_idx < 0:
                self.current_group_idx = len(self.groups) - 1
            self.current_idx_in_group = 0
            self.update_view()

        # Zooming
        elif key in (Qt.Key_Plus, Qt.Key_Equal) or (key == Qt.Key_Plus and mods & Qt.ControlModifier):
            self._zoom_in()
        elif key == Qt.Key_Minus or (key == Qt.Key_Minus and mods & Qt.ControlModifier):
            self._zoom_out()
        elif key == Qt.Key_0:
            self._zoom_reset()

    def _zoom_in(self, center=None):
        self._set_zoom(self.zoom_factor * 1.25, center)

    def _zoom_out(self, center=None):
        self._set_zoom(self.zoom_factor / 1.25, center)

    def _zoom_reset(self):
        self._set_zoom(1.0, None)

    def _set_zoom(self, factor, center):
        new_factor = max(1.0, factor)
        if abs(new_factor - self.zoom_factor) < 0.01:
            return

        self.zoom_factor = new_factor
        self.zoom_center = center
        self._render_timer.start(self._render_interval_ms)


# Entrypoint
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImageGrouperBrowser()
    win.show()

    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        if os.path.isdir(folder_path):
            win.load_folder(folder_path)
        else:
            print(f"Error: Folder not found at '{folder_path}'")

    sys.exit(app.exec_())

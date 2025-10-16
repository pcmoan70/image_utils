
import sys
import os
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QFileDialog, QPushButton, QSplitter, QSizePolicy, QListView
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSize
from PIL import Image

import numpy as np
import piexif
from datetime import datetime, timedelta


# ...existing code...

class ThumbWorker(QObject):
    thumb_ready = pyqtSignal(int, QPixmap)
    progress = pyqtSignal(int, int)

    def __init__(self, image_paths, thumb_size=160):
        super().__init__()
        self.image_paths = image_paths
        self.thumb_size = thumb_size
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        for i, p in enumerate(self.image_paths):
            if self._abort:
                break
            pix = None
            try:
                exif_dict = None
                try:
                    exif_dict = piexif.load(p)
                except Exception:
                    pass
                thumb_data = None
                if exif_dict:
                    thumb_data = exif_dict.get('thumbnail')
                if thumb_data:
                    # Embedded thumbnail is JPEG bytes
                    from io import BytesIO
                    thumb_img = Image.open(BytesIO(thumb_data))
                    thumb_img = thumb_img.convert('RGB')
                    thumb_img.thumbnail((self.thumb_size, self.thumb_size))
                    qimg = QImage(thumb_img.tobytes('raw', 'RGB'), thumb_img.width, thumb_img.height, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qimg)
                else:
                    # Fallback: generate thumbnail from image
                    thumb = Image.open(p)
                    thumb = thumb.convert('RGB')
                    thumb.thumbnail((self.thumb_size, self.thumb_size))
                    qimg = QImage(thumb.tobytes('raw', 'RGB'), thumb.width, thumb.height, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qimg)
                self.thumb_ready.emit(i, pix)
            except Exception:
                continue
            self.progress.emit(i+1, len(self.image_paths))

def is_image(path):
    # Skip AppleDouble files only
    base = os.path.basename(path)
    if base.startswith('._'):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

def get_image_paths(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if is_image(p):
                paths.append(p)
    return sorted(paths)

# --- EXIF grouping helpers ---
def get_exif_datetime(path):
    try:
        exif_dict = piexif.load(path)
        dt_bytes = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
        if dt_bytes:
            dt_str = dt_bytes.decode() if isinstance(dt_bytes, bytes) else dt_bytes
            return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    return None

def group_images_by_time(image_paths, max_gap_seconds=2):
    # Returns: list of groups (list of image paths), and a flat index->group mapping
    if not image_paths:
        return [], []
    # Get datetimes for all images
    dt_list = []
    for p in image_paths:
        dt = get_exif_datetime(p)
        dt_list.append(dt)
    # Sort by datetime (fallback to filename order if missing)
    sorted_items = sorted(zip(image_paths, dt_list), key=lambda x: (x[1] or datetime.min, x[0]))
    groups = []
    current_group = []
    last_dt = None
    for p, dt in sorted_items:
        if not current_group:
            current_group.append(p)
            last_dt = dt
        else:
            if dt and last_dt and (dt - last_dt).total_seconds() > max_gap_seconds:
                groups.append(current_group)
                current_group = [p]
            else:
                current_group.append(p)
            last_dt = dt or last_dt
    if current_group:
        groups.append(current_group)
    # Build flat index->group mapping
    group_indices = []
    for i, g in enumerate(groups):
        group_indices.extend([i]*len(g))
    return groups, group_indices

def pil2pixmap(im):
    if im.mode == 'RGB':
        r, g, b = im.split()
        im = Image.merge('RGB', (b, g, r))
    elif im.mode == 'RGBA':
        r, g, b, a = im.split()
        im = Image.merge('RGBA', (b, g, r, a))
    elif im.mode == 'L':
        im = im.convert('RGBA')
        r, g, b, a = im.split()
        im = Image.merge('RGBA', (b, g, r, a))

    im2 = im.convert('RGBA')
    data = im2.tobytes('raw', 'RGBA')
    qim = QImage(data, im.width, im.height, QImage.Format_ARGB32)
    return QPixmap.fromImage(qim)

class ImageGrouperBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Grouper Browser (PyQt5)')
        self.resize(1600, 1000)
        self.folder = ''
        self.images = []
        self.current_idx = 0
        self.groups = []
        self.group_indices = []
        self.thumb_thread = None
        self.thumb_worker = None
        self.zoom_factor = 1.0
        self._current_image = None
        self.panning = False
        self.pan_last_pos = None
        self.init_ui()

    def mousePressEvent(self, event):
        if self.zoom_factor > 1.0 and self.image_label.underMouse():
            if event.button() == Qt.LeftButton:
                self.panning = True
                self.pan_last_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.pan_last_pos
            self.pan_last_pos = event.pos()

            # Convert delta to image coordinates
            w, h = self._current_image.size
            label_w = self.image_label.width()
            label_h = self.image_label.height()

            dx = delta.x() * (w / self.zoom_factor) / label_w
            dy = delta.y() * (h / self.zoom_factor) / label_h

            cx, cy = self.zoom_center
            self.zoom_center = (cx - dx, cy - dy)
            
            self.show_zoomed_image()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)

    def on_thumbnail_clicked(self, item):
        idx = self.group_list.row(item)
        
        # Get the current group
        img_path = self.images[self.current_idx]
        group_idx = None
        for i, group in enumerate(self.groups):
            if img_path in group:
                group_idx = i
                break
        if group_idx is None:
            return # Should not happen
        
        group = self.groups[group_idx]
        
        self.current_idx = self.images.index(group[idx])
        self.update_view()

    def init_ui(self):
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.showFullScreen()
        layout = QVBoxLayout(self)
        # Info/status area (fixed at top)
        info_area = QVBoxLayout()
        self.status_label = QLabel('Ready')
        self.status_label.setFont(QFont('Arial', 12))
        info_area.addWidget(self.status_label)
        self.info_label = QLabel('No folder selected')
        self.info_label.setFont(QFont('Arial', 16))
        self.info_label.setWordWrap(True)
        info_area.addWidget(self.info_label)
        layout.addLayout(info_area)
        # Image display (center, expands)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=2)
        # Group panel (bottom)
        self.group_list = QListWidget()
        self.group_list.setFlow(QListWidget.LeftToRight)
        self.group_list.setFixedHeight(180)
        self.group_list.setSpacing(16)
        self.group_list.setViewMode(QListView.IconMode)
        self.group_list.setIconSize(QSize(self.group_list.height(), self.group_list.height()))
        self.group_list.setGridSize(QSize(350, self.group_list.height()))
        self.group_list.itemClicked.connect(self.on_thumbnail_clicked)
        layout.addWidget(self.group_list)
        # Folder select button
        btn_layout = QHBoxLayout()
        self.folder_btn = QPushButton('Select Folder')
        self.folder_btn.clicked.connect(self.select_folder_dialog)
        btn_layout.addWidget(self.folder_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        # Keyboard navigation
        self.setFocusPolicy(Qt.StrongFocus)

    def select_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Image Folder', os.path.expanduser('~'))
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        self.status_label.setText(f'Loading images from: {folder}')
        QApplication.processEvents()
        self.folder = folder
        self.images = get_image_paths(folder)
        self.current_idx = 0
        self.groups, self.group_indices = group_images_by_time(self.images, max_gap_seconds=2)
        self.status_label.setText(f'Loaded {len(self.images)} images in {len(self.groups)} groups from: {folder}')
        self.update_view()
        self.start_thumb_thread()

    def update_view(self):
        if not self.images or not self.groups:
            self.status_label.setText('No images found.')
            self.info_label.setText('No images found in folder.')
            self.image_label.clear()
            self.group_list.clear()
            self._current_image = None
            return
        # Always recompute group_idx and group for the current image
        img_path = self.images[self.current_idx]
        group_idx = None
        for i, group in enumerate(self.groups):
            if img_path in group:
                group_idx = i
                break
        if group_idx is None:
            group_idx = 0
            group = self.groups[0]
            self.current_idx = self.images.index(group[0])
        else:
            group = self.groups[group_idx]
        idx_in_group = group.index(img_path)
        self.status_label.setText(
            f'Navigation: ←/→: prev/next in group | ↑/↓: prev/next group | Ctrl+wheel/+/-: zoom | Q: quit | S: store\n'
            f'Group {group_idx+1}/{len(self.groups)} | Image {idx_in_group+1}/{len(group)} | {os.path.basename(img_path)}'
        )
        QApplication.processEvents()
        try:
            im = Image.open(img_path)
            self._current_image = im.copy()
            self.show_zoomed_image()
            self.info_label.setText(
                f'Group {group_idx+1}/{len(self.groups)}\n'
                f'Image {idx_in_group+1}/{len(group)}: {os.path.basename(img_path)}\n'
                f'Path: {img_path}'
            )
        except Exception as e:
            self._current_image = None
            self.image_label.setText(f"[Unreadable image: {os.path.basename(img_path)}]")
            self.info_label.setText(f"Unreadable image: {img_path}\n{e}")
            self.status_label.setText(f'Error loading image {idx_in_group+1}/{len(group)} in group {group_idx+1}/{len(self.groups)}')
        # Show only thumbnails for current group
        self.group_list.clear()
        for i, p in enumerate(group):
            item = QListWidgetItem()
            item.setToolTip(os.path.basename(p))
            self.group_list.addItem(item)
        self.group_list.setCurrentRow(idx_in_group)
        self.group_list.scrollToItem(self.group_list.currentItem())
        # Start thumbnail thread for current group only
        self.start_thumb_thread(group)

    def show_zoomed_image(self):
        if self._current_image is None:
            self.image_label.clear()
            return
        im = self._current_image
        w, h = im.size
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        if self.zoom_factor > 1.0:
            if not hasattr(self, 'zoom_center') or self.zoom_center is None:
                self.zoom_center = (w // 2, h // 2)
            cx, cy = self.zoom_center
            
            # Corrected crop size calculation
            crop_w = int(w / self.zoom_factor)
            crop_h = int(h / self.zoom_factor)

            if crop_w < 1 or crop_h < 1:
                return

            left = max(0, cx - crop_w // 2)
            upper = max(0, cy - crop_h // 2)
            right = min(w, left + crop_w)
            lower = min(h, upper + crop_h)
            crop_box = (left, upper, right, lower)
            
            try:
                cropped = im.crop(crop_box)
                
                # Preserve aspect ratio
                w_crop, h_crop = cropped.size
                scale = min(label_w / w_crop, label_h / h_crop, 1.0)
                new_w = int(w_crop * scale)
                new_h = int(h_crop * scale)

                im_scaled = cropped.resize((new_w, new_h), Image.LANCZOS)
                pix = pil2pixmap(im_scaled)
                self.image_label.setPixmap(pix)
                self.image_label.setAlignment(Qt.AlignCenter)
                self.image_label.setScaledContents(False)
            except Exception as e:
                print(f"Error while zooming: {e}")
                self.image_label.setText("Error during zoom")
        else:
            self.zoom_factor = 1.0
            scale = min(label_w / w, label_h / h, 1.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            im_scaled = im.resize((new_w, new_h), Image.LANCZOS)
            pix = pil2pixmap(im_scaled)
            self.image_label.setPixmap(pix)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setScaledContents(False)
            self.zoom_center = None
    def resizeEvent(self, event):
        # Rescale image to fit window when not zoomed
        if self._current_image is not None and self.zoom_factor == 1.0:
            self.show_zoomed_image()
        super().resizeEvent(event)

    def start_thumb_thread(self, image_list=None):
        # Abort previous worker if running
        if self.thumb_worker:
            self.thumb_worker.abort()
        if image_list is None:
            image_list = self.images
        self.thumb_worker = ThumbWorker(image_list, thumb_size=self.group_list.height())
        self.thumb_worker.thumb_ready.connect(self.set_thumb)
        self.thumb_worker.progress.connect(self.set_thumb_progress)
        self.thumb_thread = threading.Thread(target=self.thumb_worker.run, daemon=True)
        self.thumb_thread.start()

    def set_thumb(self, idx, pix):
        from PyQt5.QtGui import QIcon
        item = self.group_list.item(idx)
        if item:
            # Scale pixmap to fill height of group_list
            h = self.group_list.height()
            scaled_pix = pix.scaledToHeight(h, Qt.SmoothTransformation)
            item.setIcon(QIcon(scaled_pix))

    def set_thumb_progress(self, done, total):
        self.status_label.setText(f'Generating thumbnails: {done}/{total}')

    def keyPressEvent(self, event):
        if not self.images or not self.groups:
            return
        # Only update current_idx and call update_view; let update_view recompute group
        img_path = self.images[self.current_idx]
        group_idx = None
        for i, group in enumerate(self.groups):
            if img_path in group:
                group_idx = i
                break
        if group_idx is None:
            group_idx = 0
            group = self.groups[0]
        else:
            group = self.groups[group_idx]
        idx_in_group = group.index(img_path)
        if event.key() in (Qt.Key_Right, Qt.Key_L):
            if idx_in_group < len(group) - 1:
                self.current_idx = self.images.index(group[idx_in_group + 1])
                self.update_view()
        elif event.key() in (Qt.Key_Left, Qt.Key_H):
            if idx_in_group > 0:
                self.current_idx = self.images.index(group[idx_in_group - 1])
                self.update_view()
        elif event.key() == Qt.Key_Up:
            if group_idx > 0:
                self.current_idx = self.images.index(self.groups[group_idx - 1][0])
                self.zoom_factor = 1.0
                self.update_view()
        elif event.key() == Qt.Key_Down:
            if group_idx < len(self.groups) - 1:
                self.current_idx = self.images.index(self.groups[group_idx + 1][0])
                self.zoom_factor = 1.0
                self.update_view()
            else:
                self.current_idx = self.images.index(self.groups[0][0])
                self.zoom_factor = 1.0
                self.update_view()
        elif event.key() == Qt.Key_Q:
            self.close()
        elif event.key() == Qt.Key_S:
            pass
        elif (event.modifiers() & Qt.ControlModifier) and event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_factor = min(self.zoom_factor * 1.25, 10.0)
            self.show_zoomed_image()
        elif (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_Minus:
            self.zoom_factor = max(self.zoom_factor / 1.25, 0.1)
            self.show_zoomed_image()
        elif (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_0:
            self.zoom_factor = 1.0
            self.show_zoomed_image()
        elif event.key() == Qt.Key_Q:
            self.close()
        elif event.key() == Qt.Key_S:
            # TODO: implement store
            pass
        elif (event.modifiers() & Qt.ControlModifier) and event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_factor = min(self.zoom_factor * 1.25, 10.0)
            self.show_zoomed_image()
        elif (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_Minus:
            self.zoom_factor = max(self.zoom_factor / 1.25, 0.1)
            self.show_zoomed_image()
        elif (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_0:
            self.zoom_factor = 1.0
            self.show_zoomed_image()
    def wheelEvent(self, event):

        if self._current_image is not None and (event.modifiers() & Qt.ControlModifier):
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor = min(self.zoom_factor * 1.1, 10.0)
            else:
                self.zoom_factor = max(self.zoom_factor / 1.1, 0.1)
            self.show_zoomed_image()


if __name__ == "__main__":
    print("Starting Image Grouper Browser...")
    app = QApplication(sys.argv)
    win = ImageGrouperBrowser()
    win.show()

    # Check for folder path from command line
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        if os.path.isdir(folder_path):
            win.load_folder(folder_path)
        else:
            print(f"Error: Folder not found at '{folder_path}'")

    sys.exit(app.exec_())

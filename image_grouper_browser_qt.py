import sys
import os
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QFileDialog, QPushButton, QSplitter, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PIL import Image
import numpy as np
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
            try:
                thumb = Image.open(p)
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

def pil2pixmap(im):
    if im.mode != 'RGB':
        im = im.convert('RGB')
    data = im.tobytes('raw', 'RGB')
    qimg = QImage(data, im.width, im.height, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

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
        self._current_image = None  # Store current PIL image for zooming
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        # Status label (top)
        self.status_label = QLabel('Ready')
        self.status_label.setFont(QFont('Arial', 12))
        layout.addWidget(self.status_label)
        # Info label (below status)
        self.info_label = QLabel('No folder selected')
        self.info_label.setFont(QFont('Arial', 16))
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        # Image display (middle)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=2)
        # Group panel (bottom)
        self.group_list = QListWidget()
        self.group_list.setFlow(QListWidget.LeftToRight)
        self.group_list.setFixedHeight(180)
        self.group_list.setSpacing(8)
        layout.addWidget(self.group_list)
        # Folder select button
        btn_layout = QHBoxLayout()
        self.folder_btn = QPushButton('Select Folder')
        self.folder_btn.clicked.connect(self.select_folder)
        btn_layout.addWidget(self.folder_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        # Keyboard navigation
        self.setFocusPolicy(Qt.StrongFocus)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Image Folder', os.path.expanduser('~'))
        if folder:
            self.status_label.setText(f'Loading images from: {folder}')
            QApplication.processEvents()
            self.folder = folder
            self.images = get_image_paths(folder)
            self.current_idx = 0
            self.groups = [self.images] if self.images else []
            self.group_indices = list(range(len(self.images)))
            self.status_label.setText(f'Loaded {len(self.images)} images from: {folder}')
            self.update_view()
            self.start_thumb_thread()

    def update_view(self):
        if not self.images:
            self.status_label.setText('No images found.')
            self.info_label.setText('No images found in folder.')
            self.image_label.clear()
            self.group_list.clear()
            self._current_image = None
            return
        img_path = self.images[self.current_idx]
        self.status_label.setText(f'Loading image {self.current_idx+1}/{len(self.images)}...')
        QApplication.processEvents()
        try:
            im = Image.open(img_path)
            self._current_image = im.copy()
            self.show_zoomed_image()
            self.info_label.setText(f'Image {self.current_idx+1}/{len(self.images)}: {os.path.basename(img_path)}\n{img_path}')
            self.status_label.setText(f'Showing image {self.current_idx+1}/{len(self.images)}')
        except Exception as e:
            self._current_image = None
            self.image_label.setText(f"[Unreadable image: {os.path.basename(img_path)}]")
            self.info_label.setText(f"Unreadable image: {img_path}\n{e}")
            self.status_label.setText(f'Error loading image {self.current_idx+1}/{len(self.images)}')
        # Show group (all images for now) - thumbnails
        self.group_list.clear()
        for i, p in enumerate(self.images):
            item = QListWidgetItem()
            item.setToolTip(os.path.basename(p))
            self.group_list.addItem(item)
        self.group_list.setCurrentRow(self.current_idx)
        self.group_list.scrollToItem(self.group_list.currentItem())

    def show_zoomed_image(self):
        if self._current_image is None:
            self.image_label.clear()
            return
        im = self._current_image
        if self.zoom_factor != 1.0:
            w, h = im.size
            new_w = int(w * self.zoom_factor)
            new_h = int(h * self.zoom_factor)
            im = im.resize((new_w, new_h), Image.LANCZOS)
        pix = pil2pixmap(im)
        self.image_label.setPixmap(pix)

    def start_thumb_thread(self):
        # Abort previous worker if running
        if self.thumb_worker:
            self.thumb_worker.abort()
        self.thumb_worker = ThumbWorker(self.images)
        self.thumb_worker.thumb_ready.connect(self.set_thumb)
        self.thumb_worker.progress.connect(self.set_thumb_progress)
        self.thumb_thread = threading.Thread(target=self.thumb_worker.run, daemon=True)
        self.thumb_thread.start()

    def set_thumb(self, idx, pix):
        from PyQt5.QtGui import QIcon
        item = self.group_list.item(idx)
        if item:
            item.setIcon(QIcon(pix))

    def set_thumb_progress(self, done, total):
        self.status_label.setText(f'Generating thumbnails: {done}/{total}')

    def keyPressEvent(self, event):
        if not self.images:
            return
        if event.key() in (Qt.Key_Right, Qt.Key_L):
            self.current_idx = (self.current_idx + 1) % len(self.images)
            self.zoom_factor = 1.0
            self.update_view()
        elif event.key() in (Qt.Key_Left, Qt.Key_H):
            self.current_idx = (self.current_idx - 1) % len(self.images)
            self.zoom_factor = 1.0
            self.update_view()
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
        else:
            super().wheelEvent(event)
        # (Other key handlers for D, J, K can be added here if needed)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageGrouperBrowser()
    win.showMaximized()
    sys.exit(app.exec_())

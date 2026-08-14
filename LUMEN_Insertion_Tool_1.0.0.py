import os
import sys
import logging
import cv2
import csv
import numpy as np
import json

import lumen_core as core

from PyQt5.QtCore import (
    QObject, QThread, pyqtSignal, Qt, QSize, QUrl, QEvent, QTimer, QPoint, QRect,
    QLineF, QPointF, QRectF, QPropertyAnimation, pyqtProperty, pyqtSignal)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QInputDialog,
    QGraphicsDropShadowEffect, QPushButton, QLabel, QMainWindow,
    QSpacerItem, QSizePolicy, QSlider, QFileDialog, QMenu, QAction,
    QStyle, QMessageBox, QGridLayout, QCheckBox, QDialog, QDialogButtonBox, QProgressDialog)
from PyQt5.QtGui import (
    QCursor, QPixmap, QColor, QPainter, QBrush, QPen, QImage, QPolygonF)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

def app_directory() -> str:
    """Where the application lives — the bundle folder when packaged."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def user_data_directory() -> str:
    """A per-user, definitely-writable folder for logs and settings.

    A packaged build may be launched from a read-only folder, a network share
    or a memory stick, so the log cannot simply go in the working directory:
    that would fail at import time, before there is any window to report it in.
    """
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        root = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        root = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share")

    directory = os.path.join(root, "LUMEN")
    try:
        os.makedirs(directory, exist_ok=True)
        return directory
    except OSError:
        import tempfile
        return tempfile.gettempdir()


LOG_PATH = os.path.join(user_data_directory(), "app_debug.log")

logging.basicConfig(
    filename=LOG_PATH,
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Application start; log at %s", LOG_PATH)

# Let a bundled ffmpeg sitting beside the application be found, so audio works
# without anything being installed.
core.register_binary_dir(app_directory())

def log_uncaught_exceptions(exctype, value, tb):
    import traceback
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    logging.error("Uncaught exception: %s", err_msg)
    print("Uncaught exception:", err_msg)

sys.excepthook = log_uncaught_exceptions

def ms_to_mmss(milliseconds: int) -> str:
    """Utility to convert milliseconds -> 'MM:SS' format."""
    seconds = milliseconds // 1000
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02}:{ss:02}"

# The Kalman filter, planar tracker, compositor and geometry helpers all live
# in lumen_core so that the preview and the render share one implementation and
# cannot drift apart, and so they can be tested without a display.
SimpleKalmanFilter = core.SimpleKalmanFilter


class AdjustmentDialog(QDialog):
    def __init__(self, adjustment_type: str, current_value: float, parent=None):
        super().__init__(parent)
        self.adjustment_type = adjustment_type
        self.value = current_value
        self.setWindowTitle(f"Adjust {adjustment_type.capitalize()}")
        self.setStyleSheet("""
            QDialog {
                background-color: #2A2A2A;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                background: #444444;
                height: 6px;
            }
            QSlider::handle:horizontal {
                background: #888888;
                width: 14px;
            }
            QPushButton {
                background-color: #3C3C3C;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)

        layout = QVBoxLayout(self)

        self.label = QLabel(f"{adjustment_type.capitalize()}: {current_value}")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        if adjustment_type == "brightness":
            self.slider.setRange(-100, 100)
            self.slider.setValue(int(current_value))
        elif adjustment_type == "contrast":
            self.slider.setRange(0, 300)
            self.slider.setValue(int(current_value*100))
        layout.addWidget(self.slider)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btn_box, alignment=Qt.AlignRight)
        # ^ put them bottom-right

        self.slider.valueChanged.connect(self.on_slider_changed)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

    def on_slider_changed(self, val):
        if self.adjustment_type == "brightness":
            self.value = val
            self.label.setText(f"Brightness: {val}")
        else:  # contrast
            self.value = val / 100.0
            self.label.setText(f"Contrast: {self.value:.2f}")

    def get_value(self):
        return self.value

class QSwitch(QWidget):
    toggled = pyqtSignal(bool)  # Signal: emit True when turned on, False when turned off

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self._handle_position = 0.0  # from 0.0 (left) to 1.0 (right)
        self.setFixedSize(60, 30)  # overall size of the toggle

        self._anim = QPropertyAnimation(self, b"handlePos", self)
        self._anim.setDuration(200)  # ms
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.finished.connect(self._on_anim_finished)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked == checked:
            return  # no change
        self._checked = checked
        if checked:
            self._anim.setDirection(QPropertyAnimation.Forward)
        else:
            self._anim.setDirection(QPropertyAnimation.Backward)
        self._anim.start()

    def toggle(self):
        self.setChecked(not self._checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle()
        super().mousePressEvent(event)

    def getHandlePos(self):
        return self._handle_position

    def setHandlePos(self, pos):
        self._handle_position = pos
        self.update()

    handlePos = pyqtProperty(float, fget=getHandlePos, fset=setHandlePos)

    def _on_anim_finished(self):
        # After the animation finishes, emit toggled
        self.toggled.emit(self._checked)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the background
        background_color = QColor("#00A000") if self._checked else QColor("#FF0000")
        painter.setBrush(background_color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)

        # Draw the handle
        handle_radius = 13
        handle_color = QColor("#FFFFFF")  # White handle
        painter.setBrush(handle_color)
        painter.setPen(Qt.NoPen)

        available_movement = self.width() - 2 * handle_radius  # 60 - 26 = 34
        handle_x = self._handle_position * available_movement
        handle_y = (self.height() - 2 * handle_radius) / 2  # Center vertically

        handle_center_x = int(handle_x + handle_radius)
        handle_center_y = int(handle_y + handle_radius)

        painter.drawEllipse(QPoint(handle_center_x, handle_center_y), handle_radius, handle_radius)
        painter.end()

class HoverButton(QPushButton):
    """A button that shows a shadow and hand cursor on hover."""
    def __init__(self, text="", parent=None):
        super(HoverButton, self).__init__(text, parent)
        self.setFlat(True)
        self.setStyleSheet("""
            QPushButton {
                border: none;
                color: white;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #E6E6E6;
            }
        """)
        self.hover_effect = None

    def enterEvent(self, event):
        logging.debug(f"Hover enter on button '{self.text()}'")
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.hover_effect = QGraphicsDropShadowEffect(self)
        self.hover_effect.setBlurRadius(15)
        self.hover_effect.setColor(QColor(0, 0, 0, 80))
        self.hover_effect.setOffset(0, 3)
        self.setGraphicsEffect(self.hover_effect)
        super(HoverButton, self).enterEvent(event)

    def leaveEvent(self, event):
        logging.debug(f"Hover leave on button '{self.text()}'")
        self.setGraphicsEffect(None)
        self.hover_effect = None
        super(HoverButton, self).leaveEvent(event)

class IconWidget(QFrame):
    """A widget that contains an icon and a label below it."""
    def __init__(self, icon_char, label_text, parent=None):
        super(IconWidget, self).__init__(parent)
        self.selected = False

        self.default_base_color = "#1A1A1A"
        self.default_hover_color = "#333333"
        self.default_selected_color = "#555555"

        if label_text in ["Draw", "Overlay"]:
            self.selected_color = "#00A000"
        else:
            self.selected_color = self.default_selected_color

        self.base_color = self.default_base_color
        self.hover_color = self.default_hover_color

        self.setFixedSize(80, 100)
        self.set_normal_style()

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)

        self.icon_label = QLabel(icon_char)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 24px; color: white;")

        self.meaning_label = QLabel(label_text)
        self.meaning_label.setAlignment(Qt.AlignCenter)
        self.meaning_label.setStyleSheet("font-size: 12px; color: #E0E0E0;")

        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.meaning_label)
        self.setLayout(self.layout)

    def set_normal_style(self):
        self.setStyleSheet(f"""
            QFrame {{
                border-radius: 10px;
                background-color: {self.base_color};
            }}
        """)

    def enterEvent(self, event):
        if not self.selected:
            self.setStyleSheet(f"""
                QFrame {{
                    border-radius: 10px;
                    background-color: {self.hover_color};
                }}
            """)
        super(IconWidget, self).enterEvent(event)

    def leaveEvent(self, event):
        if not self.selected:
            self.set_normal_style()
        super(IconWidget, self).leaveEvent(event)

    def set_selected(self, selected):
        self.selected = selected
        if self.selected:
            self.setStyleSheet(f"""
                QFrame {{
                    border-radius: 10px;
                    background-color: {self.selected_color};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    border-radius: 10px;
                    background-color: {self.base_color};
                }}
            """)

class MagnifierWidget(QWidget):
    def __init__(self, parent=None):
        super(MagnifierWidget, self).__init__(parent)
        self.setStyleSheet("background-color: black; border: 2px solid black;")
        self.setMinimumSize(150, 150)

        # The base frame (np.ndarray, BGR format)
        self.base_frame = None
        
        # List of up to 4 points in the base frame’s coordinate space.
        self.overlay_points = []
        
        # Zoom factor (higher => more magnification => smaller source ROI).
        self.zoom_factor = 4.0
        
        # Extra buffer in pixels added around the source ROI.
        self.buffer = 10
        
        # Index of the “active” point (0..3). -1 => none selected => crosshair is red.
        self.selected_index = -1

        # For manual resizing
        self.resizing = False
        self.last_mouse_pos = None
        self.resize_handle_size = 20  # size of the “corner handle”
        # Set here as well as on the first manual resize: update_magnifier_dimensions
        # reads it as soon as a second corner exists, which is long before the
        # user has had any reason to drag the magnifier.
        self.user_resized = False
        
        self.setMouseTracking(True)

    def sizeHint(self):
        # By default, just return current size
        return QSize(self.width(), self.height())

    def setData(
        self, 
        base_frame: np.ndarray, 
        overlay_points: list, 
        zoom_factor: float = None, 
        buffer: int = None, 
        selected_index: int = None
    ):
        """
        Update the magnifier data and refresh.

        :param base_frame:     np.ndarray BGR image
        :param overlay_points: up to 4 (x, y) points in base_frame coords
        :param zoom_factor:    optional new zoom factor
        :param buffer:         optional new pixel buffer
        :param selected_index: which corner is active (0..3), or -1 for none
        """
        self.base_frame = base_frame
        self.overlay_points = overlay_points

        if zoom_factor is not None:
            self.zoom_factor = zoom_factor
        if buffer is not None:
            self.buffer = buffer
        if selected_index is not None:
            self.selected_index = selected_index

        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if (event.x() >= self.width() - self.resize_handle_size and
                event.y() >= self.height() - self.resize_handle_size):
                self.resizing = True
                self.last_mouse_pos = event.globalPos()
                self.user_resized = True   # This flag disables auto‑resizing
                event.accept()
                return
        super(MagnifierWidget, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing:
            delta = event.globalPos() - self.last_mouse_pos
            new_w = max(self.minimumWidth(), self.width() + delta.x())
            new_h = max(self.minimumHeight(), self.height() + delta.y())

            self.resize(new_w, new_h)
            self.last_mouse_pos = event.globalPos()
            event.accept()
            return
        else:
            # Change mouse cursor if near bottom-right corner
            if (event.x() >= self.width() - self.resize_handle_size
                and event.y() >= self.height() - self.resize_handle_size):
                self.setCursor(Qt.SizeFDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super(MagnifierWidget, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.resizing:
            self.resizing = False
            event.accept()
            return
        super(MagnifierWidget, self).mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Fill the background and draw a black border:
        painter.fillRect(self.rect(), Qt.black)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Draw dividing lines for 4 quadrants
        mid_x = w // 2
        mid_y = h // 2

        painter.setPen(QPen(Qt.white, 1))
        painter.drawLine(mid_x, 0, mid_x, h)
        painter.drawLine(0, mid_y, w, mid_y)

        # If no base_frame, done
        if self.base_frame is None:
            painter.end()
            return

        frame_h, frame_w = self.base_frame.shape[:2]

        # Quadrants: (We “swap” corners 2 and 3)
        #  index 0 => top-left
        #  index 1 => top-right
        #  index 2 => bottom-right
        #  index 3 => bottom-left
        quadrants = [
            (QRect(0, 0, mid_x, mid_y)),               # quadrant for point0
            (QRect(mid_x, 0, w - mid_x, mid_y)),       # quadrant for point1
            (QRect(mid_x, mid_y, w - mid_x, h - mid_y)),  # quadrant for point2
            (QRect(0, mid_y, mid_x, h - mid_y))           # quadrant for point3
        ]

        # For each point, show a magnified region in the correct quadrant
        for i, (px, py) in enumerate(self.overlay_points):
            if i >= 4:
                break

            dest_rect = quadrants[i]
            # figure out the smaller dimension of quadrant => used as ROI size
            dest_size = min(dest_rect.width(), dest_rect.height())
            src_size = int(dest_size / self.zoom_factor) + self.buffer

            # center the ROI around (px, py)
            src_x = int(px - src_size/2)
            src_y = int(py - src_size/2)

            # clamp ROI inside base_frame
            src_x = max(0, min(src_x, frame_w - src_size))
            src_y = max(0, min(src_y, frame_h - src_size))

            roi = self.base_frame[src_y : src_y + src_size, src_x : src_x + src_size].copy()

            # resize ROI to fill quadrant
            resized_roi = cv2.resize(roi, (dest_rect.width(), dest_rect.height()),
                                     interpolation=cv2.INTER_LINEAR)

            # convert BGR->RGB->QImage
            resized_rgb = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
            bytes_per_line = dest_rect.width() * 3
            qimg = QImage(resized_rgb.data, dest_rect.width(), dest_rect.height(),
                          bytes_per_line, QImage.Format_RGB888)

            # draw it in quadrant
            painter.drawImage(dest_rect, qimg)

            # crosshair color = green if i == selected_index, else red
            cross_color = QColor(0, 255, 0) if (i == self.selected_index) else QColor(255, 0, 0)
            painter.setPen(QPen(cross_color, 2))

            cx = dest_rect.x() + dest_rect.width() // 2
            cy = dest_rect.y() + dest_rect.height() // 2
            painter.drawLine(QPoint(cx - 5, cy), QPoint(cx + 5, cy))
            painter.drawLine(QPoint(cx, cy - 5), QPoint(cx, cy + 5))

        painter.end()

class TrackingOverlay(QWidget):
    points_updated = pyqtSignal(int, list)
    shape_changed = pyqtSignal()
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(TrackingOverlay, self).__init__(parent)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

        self.tracking_enabled = False
        self.points = []  # list of (x, y) corner coords
        self.drag_index = -1  # index of corner being dragged, or -1 if none
        self.selected_point_index = -1  # which corner is selected (via keys, etc.)

        # Curved edges. Each edge carries two Bezier control points, stored as
        # offsets from their straight-line positions so they survive the
        # corners moving. All-zero offsets == a plain straight-edged quad.
        self.curved_enabled = False
        self.curvature = np.zeros((4, 2, 2), np.float32)
        self.drag_control = None      # (edge_index, control_index) while dragging

        # Overlay references. The creative is kept as a BGRA numpy array so the
        # preview and the render composite from exactly the same pixels.
        self.overlay_bgra = None
        self.overlay_source_path = None
        self.inserted_overlay_is_video = False
        self.overlay_video_cap = None
        self.overlay_video_path = None
        self.inserted_overlay_start_frame = 0

        # For hovering
        self.active_point_index = -1

        # Overlay effects
        self.brightness = 0  # Range -100..100
        self.contrast = 1.0   # Range 0..3
        self.colourise_enabled = False
        self.colourise_factor = 0

        # Tracking history: frame_index -> list of 4 corners
        self.tracking_history = {}

        # Widget config
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            disp_x, disp_y = event.pos().x(), event.pos().y()
            rx, ry = self.display_to_raw(disp_x, disp_y)

            found_drag = False
            for i, (px, py) in enumerate(self.points):
                dx, dy = self.raw_to_display(px, py)
                if abs(dx - disp_x) < 10 and abs(dy - disp_y) < 10:
                    self.drag_index = i
                    found_drag = True
                    break

            # Corners win ties, then Bezier handles when curving is on.
            if not found_drag and self.curved_enabled and len(self.points) == 4:
                hit = self.control_at(disp_x, disp_y)
                if hit is not None:
                    self.drag_control = hit
                    found_drag = True

            if not found_drag:
                if len(self.points) < 4:
                    self.points.append((rx, ry))
                    self.update()
                    if len(self.points) == 4:
                        self.auto_disable_tracking()
            if self.points:
                self.selected_point_index = len(self.points) - 1

            self.shape_changed.emit()

            # Immediately take focus so key events (1-4) work right away.
            self.setFocus()
        super(TrackingOverlay, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # REMOVED the if not self.tracking_enabled check
        disp_x, disp_y = event.pos().x(), event.pos().y()
        rx, ry = self.display_to_raw(disp_x, disp_y)

        if self.drag_control is not None:
            edge, slot = self.drag_control
            self.set_control_point(edge, slot, (rx, ry))
            self.update()
            self.shape_changed.emit()
            super(TrackingOverlay, self).mouseMoveEvent(event)
            return
        elif self.drag_index != -1:
            # dragging a corner
            self.points[self.drag_index] = (rx, ry)
            self.update()
            self.shape_changed.emit()
            super(TrackingOverlay, self).mouseMoveEvent(event)
            return
        else:
            # Not dragging => see if we are hovering
            hover_idx = -1
            for i, (px, py) in enumerate(self.points):
                dx, dy = self.raw_to_display(px, py)
                if abs(dx - disp_x) < 10 and abs(dy - disp_y) < 10:
                    hover_idx = i
                    break

            if hover_idx != self.active_point_index:
                self.active_point_index = hover_idx
                self.shape_changed.emit()

        super(TrackingOverlay, self).mouseMoveEvent(event)

    def control_at(self, disp_x, disp_y, radius=8):
        """Which Bezier handle, if any, is under this display position."""
        if not self.curved_enabled or len(self.points) != 4:
            return None
        region = self.current_region()
        for edge in range(4):
            for slot in range(2):
                cx, cy = self.raw_to_display(*region.controls[edge, slot])
                if abs(cx - disp_x) < radius and abs(cy - disp_y) < radius:
                    return (edge, slot)
        return None

    def set_control_point(self, edge: int, slot: int, point):
        """Move a Bezier handle to a point in raw video coordinates.

        The region stores the bend in the edge's own frame, so the curve
        survives the corners being dragged and rotates with the surface once
        tracking picks it up.
        """
        region = self.current_region()
        region.set_control_point(edge, slot, point)
        self.curvature = region.curvature

    def mouseReleaseEvent(self, event):
        was_dragging = (self.drag_index != -1 or self.drag_control is not None)
        self.drag_index = -1
        self.drag_control = None

        mw = self.get_main_window()
        if mw:
            cpanel = mw.central_panel
            if cpanel and cpanel.cap and cpanel.cap.isOpened():
                current_frame_index = cpanel.get_current_frame_index()
                if len(self.points) == 4:
                    self.tracking_history[current_frame_index] = self.points[:]
                else:
                    if current_frame_index in self.tracking_history:
                        del self.tracking_history[current_frame_index]

        # shape changed
        self.shape_changed.emit()
        super(TrackingOverlay, self).mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5]:
            num = key - Qt.Key_0
            if num <= 4:
                self.selected_point_index = num - 1
            else:
                self.selected_point_index = 4

            # Immediately redraw so the newly selected corner turns green
            self.update()
            self.shape_changed.emit()
            self.selection_changed.emit()
            event.accept()
            return

        if self.selected_point_index < 0:
            return

        step = 1
        dx, dy = 0, 0
        if key == Qt.Key_Left:
            dx = -step
        elif key == Qt.Key_Right:
            dx = step
        elif key == Qt.Key_Up:
            dy = -step
        elif key == Qt.Key_Down:
            dy = step

        if dx == 0 and dy == 0:
            # No arrow key => do nothing
            return

        if self.selected_point_index < 4:
            # move just that corner
            if len(self.points) == 4:
                px, py = self.points[self.selected_point_index]
                self.points[self.selected_point_index] = (px + dx, py + dy)
        else:
            # move entire shape
            if len(self.points) == 4:
                for i in range(4):
                    px, py = self.points[i]
                    self.points[i] = (px + dx, py + dy)

        self.update()
        self.shape_changed.emit()
        event.accept()

    def set_tracking_enabled(self, enabled: bool):
        self.tracking_enabled = enabled
        if enabled:
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.show()
        else:
            self.drag_index = -1
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def set_brightness(self, brightness: int):
        self.brightness = brightness
        self.update()

    def set_contrast(self, contrast: float):
        self.contrast = contrast
        self.update()

    def auto_disable_tracking(self):
        mw = self.get_main_window()
        if mw:
            for icon_widget in mw.left_col.icons:
                if icon_widget.meaning_label.text() == "Draw":
                    icon_widget.set_selected(False)

            mw.central_panel.switch.setChecked(True)
            mw.central_panel.magnifier_switch.setChecked(True)
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def get_main_window(self):
        w = self.parentWidget()
        while w:
            if isinstance(w, QMainWindow):
                return w
            w = w.parentWidget()
        return None

    def reset(self):
        self.points.clear()
        self.drag_index = -1
        self.drag_control = None
        self.curvature = np.zeros((4, 2, 2), np.float32)
        self.remove_inserted_overlay()
        self.tracking_history.clear()
        self.update()

    def has_overlay(self) -> bool:
        return self.overlay_bgra is not None or self.inserted_overlay_is_video

    def current_region(self, points=None) -> core.Region:
        """Build the region described by the current corners and curvature."""
        corners = self.points if points is None else points
        return core.Region(corners, self.curvature, self.curved_enabled)

    def set_curved_enabled(self, enabled: bool):
        self.curved_enabled = bool(enabled)
        self.shape_changed.emit()
        self.update()

    def reset_curvature(self):
        self.curvature = np.zeros((4, 2, 2), np.float32)
        self.shape_changed.emit()
        self.update()

    def insert_image_overlay(self, source, start_frame_index: int):
        """Insert a still creative. ``source`` may be a path or a QPixmap.

        A path is preferred: reading the file directly keeps the creative's
        alpha channel intact at full fidelity instead of round-tripping it
        through a QPixmap.
        """
        self.overlay_source_path = None
        if isinstance(source, str):
            try:
                self.overlay_bgra = core.load_image_bgra(source)
                self.overlay_source_path = source
            except IOError as exc:
                logging.error("Could not read overlay image %s: %s", source, exc)
                return
        else:
            self.overlay_bgra = self.qpixmap_to_bgra(source)

        self.inserted_overlay_is_video = False
        if self.overlay_video_cap:
            self.overlay_video_cap.release()
        self.overlay_video_cap = None
        self.overlay_video_path = None
        self.inserted_overlay_start_frame = start_frame_index
        self.update()

    def insert_video_overlay(self, start_frame_index: int, video_path: str):
        self.overlay_bgra = None
        self.overlay_source_path = video_path
        self.inserted_overlay_is_video = True
        self.overlay_video_path = video_path
        self.inserted_overlay_start_frame = start_frame_index

        if self.overlay_video_cap:
            self.overlay_video_cap.release()
            self.overlay_video_cap = None

        self.overlay_video_cap = cv2.VideoCapture(video_path)
        if not self.overlay_video_cap.isOpened():
            logging.error(f"Cannot open overlay video: {video_path}")
            self.inserted_overlay_is_video = False
            self.overlay_video_cap = None
            return

        self.update()

    def remove_inserted_overlay(self):
        self.overlay_bgra = None
        self.overlay_source_path = None
        self.inserted_overlay_is_video = False
        if self.overlay_video_cap:
            self.overlay_video_cap.release()
            self.overlay_video_cap = None
        self.overlay_video_path = None
        self.inserted_overlay_start_frame = 0
        self.update()

        mw = self.get_main_window()
        if mw and mw.central_panel and mw.central_panel.tracking_mode:
            self.setFocus()

    def update_overlay_video_frame_by_index(self, overlay_frame_idx: int):
        if not self.inserted_overlay_is_video or not self.overlay_video_cap:
            return

        total_overlay_frames = int(self.overlay_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if overlay_frame_idx < 0 or overlay_frame_idx >= total_overlay_frames:
            self.overlay_bgra = None
            self.update()
            return

        self.overlay_video_cap.set(cv2.CAP_PROP_POS_FRAMES, overlay_frame_idx)
        ret, frame = self.overlay_video_cap.read()
        if not ret:
            self.overlay_bgra = None
            self.update()
            return

        self.overlay_bgra = core.to_bgra(frame)
        self.update()

    def styled_overlay(self, base_frame=None, quad=None) -> np.ndarray:
        """The creative with brightness/contrast/colourise applied.

        The render calls the identical helper, so what is previewed is what is
        written to the file.
        """
        if self.overlay_bgra is None:
            return None
        styled = core.apply_brightness_contrast(
            self.overlay_bgra, self.brightness, self.contrast)
        if self.colourise_enabled and base_frame is not None and quad is not None:
            styled = core.apply_colourise(styled, base_frame, quad)
        return styled

    def apply_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        return core.apply_brightness_contrast(img, self.brightness, self.contrast)

    def paintEvent(self, event):
        """Draw the outline and its handles.

        The creative itself is composited into the video frame by the central
        panel using the very same code the renderer uses, so this widget only
        has to draw the interactive furniture on top.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        valid = len(self.points) == 4 and core.is_valid_quad(self.points)
        outline = QColor(0, 255, 0) if valid or len(self.points) < 4 else QColor(255, 140, 0)

        # Outline: the true rendered boundary, curves included.
        if len(self.points) == 4:
            region = self.current_region()
            folded = self.curved_enabled and core.region_is_folded(region)
            if folded:
                outline = QColor(255, 60, 60)
            boundary = core.region_boundary(region, samples=32)

            painter.setPen(QPen(outline, 2))
            display = [self.raw_to_display(x, y) for x, y in boundary]
            for j in range(len(display)):
                x1, y1 = display[j]
                x2, y2 = display[(j + 1) % len(display)]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        elif len(self.points) >= 2:
            painter.setPen(QPen(outline, 2))
            for j in range(len(self.points) - 1):
                x1, y1 = self.raw_to_display(*self.points[j])
                x2, y2 = self.raw_to_display(*self.points[j + 1])
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Bezier control handles, only while curved editing is on.
        if self.curved_enabled and len(self.points) == 4:
            region = self.current_region()
            painter.setPen(QPen(QColor(120, 180, 255, 180), 1, Qt.DashLine))
            for edge in range(4):
                anchor_a = self.raw_to_display(*self.points[edge])
                anchor_b = self.raw_to_display(*self.points[(edge + 1) % 4])
                for slot, anchor in ((0, anchor_a), (1, anchor_b)):
                    cx, cy = self.raw_to_display(*region.controls[edge, slot])
                    painter.drawLine(int(anchor[0]), int(anchor[1]), int(cx), int(cy))

            for edge in range(4):
                for slot in range(2):
                    cx, cy = self.raw_to_display(*region.controls[edge, slot])
                    active = self.drag_control == (edge, slot)
                    painter.setPen(QPen(QColor(0, 255, 255) if active
                                        else QColor(120, 180, 255), 2))
                    painter.setBrush(QBrush(QColor(120, 180, 255, 150)))
                    painter.drawRect(int(cx) - 4, int(cy) - 4, 8, 8)

        # Corner handles last so they sit above everything.
        for i, (rx, ry) in enumerate(self.points):
            disp_x, disp_y = self.raw_to_display(rx, ry)
            selected = i == self.selected_point_index
            painter.setPen(QPen(QColor(0, 255, 0) if selected else QColor(255, 0, 0), 2))

            if i < 4:
                fill_color = QColor(0, 255, 0, 128) if selected else QColor(255, 0, 0, 128)
                painter.setBrush(QBrush(fill_color))
                painter.drawEllipse(int(disp_x) - 5, int(disp_y) - 5, 10, 10)
            else:
                fill_color = QColor(0, 0, 255, 128)
                if selected:
                    painter.setPen(QPen(QColor(0, 255, 255), 2))
                    fill_color = QColor(0, 255, 255, 128)
                painter.setBrush(QBrush(fill_color))
                painter.drawEllipse(int(disp_x) - 6, int(disp_y) - 6, 12, 12)

        painter.end()

    def qpixmap_to_bgra(self, pm: QPixmap) -> np.ndarray:
        qimg = pm.toImage().convertToFormat(QImage.Format_RGBA8888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(w * h * 4)
        rgba = np.frombuffer(ptr, np.uint8).reshape(h, w, 4)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        return bgra

    def bgra_to_qimage(self, bgra: np.ndarray) -> QImage:
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        h, w, _ = rgba.shape
        # QImage wraps the buffer without owning it, and `rgba` is a local that
        # would be collected the moment this returns, leaving the QImage
        # pointing at freed memory. Copy so the QImage owns its pixels.
        return QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()

    def display_to_raw(self, disp_x, disp_y):
        mw = self.get_main_window()
        if not mw or not mw.central_panel or mw.central_panel.prev_frame is None:
            # If we have no frame, just return unchanged
            return disp_x, disp_y

        video_label = mw.central_panel.video_label
        raw_frame = mw.central_panel.prev_frame
        raw_h, raw_w, _ = raw_frame.shape

        disp_w = video_label.width()
        disp_h = video_label.height()

        scale_x = raw_w / float(disp_w)
        scale_y = raw_h / float(disp_h)

        raw_x = disp_x * scale_x
        raw_y = disp_y * scale_y
        return raw_x, raw_y

    def raw_to_display(self, raw_x, raw_y):
        mw = self.get_main_window()
        if not mw or not mw.central_panel or mw.central_panel.prev_frame is None:
            return raw_x, raw_y

        video_label = mw.central_panel.video_label
        raw_frame = mw.central_panel.prev_frame
        raw_h, raw_w, _ = raw_frame.shape

        disp_w = video_label.width()
        disp_h = video_label.height()

        scale_x = disp_w / float(raw_w)
        scale_y = disp_h / float(raw_h)

        disp_x = raw_x * scale_x
        disp_y = raw_y * scale_y
        return disp_x, disp_y

    def toggle_colourise(self, enable: bool):
        self.colourise_enabled = enable
        self.update()  # force repaint 

    def add_fifth_node(self):
        if len(self.points) == 4:
            avg_x = sum(p[0] for p in self.points) / 4.0
            avg_y = sum(p[1] for p in self.points) / 4.0
            self.points.append((avg_x, avg_y))  # 5th node

    def _color_transfer(self, source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """Kept as a thin alias; the implementation lives in lumen_core."""
        return core.color_transfer(source_bgr, target_bgr)

class TitleBar(QWidget):
    def __init__(self, parent=None):
        super(TitleBar, self).__init__(parent)
        self.parent = parent
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #1A1A1A;")
        
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 10, 0)
        self.layout.setSpacing(10)
        
        spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.layout.addItem(spacer)
        
        self.hamburger_btn = QPushButton("☰")
        self.hamburger_btn.setStyleSheet("""
            QPushButton {
                color: white;
                font-size: 20px;
                border: none;
            }
            QPushButton:hover {
                color: #E6E6E6;
            }
        """)
        self.hamburger_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.hamburger_btn.setFixedSize(QSize(40, 40))
        self.hamburger_btn.clicked.connect(self.show_hamburger_menu)
        self.layout.addWidget(self.hamburger_btn, 0, Qt.AlignLeft)
        
        self.menu = QMenu()
        self.menu.setStyleSheet("""
            QMenu {
                background-color: #2A2A2A;
                border: 1px solid #444444;
                padding: 10px;
                border-radius: 10px;
                color: white;
                font-size: 14px;
            }
            QMenu::item {
                padding: 8px 20px;
                margin: 4px 0;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background-color: #444444;
                color: #E6E6E6;
            }
        """)
        
        load_menu = QMenu("Load", self.menu)
        load_menu.addAction(QAction("Base Video", self, triggered=self.parent.upload_base_video))
        load_menu.addAction(QAction("Creative Overlay", self, triggered=self.parent.upload_overlay))
        load_menu.addAction(QAction("Tracking Points", self, triggered=self.parent.load_tracking_points))
        load_menu.addAction(QAction("Project", self, triggered=self.parent.load_project))

        save_menu = QMenu("Save", self.menu)
        save_menu.addAction(QAction("Tracking Points", self, triggered=self.parent.save_tracking_points))
        save_menu.addAction(QAction("AOI Geometry", self, triggered=self.parent.save_aoi_geometry))
        save_menu.addAction(QAction("Project", self, triggered=self.parent.save_project))

        self.menu.addMenu(load_menu)
        self.menu.addMenu(save_menu)
        self.menu.addSeparator()
        detect_action = QAction("Find Target from Image...", self,
                                triggered=self.parent.detect_from_reference)
        detect_action.setToolTip("Upload a picture of the billboard, screen or "
                                 "poster to locate it automatically")
        self.menu.addAction(detect_action)
        self.menu.addSeparator()
        self.menu.addAction(QAction("Help", self))
        self.menu.addAction(QAction("Quit", self, triggered=self.parent.close))
        
        self.center_widget = QWidget()
        self.center_layout = QHBoxLayout()
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = QLabel("KABIRI")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: white;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        self.logo_label = QLabel()
        try:
            self.logo_label.setPixmap(QPixmap("logo.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except:
            logging.warning("logo.png not found. Skipping logo display.")
            self.logo_label.setText("Logo")
            self.logo_label.setStyleSheet("color: white; font-size: 14px;")
        self.logo_label.setAlignment(Qt.AlignCenter)
        
        self.center_layout.addWidget(self.title_label)
        self.center_layout.addWidget(self.logo_label)
        self.center_widget.setLayout(self.center_layout)
        self.layout.addWidget(self.center_widget, 1, Qt.AlignCenter)
        
        self.right_layout = QHBoxLayout()
        self.right_layout.setSpacing(20)
        
        self.minimize_btn = QPushButton("–")
        self.minimize_btn.setStyleSheet("""
            QPushButton {
                color: white;
                font-size: 20px;
                border: none;
            }
            QPushButton:hover {
                color: #E6E6E6;
            }
        """)
        self.minimize_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.minimize_btn.setFixedSize(QSize(40, 40))
        self.minimize_btn.clicked.connect(self.parent.showMinimized)
        self.right_layout.addWidget(self.minimize_btn, 0, Qt.AlignRight)
        
        self.close_btn = QPushButton("✕")
        self.close_btn.setStyleSheet("""
            QPushButton {
                color: white;
                font-size: 20px;
                border: none;
            }
            QPushButton:hover {
                color: #E6E6E6;
            }
        """)
        self.close_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.close_btn.setFixedSize(QSize(40, 40))
        self.close_btn.clicked.connect(self.parent.close)
        self.right_layout.addWidget(self.close_btn, 0, Qt.AlignRight)
        
        self.layout.addLayout(self.right_layout)
        self.setLayout(self.layout)
        self.setFixedHeight(60)
        
    def show_hamburger_menu(self):
        pos = self.hamburger_btn.mapToGlobal(self.hamburger_btn.rect().bottomRight())
        pos.setX(pos.x() + 25)
        self.menu.exec_(pos)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_offset = self.parent.pos() - event.globalPos()
        super(TitleBar, self).mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.parent.move(event.globalPos() + self.drag_offset)

class LeftColumn(QWidget):
    drawClicked = pyqtSignal(bool)
    renderClicked = pyqtSignal()
    goToFrameClicked = pyqtSignal()
    brightnessClicked = pyqtSignal()
    contrastClicked = pyqtSignal()
    colouriseClicked = pyqtSignal(bool)
    curveClicked = pyqtSignal(bool)
    clearClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super(LeftColumn, self).__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #1A1A1A;")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(15)

        items = [
            ("🖱", "Draw"),
            ("📼", "Render"),
            ("👀", "Frame"),
            ("🌞", "Brightness"),
            ("⛅", "Contrast"),
            ("🎨", "Colourise"),
            ("〰", "Curve"),
            ("❌", "Clear"),
        ]

        self.icons = []
        for (icon_char, text) in items:
            icon_widget = IconWidget(icon_char, text)
            layout.addWidget(icon_widget, 0, Qt.AlignHCenter)
            icon_widget.mousePressEvent = self.make_icon_click_handler(icon_widget, icon_widget.mousePressEvent)
            self.icons.append(icon_widget)

        layout.addStretch()
        self.setLayout(layout)
        self.setFixedWidth(80)

    # Toggles that stand for a mode rather than a one-off action. Clearing
    # their highlight when a different icon is clicked would leave the lamp
    # off while the mode was still on, so they keep their state.
    STATEFUL_ICONS = ("Draw", "Colourise", "Curve")

    def make_icon_click_handler(self, icon_widget, original_mouse_press):
        def handler(event):
            # First, handle the visual selection logic for the icons
            for iw in self.icons:
                if iw != icon_widget and iw.meaning_label.text() not in self.STATEFUL_ICONS:
                    # Deselect the one-shot icons
                    iw.set_selected(False)
            
            # Toggle the clicked icon's state
            icon_widget.set_selected(not icon_widget.selected)
            original_mouse_press(event)

            # Get the icon's text and its new selection state
            text = icon_widget.meaning_label.text()
            is_selected = icon_widget.selected

            # --- Emit signals instead of performing actions directly ---
            if text == "Draw":
                self.drawClicked.emit(is_selected)
            elif text == "Render":
                self.renderClicked.emit()
                icon_widget.set_selected(False)  # This is an instant action, so deselect it
            elif text == "Frame":
                if is_selected:
                    self.goToFrameClicked.emit()
                icon_widget.set_selected(False)  # Instant action, deselect
            elif text == "Brightness":
                if is_selected:
                    self.brightnessClicked.emit()
                icon_widget.set_selected(False)  # Instant action, deselect
            elif text == "Contrast":
                if is_selected:
                    self.contrastClicked.emit()
                icon_widget.set_selected(False)  # Instant action, deselect
            elif text == "Colourise":
                self.colouriseClicked.emit(is_selected) # This is a state, so don't deselect
            elif text == "Curve":
                self.curveClicked.emit(is_selected)  # Also a state, keep it lit
            elif text == "Clear":
                if is_selected:
                    self.clearClicked.emit()
                icon_widget.set_selected(False)  # Instant action, deselect

            # After any icon interaction, ensure focus is correct.
            main_window = self.get_main_window()
            if main_window and main_window.central_panel.tracking_mode:
                main_window.central_panel.tracking_overlay.setFocus()

        return handler

    def get_main_window(self):
        w = self.parentWidget()
        while w:
            if isinstance(w, QMainWindow):
                return w
            w = w.parentWidget()
        return None

class CentralPanel(QWidget):
    def __init__(self, parent=None):
        super(CentralPanel, self).__init__(parent)

        self.tracking_mode = False
        self.cap = None
        self.playing = False
        self.prev_frame = None
        self.fps = 30
        self.tracking_history = {}  # optional mirror of tracking_overlay
        self.kalman_filters = []

        # The single source of truth for "which frame is on screen". Deriving
        # it from CAP_PROP_POS_MSEC in one place and CAP_PROP_POS_FRAMES in
        # another used to make them disagree by a frame, which quietly
        # misfiled tracking data and offset video creatives.
        self.current_frame_index = 0
        self.total_frames = 0

        # Feature-based planar tracker, rebuilt whenever the shape is re-drawn.
        self.tracker = None

        # 1) The container for video + overlay
        self.video_container = QWidget(self)
        self.video_layout = QGridLayout(self.video_container)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(0)

        self.video_label = QLabel(self.video_container)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setScaledContents(True)
        self.video_layout.addWidget(self.video_label, 0, 0)

        self.tracking_overlay = TrackingOverlay(self.video_container)
        self.tracking_overlay.setFocusPolicy(Qt.StrongFocus)
        self.video_layout.addWidget(self.tracking_overlay, 0, 0)
        self.tracking_overlay.raise_()
        self.tracking_overlay.hide()
        self.tracking_overlay.shape_changed.connect(self.on_shape_changed)
        self.tracking_overlay.selection_changed.connect(self.update_magnifier)

        # Frame label
        self.frame_label = QLabel("Frame: 0", self.video_container)
        self.frame_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                background-color: #1A1A1A;
                padding: 5px;
            }
        """)
        self.frame_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.frame_label.setFixedWidth(120)

        # Tracking switch
        self.switch = QSwitch(self)
        self.switch.setToolTip("Toggle Tracking ON/OFF")
        self.switch.toggled.connect(self.on_tracking_toggled)

        self.tracking_label = QLabel("Tracking Disabled", self.video_container)
        self.tracking_label.setStyleSheet("""
            QLabel {
                color: red;
                font-size: 14px;
                background-color: #1A1A1A;
                padding: 5px;
            }
        """)
        self.tracking_label.setMinimumWidth(150)

        # Magnifier switch + label + widget
        self.magnifier_switch = QSwitch(self)
        self.magnifier_switch.setToolTip("Toggle Magnifier ON/OFF")
        self.magnifier_switch.toggled.connect(self.on_magnifier_toggled)

        self.magnifier_label = QLabel("Magnifier Off", self.video_container)
        self.magnifier_label.setStyleSheet("""
            QLabel {
                color: red;
                font-size: 14px;
                background-color: #1A1A1A;
                padding: 5px;
                min-width: 150px;
            }
        """)
        self.magnifier_label.show()
        self.magnifier = MagnifierWidget(self.video_container)
        self.magnifier.hide()

        # Playback controls
        self.controls_frame = QFrame(self.video_container)
        self.controls_frame.setStyleSheet("""
            QFrame {
                background-color: #1A1A1A;
                padding: 10px;
            }
        """)
        self.controls_frame.setFixedHeight(60)

        self.rewind_btn = HoverButton()
        self.rewind_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.rewind_btn.setIconSize(QSize(24, 24))
        self.rewind_btn.setStyleSheet(self.button_style())
        self.rewind_btn.clicked.connect(self.on_rewind)

        self.play_pause_btn = HoverButton()
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_btn.setIconSize(QSize(24, 24))
        self.play_pause_btn.setStyleSheet(self.button_style())
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)

        self.forward_btn = HoverButton()
        self.forward_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.forward_btn.setIconSize(QSize(24, 24))
        self.forward_btn.setStyleSheet(self.button_style())
        self.forward_btn.clicked.connect(self.on_forward)

        self.del_btn = HoverButton("D")
        self.del_btn.setStyleSheet(self.button_style())
        self.del_btn.setToolTip("Delete shape (D key)")
        self.del_btn.setFixedSize(40, 40)
        self.del_btn.clicked.connect(self.on_delete_shape)

        self.copy_btn = HoverButton("C")
        self.copy_btn.setStyleSheet(self.button_style())
        self.copy_btn.setToolTip("Copy shape from previous frame (C key)")
        self.copy_btn.setFixedSize(40, 40)
        self.copy_btn.clicked.connect(self.on_copy_shape)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setStyleSheet("""
            QSlider {
                background: transparent;
            }
            QSlider::groove:horizontal {
                background: #444444;       /* dark grey groove */
                height: 6px;
                border-radius: 3px;
                margin: 0 10px;            /* some side margin */
            }
            QSlider::handle:horizontal {
                background: white;         /* white handle */
                border: 2px solid #AAAAAA; /* or #ccc */
                width: 14px;
                margin: -4px 0;           /* center handle */
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #777777;       /* left side fill color */
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #222222;       /* right side behind handle */
                border-radius: 3px;
            }
        """)

        self.timestamp_label = QLabel("00:00 / 00:00")
        self.timestamp_label.setStyleSheet("color: white; font-size: 14px; padding: 3px;")
        self.timestamp_label.setAlignment(Qt.AlignCenter)

        # Arrange the main controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.rewind_btn)
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.forward_btn)
        controls_layout.addWidget(self.slider, 1)
        controls_layout.addWidget(self.timestamp_label)
        controls_layout.addWidget(self.copy_btn)
        controls_layout.addWidget(self.del_btn)
        self.controls_frame.setLayout(controls_layout)

        # Final panel layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_layout.addWidget(self.video_container, stretch=1)
        self.setLayout(self.main_layout)

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_frame)

        self.frame_timer = QTimer()
        self.frame_timer.setInterval(100)
        self.frame_timer.timeout.connect(self.update_frame_label)
        self.frame_timer.start()

        # Focus
        self.setFocusPolicy(Qt.StrongFocus)
        self.tracking_overlay.setFocusPolicy(Qt.StrongFocus)

        logging.debug("CentralPanel initialized.")

    def button_style(self):
        return """
            QPushButton {
                background: #3C3C3C;
                border: none;
                border-radius: 20px;
                width: 40px;
                height: 40px;
                color: white;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #2A2A2A;
            }
        """

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        # If user presses D/C, same as the button
        if event.key() == Qt.Key_D:
            self.on_delete_shape()
            self.tracking_overlay.setFocus()
        elif event.key() == Qt.Key_C:
            self.on_copy_shape()
            self.tracking_overlay.setFocus()

    def on_delete_shape(self):
        current_frame_index = self.get_current_frame_index()
        # Delete from tracking_history
        if current_frame_index in self.tracking_overlay.tracking_history:
            del self.tracking_overlay.tracking_history[current_frame_index]
        if current_frame_index in self.tracking_history:
            del self.tracking_history[current_frame_index]
        # Clear points if on that frame
        if self.get_current_frame_index() == current_frame_index:
            self.tracking_overlay.points.clear()
            self.tracking_overlay.update()
        logging.debug(f"Deleted shape for frame {current_frame_index}")

    def on_copy_shape(self):
        current_frame_index = self.get_current_frame_index()
        if current_frame_index <= 0:
            return
        prev_index = current_frame_index - 1
        if prev_index in self.tracking_overlay.tracking_history:
            pts = self.tracking_overlay.tracking_history[prev_index][:]
            self.tracking_overlay.tracking_history[current_frame_index] = pts
            self.tracking_history[current_frame_index] = pts
            if self.get_current_frame_index() == current_frame_index:
                self.tracking_overlay.points = pts
                self.tracking_overlay.update()
            logging.debug(f"Copied shape from frame {prev_index} to {current_frame_index}")

    def on_rewind(self):
        if self.cap and self.cap.isOpened():
            # When going backward, always disable tracking to prevent overwrites
            self.disable_auto_tracking()

            current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            new_frame_index = max(0, current_frame_index - 1)

            if new_frame_index != current_frame_index:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_index)
                self.read_frame()

    def on_forward(self):
        if self.cap and self.cap.isOpened():
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Get the index of the currently displayed frame
            current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            # Calculate the target frame, ensuring it doesn't exceed the total
            new_frame_index = min(total_frames - 1, current_frame_index + 1)

            # Only proceed if there is a change
            if new_frame_index != current_frame_index:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_index)
                self.read_frame()
                if self.tracking_mode:
                    self.tracking_overlay.setFocus()

    def jump_to_frame(self, frame_index):
        if not self.cap or not self.cap.isOpened():
            return

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0:
            frame_index = 0
        elif frame_index >= total_frames:
            frame_index = total_frames - 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.playing = False
        self.timer.stop()
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        
        # Turn off tracking on jump.
        self.switch.setChecked(False)
        self.tracking_mode = False
        self.reset_tracker()

        self.read_frame()

        # If there is tracking history for this frame, load it; otherwise clear.
        if frame_index in self.tracking_history:
            self.tracking_overlay.points = self.tracking_history[frame_index][:]
        else:
            self.tracking_overlay.points.clear()

        self.tracking_overlay.update()

        # (Optional) Set focus if you need it.
        if self.tracking_mode:
            self.tracking_overlay.setFocus()

    def toggle_play_pause(self):
        self.playing = not self.playing
        print(f"Toggling play: {self.playing}")
        if self.playing:
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            interval_ms = max(1, int(1000 / self.fps))
            print(f"Starting timer with interval={interval_ms} ms, FPS={self.fps}")
            self.timer.start(interval_ms)
        else:
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()

        # <<<< Add this line >>>>
        if self.tracking_mode:
            self.tracking_overlay.setFocus()

    def disable_auto_tracking(self):
        """Turns off the tracking mode and updates the UI switch."""
        self.tracking_mode = False
        self.switch.setChecked(False)
        self.tracking_label.setText("Tracking Disabled")
        self.tracking_label.setStyleSheet("""
            QLabel {
                color: red;
                font-size: 14px;
                background-color: #1A1A1A;
                padding: 5px;
                min-width: 150px;
            }
        """)

    def on_shape_changed(self):
        # A hand edit invalidates the tracker's anchor, so drop it; it will be
        # rebuilt from the corrected shape on the next tracked frame.
        self.reset_tracker()
        if len(self.tracking_overlay.points) == 4 and self.prev_frame is not None:
            self.refresh_display()
        self.update_magnifier_dimensions()
        self.update_magnifier()

    def refresh_display(self):
        """Redraw the current frame with the creative composited in.

        Used when the shape or a creative setting changes while playback is
        paused, so the preview updates without having to step a frame.
        """
        if self.prev_frame is None:
            return
        overlay = self.tracking_overlay
        display_frame = self.prev_frame
        if len(overlay.points) == 4 and overlay.has_overlay():
            styled = overlay.styled_overlay(self.prev_frame, overlay.points)
            if styled is not None:
                region = overlay.current_region()
                if core.is_valid_region(region):
                    display_frame = core.composite_region(
                        self.prev_frame, styled, region)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def get_current_frame_index(self):
        """The index of the frame currently on screen."""
        return self.current_frame_index

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.playing = False
            return

        current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.current_frame_index = current_frame_index
        if not self.slider.isSliderDown():
            self.slider.setValue(current_frame_index)

        overlay = self.tracking_overlay

        # --- TRACKING ---------------------------------------------------
        # The tracker follows the texture across the whole region and fits a
        # homography to it; the Kalman filters then smooth the result and
        # carry the shape through brief dropouts.
        if self.tracking_mode and len(overlay.points) == 4:
            if not self.kalman_filters:
                self.kalman_filters = core.make_filters(overlay.points)
            if self.tracker is None and self.prev_frame is not None:
                self.tracker = core.PlanarTracker(self.prev_frame, overlay.points)

            measurement = None
            if self.tracker is not None:
                result = self.tracker.track(frame)
                if result.ok:
                    measurement = result.quad
                else:
                    logging.debug("Tracking step failed on frame %d: %s",
                                  current_frame_index, result.reason)

            smoothed = core.smooth_quad(self.kalman_filters, measurement)
            overlay.points = [(float(x), float(y)) for x, y in smoothed]

            overlay.tracking_history[current_frame_index] = overlay.points[:]
            self.tracking_history[current_frame_index] = overlay.points[:]
        elif not self.tracking_mode:
            # Tracking off: show whatever was recorded for this frame.
            self.kalman_filters = []
            self.tracker = None
            if current_frame_index in self.tracking_history:
                overlay.points = self.tracking_history[current_frame_index][:]
            else:
                overlay.points = []

        # If an overlay video is inserted, advance it to the matching frame.
        if overlay.inserted_overlay_is_video:
            overlay_index = current_frame_index - overlay.inserted_overlay_start_frame
            overlay.update_overlay_video_frame_by_index(overlay_index)

        # --- DISPLAY ----------------------------------------------------
        # Composite through the same code path the renderer uses, so the
        # preview is a true preview rather than a lookalike.
        self.prev_frame = frame.copy()
        display_frame = frame
        if len(overlay.points) == 4 and overlay.has_overlay():
            styled = overlay.styled_overlay(frame, overlay.points)
            if styled is not None:
                region = overlay.current_region()
                if core.is_valid_region(region):
                    display_frame = core.composite_region(frame, styled, region)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h_frame, w_frame, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w_frame, h_frame, ch * w_frame,
                       QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

        self.update_magnifier()

    def update_frame_label(self):
        if self.cap and self.cap.isOpened():
            pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_idx = int(pos_msec / (1000.0 / self.fps)) if self.fps>0 else 0
            self.frame_label.setText(f"Frame: {frame_idx}")

            current_time = ms_to_mmss(int(pos_msec))
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_count > 0 and self.fps > 0:
                total_duration_ms = (frame_count / self.fps) * 1000
                total_str = ms_to_mmss(int(total_duration_ms))
            else:
                total_str = "00:00"
            self.timestamp_label.setText(f"{current_time} / {total_str}")
        else:
            self.frame_label.setText("Frame: 0")
            self.timestamp_label.setText("00:00 / 00:00")

    def resizeEvent(self, event):
        super(CentralPanel, self).resizeEvent(event)
        width = self.width()
        height = int(width / 16 * 9)
        if height > self.height():
            height = self.height()
            width = int(height * 16 / 9)

        self.video_container.setGeometry(0, 0, width, height)
        self.frame_label.move(10, 10)
        self.frame_label.raise_()

        switch_x = width - self.switch.width() - 20
        switch_y = 10
        self.switch.move(switch_x, switch_y)
        self.switch.raise_()

        label_w = self.tracking_label.sizeHint().width()
        label_h = self.tracking_label.sizeHint().height()
        self.tracking_label.move(switch_x - label_w - 10, switch_y + (self.switch.height() - label_h)//2)
        self.tracking_label.raise_()

        mag_label_y = switch_y + self.switch.height() + 10
        self.magnifier_label.move(switch_x - label_w - 10, mag_label_y)
        self.magnifier_label.raise_()

        self.magnifier_switch.move(switch_x, mag_label_y)
        self.magnifier_switch.raise_()

        self.magnifier.move(10, 60)
        self.magnifier.raise_()

        controls_width = int(width*0.6)
        controls_x = (width - controls_width)//2
        controls_y = height - self.controls_frame.height() - 20
        self.controls_frame.setGeometry(controls_x, controls_y, controls_width, self.controls_frame.height())
        self.controls_frame.raise_()

    def toggle_tracking(self, checked: bool):
        if checked:
            if len(self.tracking_overlay.points) == 4:
                self.tracking_mode = True
            else:
                self.switch.setChecked(False)
                QMessageBox.information(
                    self, "Cannot Enable Tracking",
                    "Draw all four corners of the area before turning tracking on."
                )
        else:
            self.tracking_mode = False

    def on_tracking_toggled(self, on: bool):
        self.tracking_mode = on
        if on:
            self.tracking_label.setText("Tracking Enabled")
            self.tracking_label.setStyleSheet("""
                QLabel {
                    color: limegreen;
                    font-size: 14px;
                    background-color: #1A1A1A;
                    padding: 5px;
                    min-width: 150px;
                }
            """)
        else:
            self.tracking_label.setText("Tracking Disabled")
            self.tracking_label.setStyleSheet("""
                QLabel {
                    color: red;
                    font-size: 14px;
                    background-color: #1A1A1A;
                    padding: 5px;
                    min-width: 150px;
                }
            """)
            self.tracking_overlay.points.clear()
            self.tracking_overlay.update()

    def set_tracking_toggle_state(self, on: bool):
        if on:
            self.tracking_toggle.setCheckState(Qt.Checked)
            self.tracking_status_label.setText("Tracking Enabled")
            self.tracking_status_label.setStyleSheet("""
                QLabel {
                    color: limegreen;
                    font-size: 14px;
                    background-color: #1A1A1A;
                    padding: 5px;
                }
            """)
        else:
            self.tracking_toggle.setCheckState(Qt.Unchecked)
            self.tracking_status_label.setText("Tracking Disabled")
            self.tracking_status_label.setStyleSheet("""
                QLabel {
                    color: red;
                    font-size: 14px;
                    background-color: #1A1A1A;
                    padding: 5px;
                }
            """)

    def track_with_optical_flow(self, old_frame, new_frame, old_points):
        """Advance a quad by one frame.

        Superseded by :class:`lumen_core.PlanarTracker`, which tracks the
        texture across the whole region instead of the four corner points and
        fits a homography to it. Kept as a one-shot convenience wrapper.
        """
        if len(old_points) != 4:
            return None
        tracker = core.PlanarTracker(old_frame, old_points)
        result = tracker.track(new_frame)
        if not result.ok:
            return None
        return [(float(x), float(y)) for x, y in result.quad]

    def reset_tracker(self):
        """Re-anchor tracking to the shape as it now stands.

        Called whenever the user edits the shape by hand: without this the
        tracker would keep propagating from the pre-edit corners and drag the
        correction straight back out again.
        """
        self.tracker = None
        self.kalman_filters = []

    def update_magnifier_dimensions(self):
        shape_pts = self.tracking_overlay.points
        if len(shape_pts) < 2:
            return
        # Skip auto-resizing if the user has manually resized.
        if self.magnifier.resizing or self.magnifier.user_resized:
            return
        xs = [p[0] for p in shape_pts]
        ys = [p[1] for p in shape_pts]
        w_pts = max(xs) - min(xs)
        h_pts = max(ys) - min(ys)
        if w_pts < 1 or h_pts < 1:
            return

        new_aspect = float(w_pts) / float(h_pts)
        if new_aspect >= 1.0:
            computed_w = 200
            computed_h = int(computed_w / new_aspect)
        else:
            computed_h = 200
            computed_w = int(computed_h * new_aspect)
        computed_w = max(150, computed_w)
        computed_h = max(150, computed_h)

        current_w = self.magnifier.width()
        current_h = self.magnifier.height()
        new_w = current_w
        new_h = current_h
        if computed_w > current_w:
            new_w = computed_w
        if computed_h > current_h:
            new_h = computed_h
        if (new_w != current_w) or (new_h != current_h):
            self.magnifier.setFixedSize(new_w, new_h)

    def update_magnifier(self):
        if not self.magnifier_switch.isChecked():
            return

        if self.prev_frame is None:
            self.magnifier.setData(None, [], selected_index=-1)
            return

        base_bgr = self.prev_frame.copy()
        shape_pts = self.tracking_overlay.points[:]
        selected_idx = self.tracking_overlay.selected_point_index

        # Now we pass selected_index so the correct quadrant is drawn in green
        self.magnifier.setData(
            base_frame=base_bgr,
            overlay_points=shape_pts,
            selected_index=selected_idx
        )

    def load_video(self, file_path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(file_path)
        if self.cap.isOpened():
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setEnabled(True)
            self.slider.setRange(0, total_frames - 1)
            self.slider.setValue(0)
            self.slider.valueChanged.connect(self.on_slider_scrub)
            self.slider.sliderPressed.connect(self.on_slider_pressed)
            self.slider.sliderReleased.connect(self.on_slider_released)
            self.slider.valueChanged.connect(self.on_slider_value_changed)
        else:
            logging.error(f"Could not open video: {file_path}")
            return

        self.current_video_path = file_path
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = actual_fps if actual_fps and actual_fps > 0 else 30

        self.total_frames = total_frames
        self.current_frame_index = 0
        self.reset_tracker()

        ret, frame = self.cap.read()
        if ret:
            self.prev_frame = frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.read_frame()
        else:
            logging.error("Failed to read initial frame from video.")
            return

        self.playing = False
        self.timer.stop()
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        logging.debug(f"Loaded {file_path}, FPS={self.fps}")

        self.tracking_overlay.reset()

    def enable_tracking_mode(self):
        """
        Turn on the overlay’s mouse + keyboard interactions,
        then explicitly setFocus() so that 1..5 keys & arrow keys
        *always* go to the overlay.
        """
        self.tracking_overlay.set_tracking_enabled(True)
        self.tracking_overlay.show()
        self.tracking_overlay.raise_()
        self.tracking_overlay.setFocus()  # <--- ADDED LINE

    def disable_tracking_mode(self):
        """
        Turn off the overlay’s interactions. The user can’t draw or
        move corners with mouse or keys while disabled.
        """
        self.tracking_overlay.set_tracking_enabled(False)

    def on_magnifier_toggled(self, on: bool):
        if on:
            self.magnifier_label.setText("Magnifier On")
            self.magnifier_label.setStyleSheet("""
                QLabel {
                    color: limegreen;
                    font-size: 14px;
                    background-color: #1A1A1A;
                    padding: 5px;
                    min-width: 150px;
                }
            """)
            self.magnifier.show()
            self.update_magnifier()
        else:
            self.magnifier_label.setText("Magnifier Off")
            self.magnifier_label.setStyleSheet("""
                QLabel {
                    color: red;
                    font-size: 14px;
                    background-color: #1A1A1A;
                    padding: 5px;
                    min-width: 150px;
                }
            """)
            self.magnifier.hide()
        self.magnifier_label.show()

        if self.tracking_mode:
            self.tracking_overlay.setFocus()

    def auto_enable_tracking_if_shape_ready(self):
        if len(self.tracking_overlay.points) == 4:
            self.set_tracking_toggle_state(True)
            self.tracking_mode = True

    def save_tracking_points_for_frame(self, frame_index, points):
        try:
            with open('tracking_history.json', 'a') as f:
                data = {'frame': frame_index, 'points': points}
                f.write(json.dumps(data) + '\n')
            logging.debug(f"Saved tracking points for frame {frame_index}")
        except Exception as e:
            logging.error(f"Failed to save tracking points for frame {frame_index}: {e}")

    def on_slider_pressed(self):
        self.playing = False
        self.timer.stop()
        # Store the frame position *before* the user starts scrubbing
        self.frame_before_scrub = self.get_current_frame_index()

    def on_slider_released(self):
        target_frame = self.slider.value()
        
        # If the user scrubbed backward, disable tracking
        if target_frame < self.frame_before_scrub:
            self.disable_auto_tracking()
            
        self.jump_to_frame(target_frame)

    def on_slider_value_changed(self, value):
        self.frame_label.setText(f"Frame: {value}")

    def on_slider_scrub(self, value):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.read_frame()

class RenderSettings:
    """A self-contained snapshot of everything a render needs.

    The worker runs on another thread, so it must not reach back into live
    widgets: reading a QPixmap or a Qt-owned object off the GUI thread is not
    supported by Qt and can crash or return garbage. Everything is copied into
    plain Python and NumPy here, on the GUI thread, before the worker starts.
    """

    def __init__(self, base_video_path, out_path, start_frame, end_frame,
                 scale_factor, fps, history, curvature, curved,
                 overlay_bgra=None, overlay_video_path=None,
                 overlay_start_frame=0, brightness=0, contrast=1.0,
                 colourise=False, include_audio=True):
        self.base_video_path = base_video_path
        self.out_path = out_path
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.scale_factor = float(scale_factor)
        self.fps = float(fps)
        self.history = history               # {frame_index: 4 corners}
        self.curvature = np.array(curvature, np.float32)
        self.curved = bool(curved)
        self.overlay_bgra = overlay_bgra     # a detached BGRA copy
        self.overlay_video_path = overlay_video_path
        self.overlay_start_frame = int(overlay_start_frame)
        self.brightness = brightness
        self.contrast = contrast
        self.colourise = colourise
        self.include_audio = include_audio


class RenderWorker(QObject):
    """Composites the creative into the base video, off the GUI thread."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # Emits a final status message or error

    def __init__(self, settings: RenderSettings):
        super().__init__()
        self.settings = settings
        self.is_canceled = False

    def run(self):
        """The main rendering loop."""
        settings = self.settings
        base_cap = overlay_cap = out_writer = None
        try:
            base_cap = cv2.VideoCapture(settings.base_video_path)
            if not base_cap.isOpened():
                raise IOError("Could not open the base video for rendering.")

            scale = settings.scale_factor
            width = int(int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale)
            height = int(int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

            # Fill the gaps between tracked frames instead of holding the last
            # known position, which used to freeze the insert mid-shot.
            dense = core.interpolate_tracking(
                settings.history, settings.start_frame, settings.end_frame)
            if not dense:
                self.finished.emit("Completed (No tracking data to apply)")
                return

            out_writer = cv2.VideoWriter(
                settings.out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                settings.fps, (width, height))
            if not out_writer.isOpened():
                raise IOError(f"Could not open {settings.out_path} for writing.")

            overlay_fps = 0.0
            if settings.overlay_video_path:
                overlay_cap = cv2.VideoCapture(settings.overlay_video_path)
                if overlay_cap.isOpened():
                    overlay_fps = overlay_cap.get(cv2.CAP_PROP_FPS) or settings.fps
                else:
                    logging.error("Could not open the overlay video; "
                                  "rendering without it.")
                    overlay_cap = None

            # Seek once, then read straight through. Seeking per frame is slow
            # and, on long-GOP codecs, lands on the nearest keyframe rather
            # than the frame asked for.
            if settings.start_frame > 0:
                base_cap.set(cv2.CAP_PROP_POS_FRAMES, settings.start_frame)

            overlay_frame = None
            overlay_cursor = -1
            overlay_exhausted = False
            frame_counter = 0

            for frame_idx in range(settings.start_frame, settings.end_frame + 1):
                if self.is_canceled:
                    break

                ok, base_frame = base_cap.read()
                if not ok:
                    break
                if scale != 1.0:
                    base_frame = cv2.resize(base_frame, (width, height),
                                            interpolation=cv2.INTER_AREA)

                quad = dense.get(frame_idx)
                if quad is not None:
                    creative = settings.overlay_bgra

                    if overlay_cap is not None and not overlay_exhausted:
                        # Map by time, not by frame count, so a creative shot
                        # at a different frame rate plays at the right speed.
                        elapsed = frame_idx - settings.overlay_start_frame
                        if elapsed >= 0:
                            wanted = int(round(elapsed * overlay_fps / settings.fps))
                            while overlay_cursor < wanted:
                                got, next_frame = overlay_cap.read()
                                if not got:
                                    overlay_exhausted = True
                                    break
                                overlay_frame = next_frame
                                overlay_cursor += 1
                            creative = overlay_frame

                    if creative is not None:
                        region = core.Region(np.float32(quad) * scale,
                                             settings.curvature, settings.curved)
                        styled = core.apply_brightness_contrast(
                            creative, settings.brightness, settings.contrast)
                        if settings.colourise:
                            styled = core.apply_colourise(
                                styled, base_frame, region.corners)
                        if core.is_valid_region(region):
                            base_frame = core.composite_region(
                                base_frame, styled, region, in_place=True)

                out_writer.write(base_frame)
                frame_counter += 1
                self.progress.emit(frame_counter)

            base_cap.release()
            out_writer.release()
            base_cap = out_writer = None
            if overlay_cap:
                overlay_cap.release()
                overlay_cap = None

            if self.is_canceled:
                self.finished.emit("Canceled")
                return

            self.finished.emit(self._attach_audio())

        except Exception as e:
            logging.exception("Render worker failed")
            self.finished.emit(f"Error: {e}")
        finally:
            for handle in (base_cap, overlay_cap, out_writer):
                if handle is not None:
                    handle.release()

    def _attach_audio(self) -> str:
        """Graft the source audio back on, if ffmpeg is around to do it.

        VideoWriter can only write pictures, so the render is silent until
        ffmpeg copies the original audio across. Without ffmpeg the silent file
        still stands, and the status says so rather than failing the render.
        """
        settings = self.settings
        if not settings.include_audio:
            return "Completed"
        if not core.has_ffmpeg():
            return "Completed (no audio: ffmpeg not found)"

        base, ext = os.path.splitext(settings.out_path)
        temp_path = f"{base}.audio{ext or '.mp4'}"
        duration = ((settings.end_frame - settings.start_frame + 1)
                    / max(settings.fps, 1e-6))

        ok = core.remux_audio(settings.out_path, settings.base_video_path,
                              temp_path,
                              start_time=settings.start_frame / max(settings.fps, 1e-6),
                              duration=duration)
        if not ok:
            return "Completed (no audio: the source may have none)"

        try:
            os.replace(temp_path, settings.out_path)
        except OSError as exc:
            logging.warning("Could not swap in the audio version: %s", exc)
            return "Completed (no audio: could not replace the silent file)"
        return "Completed"

    def cancel(self):
        self.is_canceled = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logging.debug("MainWindow initiated")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setMouseTracking(True)

        self.aspect_ratio = 19 / 9
        self.setMinimumSize(1200, int(1200 / self.aspect_ratio))

        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #D3D3D3;")
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.title_bar = TitleBar(self)
        title_sep = QFrame()
        title_sep.setFrameShape(QFrame.HLine)
        title_sep.setFrameShadow(QFrame.Plain)
        title_sep.setStyleSheet("background-color: white; max-height: 1px;")

        content_frame = QFrame()
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.left_col = LeftColumn(self)
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.VLine)
        separator_line.setFrameShadow(QFrame.Plain)
        separator_line.setStyleSheet("background-color: white; width: 1px;")

        self.central_panel = CentralPanel(self)

        self.overlay_col = QWidget(self)
        self.overlay_col.setStyleSheet("background-color: #1A1A1A;")
        self.overlay_col.setFixedWidth(250)
        self.overlay_layout = QVBoxLayout()
        self.overlay_layout.setContentsMargins(10, 10, 10, 10)
        self.overlay_layout.setSpacing(10)

        library_title = QLabel("Library")
        library_title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        library_title.setAlignment(Qt.AlignCenter)
        self.overlay_layout.addWidget(library_title)

        library_spacer = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.overlay_layout.addSpacerItem(library_spacer)
        self.overlay_layout.addStretch()

        self.overlay_col.setLayout(self.overlay_layout)

        content_layout.addWidget(self.left_col)
        content_layout.addWidget(separator_line)
        content_layout.addWidget(self.central_panel)
        content_layout.addWidget(self.overlay_col)

        content_frame.setLayout(content_layout)

        main_layout.addWidget(self.title_bar, 0)
        main_layout.addWidget(title_sep, 0)
        main_layout.addWidget(content_frame, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self._connect_signals()
        self.showMaximized()
        logging.debug("MainWindow shown maximized")

        self.inserted_overlay_widget = None

    def _connect_signals(self):
        """Connects signals from child widgets to main window slots."""
        # Left Column connections
        self.left_col.drawClicked.connect(self.on_draw_clicked)
        self.left_col.renderClicked.connect(self.render_video)
        self.left_col.goToFrameClicked.connect(self.on_goto_frame_clicked)
        self.left_col.brightnessClicked.connect(self.on_brightness_clicked)
        self.left_col.contrastClicked.connect(self.on_contrast_clicked)
        self.left_col.colouriseClicked.connect(self.on_colourise_clicked)
        self.left_col.curveClicked.connect(self.toggle_curved_edges)
        self.left_col.clearClicked.connect(self.on_clear_clicked)

    def on_draw_clicked(self, is_selected):
        """Slot that handles the 'Draw' icon being clicked."""
        if is_selected:
            self.central_panel.enable_tracking_mode()
        else:
            self.central_panel.disable_tracking_mode()

    def on_goto_frame_clicked(self):
        """Slot that opens a dialog to jump to a specific frame."""
        if not self.central_panel.cap or not self.central_panel.cap.isOpened():
            QMessageBox.warning(self, "Action Failed", "Please load a base video first.")
            return
            
        total_frames = int(self.central_panel.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num, ok = QInputDialog.getInt(self, "Go to Frame", "Enter frame number:", 0, 0, max(0, total_frames - 1))
        if ok:
            self.central_panel.jump_to_frame(frame_num)

    def on_brightness_clicked(self):
        """Slot that opens the brightness adjustment dialog."""
        overlay = self.central_panel.tracking_overlay
        dlg = AdjustmentDialog("brightness", overlay.brightness, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            overlay.set_brightness(dlg.get_value())

    def on_contrast_clicked(self):
        """Slot that opens the contrast adjustment dialog."""
        overlay = self.central_panel.tracking_overlay
        dlg = AdjustmentDialog("contrast", overlay.contrast, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            overlay.set_contrast(dlg.get_value())

    def on_colourise_clicked(self, is_selected):
        """Slot that toggles the 'Colourise' effect."""
        overlay = self.central_panel.tracking_overlay
        
        if is_selected:
            # You could expand this to use your ColouriseDialog
            overlay.toggle_colourise(True)
        else:
            overlay.toggle_colourise(False)
            # Also deselect the icon if the user cancels or turns it off
            for icon in self.left_col.icons:
                if icon.meaning_label.text() == "Colourise":
                    icon.set_selected(False)
                    break

    def on_clear_clicked(self):
        """Slot that clears all tracking data after confirmation."""
        reply = QMessageBox.question(
            self,
            "Clear All Tracking",
            "Are you sure? This will remove tracking points from every frame.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.central_panel.tracking_overlay.reset()
            self.central_panel.tracking_history.clear()
            QMessageBox.information(
                self,
                "Tracking Cleared",
                "All tracking points have been removed."
            )

    def render_video(self):
        if not self.central_panel.cap or not self.central_panel.cap.isOpened():
            QMessageBox.warning(self, "Render Error", "No base video is loaded.")
            return
        if not self.central_panel.tracking_overlay.tracking_history:
            QMessageBox.warning(self, "Render Error", "No tracking data available for rendering.")
            return
        overlay = self.central_panel.tracking_overlay
        if not overlay.has_overlay():
            QMessageBox.warning(self, "Render Error", "No overlay has been inserted to render.")
            return

        # Use dialogs to get render settings... (This part is the same as your old code)
        total_frames = int(self.central_panel.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame, ok1 = QInputDialog.getInt(self, "Render Range", "Enter start frame:", 0, 0, total_frames - 1)
        if not ok1: return
        end_frame, ok2 = QInputDialog.getInt(self, "Render Range", "Enter end frame:", total_frames - 1, start_frame, total_frames - 1)
        if not ok2: return
        scale_factor, ok3 = QInputDialog.getDouble(self, "Render Scale", "Enter scale factor (e.g., 0.5 for half resolution):", 1.0, 0.1, 1.0, 2)
        if not ok3: return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Rendered Video", "", "MP4 Files (*.mp4)")
        if not out_path: return

        settings = self.build_render_settings(start_frame, end_frame,
                                              scale_factor, out_path)

        # --- Threading Setup ---
        self.render_thread = QThread()
        self.render_worker = RenderWorker(settings)
        self.render_worker.moveToThread(self.render_thread)

        # Setup progress dialog
        self.progress_dialog = QProgressDialog("Rendering...", "Cancel", 0, end_frame - start_frame + 1, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()

        # Connect signals and slots
        self.render_thread.started.connect(self.render_worker.run)
        self.render_worker.finished.connect(self.on_render_finished)
        self.render_worker.progress.connect(self.progress_dialog.setValue)
        self.progress_dialog.canceled.connect(self.render_worker.cancel)

        self.render_thread.start()

    def build_render_settings(self, start_frame, end_frame, scale_factor,
                              out_path) -> RenderSettings:
        """Copy the live GUI state into a plain snapshot for the worker.

        Done here, on the GUI thread, so the worker never touches a widget.
        """
        panel = self.central_panel
        overlay = panel.tracking_overlay

        history = {int(k): [(float(x), float(y)) for x, y in v]
                   for k, v in overlay.tracking_history.items()
                   if v is not None and len(v) == 4}

        creative = None
        if not overlay.inserted_overlay_is_video and overlay.overlay_bgra is not None:
            creative = overlay.overlay_bgra.copy()

        return RenderSettings(
            base_video_path=panel.current_video_path,
            out_path=out_path,
            start_frame=start_frame,
            end_frame=end_frame,
            scale_factor=scale_factor,
            fps=panel.fps,
            history=history,
            curvature=overlay.curvature,
            curved=overlay.curved_enabled,
            overlay_bgra=creative,
            overlay_video_path=(overlay.overlay_video_path
                                if overlay.inserted_overlay_is_video else None),
            overlay_start_frame=overlay.inserted_overlay_start_frame,
            brightness=overlay.brightness,
            contrast=overlay.contrast,
            colourise=overlay.colourise_enabled,
        )

    def on_render_finished(self, status):
        """Slot to handle cleanup after rendering finishes, fails, or is canceled."""
        self.progress_dialog.close()
        self.render_thread.quit()
        self.render_thread.wait()

        if status.startswith("Completed"):
            detail = status[len("Completed"):].strip(" ()")
            message = "Video exported successfully."
            if detail:
                message += f"\n\n{detail[0].upper()}{detail[1:]}."
            QMessageBox.information(self, "Render Complete", message)
        elif status == "Canceled":
            QMessageBox.warning(self, "Render Canceled", "The rendering process was canceled.")
        else: # Error
            QMessageBox.critical(self, "Render Failed", f"An error occurred during rendering:\n{status}")

    def save_project(self):
        """Save the whole project: every tracked frame, not just the current shape."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if not path:
            return

        overlay = self.central_panel.tracking_overlay
        data = {
            "version": 2,
            "base_video": getattr(self.central_panel, "current_video_path", ""),
            "tracking_points": [list(p) for p in overlay.points],
            "tracking_history": core.history_to_lists(overlay.tracking_history),
            "curvature": overlay.curvature.tolist(),
            "curved": overlay.curved_enabled,
            "brightness": overlay.brightness,
            "contrast": overlay.contrast,
            "colourise": overlay.colourise_enabled,
            "overlay_path": overlay.overlay_source_path,
            "overlay_is_video": overlay.inserted_overlay_is_video,
            "overlay_start_frame": overlay.inserted_overlay_start_frame,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        QMessageBox.information(self, "Saved", f"Project saved to:\n{path}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "JSON Files (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)

        base_video = data.get("base_video", "")
        if base_video:
            self.central_panel.load_video(base_video)

        overlay = self.central_panel.tracking_overlay
        history = {int(k): v for k, v in data.get("tracking_history", {}).items()}
        overlay.tracking_history = {k: [tuple(p) for p in v] for k, v in history.items()}
        self.central_panel.tracking_history = dict(overlay.tracking_history)

        if data.get("curvature") is not None:
            overlay.curvature = np.array(data["curvature"], np.float32).reshape(4, 2, 2)
        overlay.curved_enabled = bool(data.get("curved", False))
        overlay.brightness = data.get("brightness", 0)
        overlay.contrast = data.get("contrast", 1.0)
        overlay.colourise_enabled = bool(data.get("colourise", False))

        overlay_path = data.get("overlay_path")
        if overlay_path and os.path.exists(overlay_path):
            if data.get("overlay_is_video"):
                overlay.insert_video_overlay(
                    data.get("overlay_start_frame", 0), overlay_path)
            else:
                overlay.insert_image_overlay(
                    overlay_path, data.get("overlay_start_frame", 0))
        elif overlay_path:
            QMessageBox.warning(self, "Missing Creative",
                                f"Could not find the creative at:\n{overlay_path}")

        current = self.central_panel.get_current_frame_index()
        points = overlay.tracking_history.get(current) or data.get("tracking_points", [])
        overlay.points = [tuple(p) for p in points]
        self.central_panel.reset_tracker()
        self.central_panel.refresh_display()
        overlay.update()

    def detect_from_reference(self):
        """Locate the insertion area from an uploaded picture of the target.

        Instead of clicking four corners, the user hands over a photo of the
        billboard, screen or poster and the tool finds it in the footage and
        lands the quad on its actual corners.
        """
        panel = self.central_panel
        if not panel.cap or not panel.cap.isOpened():
            QMessageBox.warning(self, "Detect Error", "Please load a base video first.")
            return

        ref_path, _ = QFileDialog.getOpenFileName(
            self, "Select an image of the target", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not ref_path:
            return

        try:
            matcher = core.ReferenceMatcher(ref_path)
        except (IOError, ValueError) as exc:
            QMessageBox.warning(self, "Detect Error", str(exc))
            return

        progress = QProgressDialog("Searching for the target...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        canceled = {"value": False}

        def report(fraction):
            progress.setValue(int(fraction * 100))
            QApplication.processEvents()
            if progress.wasCanceled():
                canceled["value"] = True

        detection = None
        try:
            # Try the frame on screen first; it is what the user is looking at.
            if panel.prev_frame is not None:
                detection = matcher.locate(panel.prev_frame,
                                           frame_index=panel.get_current_frame_index())
            if detection is None and not canceled["value"]:
                detection = matcher.scan_video(panel.current_video_path,
                                               step=5, progress=report)
        except (IOError, cv2.error) as exc:
            progress.close()
            QMessageBox.warning(self, "Detect Error", str(exc))
            return
        progress.close()

        if canceled["value"]:
            return
        if detection is None:
            QMessageBox.information(
                self, "Not Found",
                "Could not find that target in the video.\n\n"
                "This matches flat, detailed things -- a billboard, a screen, a "
                "poster. Try a sharper, straighter-on picture of it, cropped to "
                "just the target.")
            return

        if detection.frame_index is not None and \
                detection.frame_index != panel.get_current_frame_index():
            panel.jump_to_frame(detection.frame_index)

        overlay = panel.tracking_overlay
        overlay.points = [(float(x), float(y)) for x, y in detection.quad]
        overlay.tracking_history[panel.get_current_frame_index()] = overlay.points[:]
        panel.tracking_history[panel.get_current_frame_index()] = overlay.points[:]
        panel.reset_tracker()
        panel.enable_tracking_mode()
        panel.refresh_display()
        overlay.update()

        QMessageBox.information(
            self, "Target Found",
            f"Found on frame {detection.frame_index} "
            f"({detection.n_inliers} matching features, "
            f"{detection.confidence:.0%} confidence).\n\n"
            "Adjust the corners if you need to, then turn tracking on.")

    def toggle_curved_edges(self, enabled: bool):
        """Switch curved edges on or off for the insertion area."""
        overlay = self.central_panel.tracking_overlay
        overlay.set_curved_enabled(enabled)
        self.central_panel.refresh_display()
        if enabled:
            QMessageBox.information(
                self, "Curved Edges On",
                "Each edge now has two square handles. Drag them to bend that "
                "edge around a curved screen or pillar.\n\n"
                "Leaving them where they are keeps the area perfectly flat.")

    def save_tracking_points(self):
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Save Tracking", "", "JSON Files (*.json)")
        if not path:
            return
        # Convert keys to strings so JSON doesn't break
        data = {str(k): v for k, v in self.central_panel.tracking_overlay.tracking_history.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        QMessageBox.information(self, "Saved", f"Tracking saved to:\n{path}")

    def load_tracking_points(self):
        import json
        path, _ = QFileDialog.getOpenFileName(self, "Load Tracking", "", "JSON Files (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)
        # Convert keys back to int
        loaded = {int(k): v for k, v in data.items()}

        # Update both overlay and central panel
        self.central_panel.tracking_overlay.tracking_history = loaded
        self.central_panel.tracking_history = loaded.copy()

        current_frame_index = self.central_panel.get_current_frame_index()
        # If the loaded file has data for the current frame, show it
        if current_frame_index in loaded:
            self.central_panel.tracking_overlay.points = loaded[current_frame_index][:]
        else:
            self.central_panel.tracking_overlay.points.clear()
        # Force an update so that the points are drawn
        self.central_panel.tracking_overlay.update()

        # Automatically enable tracking mode so that the points appear
        self.central_panel.enable_tracking_mode()

        QMessageBox.information(self, "Loaded", f"Tracking loaded from:\n{path}")

    def save_aoi_geometry(self):
        import csv
        if not self.central_panel.cap or not self.central_panel.cap.isOpened():
            QMessageBox.warning(self, "Export Error", "No base video is loaded.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save AOI Geometry", "", "CSV Files (*.csv)"
        )
        if not save_path:
            return

        # base video dims
        video_w = int(self.central_panel.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(self.central_panel.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.central_panel.fps or 30

        # Row 1 (optional)
        row1 = [
            f"name:aoi1",
            "csv_type_3",
            "width:None",
            "height:None",
            f"stim_width:{video_w}",
            f"stim_height:{video_h}",
        ]

        # Row 2 (the requested header)
        row2 = [
            "START",
            "END",
            "topleft_coords_XYZ_m",
            "topright_coords_XYZ_m",
            "bottomleft_coords_XYZ_m",
            "bottomright_coords_XYZ_m",
            "top_angle_deg",
            "left_angle_deg",
            "bottom_angle_deg",
            "right_angle_deg",
            "average_dist_from_corners",
            "disc_viewable_anagle",
            "disc_deflection_angle",
            "disc_diam",
            "points"
        ]

        geometry_data = []

        # Interpolate exactly as the renderer does. Exporting only the frames
        # that happen to be keyed would leave holes in the AOI wherever the
        # user scrubbed or re-adjusted, so a fixation landing on the ad during
        # a gap would not be attributed to it -- quietly under-counting dwell
        # time and fixation counts. It would also disagree with the rendered
        # video, which draws the ad on every frame in the range.
        frame_dict = core.interpolate_tracking(
            self.central_panel.tracking_overlay.tracking_history)

        # Sort the frame numbers so CSV rows are in ascending order
        for frame_idx in sorted(frame_dict.keys()):
            corners = frame_dict[frame_idx]

            # For the START/END times:
            start_time = frame_idx / fps
            end_time   = (frame_idx + 1) / fps

            # Build the “points” string if we have 4 corners
            if len(corners) == 4:
                x0, y0 = corners[0]
                x1, y1 = corners[1]
                x2, y2 = corners[2]
                x3, y3 = corners[3]
                points_str = f"{x0:.4f};{y0:.4f};{x1:.4f};{y1:.4f};{x2:.4f};{y2:.4f};{x3:.4f};{y3:.4f}"
            else:
                points_str = ""

            row_out = [
                f"{start_time:.3f}",
                f"{end_time:.3f}",
                "0; 0; 0",
                "0; 0; 0",
                "0; 0; 0",
                "0; 0; 0",
                "None",
                "None",
                "None",
                "None",
                "0",
                "None",
                "None",
                "None",
                points_str
            ]
            geometry_data.append(row_out)

        with open(save_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')  # or delimiter=';'
            writer.writerow(row1)
            writer.writerow(row2)
            writer.writerows(geometry_data)

        QMessageBox.information(self, "Export Complete", f"AOI Geometry saved to {save_path}.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.isMaximized():
            width = self.width()
            height = int(width / self.aspect_ratio)
            screen_geometry = QApplication.primaryScreen().availableGeometry()
            if height > screen_geometry.height():
                height = screen_geometry.height()
                width = int(height * self.aspect_ratio)
            self.resize(width, height)

    def upload_base_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Base Video", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            print(file_path)
            logging.debug(f"Base video uploaded: {file_path}")
            self.central_panel.load_video(file_path)

    def upload_overlay(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Overlay Files", "", "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mkv)"
        )
        if file_paths:
            logging.debug(f"Overlay files uploaded: {file_paths}")
            for file_path in file_paths:
                self.add_overlay(file_path)

    def add_overlay(self, file_path):
        # Create a new frame for the overlay widget.
        overlay_widget = QFrame()
        overlay_widget.setStyleSheet("background-color: #2A2A2A; border-radius: 10px; padding: 5px;")
        # Custom attributes to manage state and resources
        overlay_widget.inserted = False
        overlay_widget.inserted_is_video = False
        overlay_widget.video_path = None
        overlay_widget.image_path = None
        overlay_widget.pixmap = None
        overlay_widget.preview_timer = None
        overlay_widget.preview_cap = None

        # This is the main vertical layout for the entire widget card.
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setContentsMargins(5, 5, 5, 5)
        overlay_layout.setSpacing(10)

        # --- THIS SECTION RESTORES THE CORRECT BUTTON LAYOUT ---
        # Create a dedicated horizontal layout just for the top row of buttons.
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)

        # Create the remove button.
        remove_btn = QPushButton("✖")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: red; color: white; font-size: 16px;
                border: none; border-radius: 5px; padding: 5px;
            }
            QPushButton:hover { background-color: darkred; }
        """)
        remove_btn.setFixedSize(40, 25)
        remove_btn.setCursor(QCursor(Qt.PointingHandCursor))

        # Create the insert/select button.
        select_btn = QPushButton("Insert")
        select_btn.setObjectName("selectBtn")
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: green; color: white; font-size: 14px;
                border: none; border-radius: 5px; padding: 5px;
            }
            QPushButton:hover { background-color: darkgreen; }
        """)
        select_btn.setFixedSize(60, 25)
        select_btn.setCursor(QCursor(Qt.PointingHandCursor))

        # Create an extension label for display.
        ext_label = QLabel(file_path.split('.')[-1].upper())
        ext_label.setStyleSheet("color: white; font-size: 12px;")
        ext_label.setFixedHeight(25)
        ext_label.setAlignment(Qt.AlignCenter)

        # Add the buttons and label to the horizontal layout.
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(select_btn)
        button_layout.addWidget(ext_label)
        button_layout.addStretch()

        # Add the entire horizontal button layout to the main vertical layout.
        overlay_layout.addLayout(button_layout)
        # --- END OF RESTORED LAYOUT SECTION ---

        # Depending on file extension, load as image or video.
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            overlay_widget.inserted_is_video = False
            overlay_widget.image_path = file_path
            overlay_widget.pixmap = QPixmap(file_path)
            preview_label = QLabel()
            if not overlay_widget.pixmap.isNull():
                preview_label.setPixmap(overlay_widget.pixmap.scaledToWidth(200, Qt.SmoothTransformation))
            preview_label.setMinimumHeight(100)
            preview_label.setAlignment(Qt.AlignCenter)
            overlay_layout.addWidget(preview_label)

        elif file_path.lower().endswith(('.mp4', '.avi', '.mkv')):
            overlay_widget.inserted_is_video = True
            overlay_widget.video_path = file_path

            # Use a QLabel to display frames from OpenCV
            preview_label = QLabel("Loading preview...")
            preview_label.setStyleSheet("color: white;")
            preview_label.setMinimumSize(200, 112)
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setCursor(QCursor(Qt.PointingHandCursor))
            overlay_layout.addWidget(preview_label)

            # Create a VideoCapture and QTimer specific to this widget
            overlay_widget.preview_cap = cv2.VideoCapture(file_path)
            overlay_widget.preview_timer = QTimer(overlay_widget) # Parent to the widget

            def update_preview_frame():
                if not overlay_widget.preview_cap or not overlay_widget.preview_cap.isOpened():
                    return
                
                ret, frame = overlay_widget.preview_cap.read()
                if not ret: # If end of video, loop back to the start
                    overlay_widget.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = overlay_widget.preview_cap.read()
                    if not ret: return

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                preview_label.setPixmap(pixmap.scaledToWidth(200, Qt.SmoothTransformation))

            def toggle_preview_playback(event):
                if overlay_widget.preview_timer.isActive():
                    overlay_widget.preview_timer.stop()
                else:
                    overlay_widget.preview_timer.start()
                event.accept()

            preview_label.mousePressEvent = toggle_preview_playback

            if overlay_widget.preview_cap.isOpened():
                fps = overlay_widget.preview_cap.get(cv2.CAP_PROP_FPS)
                interval = int(1000 / fps) if fps > 0 else 40 # Default to 25fps
                overlay_widget.preview_timer.setInterval(interval)
                overlay_widget.preview_timer.timeout.connect(update_preview_frame)
                overlay_widget.preview_timer.start()
            else:
                preview_label.setText("Preview failed")

        # The rest of the function remains the same.
        self.overlay_layout.insertWidget(self.overlay_layout.count() - 1, overlay_widget)

        def remove_overlay():
            confirm = QMessageBox.question(self, "Remove Overlay", "Are you sure?", QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                if overlay_widget.inserted:
                    uninsert_overlay()
                if overlay_widget.preview_timer:
                    overlay_widget.preview_timer.stop()
                if overlay_widget.preview_cap:
                    overlay_widget.preview_cap.release()
                overlay_widget.deleteLater()

        remove_btn.clicked.connect(remove_overlay)

        def uninsert_overlay():
            overlay_widget.inserted = False
            select_btn.setText("Insert")
            select_btn.setStyleSheet("QPushButton { background-color: green; color: white; font-size: 14px; border: none; border-radius: 5px; padding: 5px; } QPushButton:hover { background-color: darkgreen; }")
            overlay_widget.setStyleSheet("background-color: #2A2A2A; border-radius: 10px; padding: 5px;")
            self.central_panel.tracking_overlay.remove_inserted_overlay()
            if self.inserted_overlay_widget == overlay_widget:
                self.inserted_overlay_widget = None
            if self.central_panel.tracking_mode:
                self.central_panel.tracking_overlay.setFocus()

        def toggle_insert():
            if self.inserted_overlay_widget and self.inserted_overlay_widget is not overlay_widget:
                old_widget = self.inserted_overlay_widget
                old_widget.inserted = False
                old_widget.setStyleSheet("background-color: #2A2A2A; border-radius: 10px; padding: 5px;")
                old_select_btn = old_widget.findChild(QPushButton, "selectBtn")
                if old_select_btn:
                    old_select_btn.setText("Insert")
                    old_select_btn.setStyleSheet("QPushButton { background-color: green; color: white; font-size: 14px; border: none; border-radius: 5px; padding: 5px; } QPushButton:hover { background-color: darkgreen; }")
                self.central_panel.tracking_overlay.remove_inserted_overlay()
                self.inserted_overlay_widget = None
            
            if not overlay_widget.inserted:
                overlay_widget.inserted = True
                select_btn.setText("Inserted")
                select_btn.setStyleSheet("QPushButton { background-color: limegreen; color: white; font-size: 14px; border: none; border-radius: 5px; padding: 5px; } QPushButton:hover { background-color: green; }")
                overlay_widget.setStyleSheet("background-color: #00A000; border-radius: 10px; padding: 5px;")
                self.inserted_overlay_widget = overlay_widget
                current_frame_index = self.central_panel.get_current_frame_index()
                
                if overlay_widget.inserted_is_video and overlay_widget.video_path:
                    self.central_panel.tracking_overlay.insert_video_overlay(start_frame_index=current_frame_index, video_path=overlay_widget.video_path)
                    self.central_panel.tracking_overlay.update_overlay_video_frame_by_index(0)
                else:
                    # Prefer the file path: reading it directly keeps the
                    # creative's alpha channel intact.
                    source = overlay_widget.image_path or overlay_widget.pixmap
                    self.central_panel.tracking_overlay.insert_image_overlay(
                        source, current_frame_index)
                
                self.central_panel.tracking_overlay.show()
                self.central_panel.tracking_overlay.update()
            else:
                uninsert_overlay()

        select_btn.clicked.connect(toggle_insert)

    def make_video_click_handler(self, player, original_event):
        def handler(event):
            if player.state() == QMediaPlayer.PlayingState:
                player.pause()
            else:
                player.play()
            original_event(event)
        return handler

if __name__ == '__main__':
    logging.debug("Application about to start QApplication")
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    * {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    """)
    window = MainWindow()
    window.show()
    logging.debug("Entering app event loop")
    sys.exit(app.exec_())

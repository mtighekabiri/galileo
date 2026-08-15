import os
import sys
import logging
import cv2
import csv
import numpy as np
import json

import galileo_core as core
import galileo_blend as blend
import galileo_morph as morphlib
import galileo_deflicker as deflicker

from PyQt5.QtCore import (
    QObject, QThread, pyqtSignal, Qt, QSize, QUrl, QEvent, QTimer, QPoint, QRect,
    QLineF, QPointF, QRectF, QPropertyAnimation, pyqtProperty, pyqtSignal)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QInputDialog,
    QGraphicsDropShadowEffect, QPushButton, QLabel, QMainWindow,
    QSpacerItem, QSizePolicy, QSlider, QFileDialog, QMenu, QAction,
    QStyle, QMessageBox, QGridLayout, QCheckBox, QDialog, QDialogButtonBox,
    QProgressDialog, QScrollArea, QSpinBox, QComboBox, QLineEdit,
    QListWidget, QListWidgetItem, QWIDGETSIZE_MAX)
from PyQt5.QtGui import (
    QCursor, QPixmap, QColor, QPainter, QBrush, QPen, QImage, QPolygonF)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

def app_directory() -> str:
    """Where the application lives — the bundle folder when packaged."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_directory() -> str:
    """Where files bundled *into* a packaged build live.

    PyInstaller unpacks bundled data to its own folder (``_internal`` for a
    one-folder build), which is not the folder holding the executable, so both
    have to be searched: one for what was shipped, one for what a user dropped
    in later.
    """
    return getattr(sys, "_MEIPASS", app_directory())


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

    directory = os.path.join(root, "Galileo")
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

# Let a bundled ffmpeg and the segmentation model sitting beside the
# application be found, so audio and occlusion work without anything installed.
for _directory in (app_directory(), resource_directory()):
    core.register_binary_dir(_directory)
    core.register_model_dir(_directory)
    core.register_model_dir(os.path.join(_directory, core.MODEL_DIR_NAME))

# Record what optional pieces were found. This is the first thing to look at
# when someone reports that occlusion is greyed out or a render came out silent.
logging.debug("Frozen: %s | app dir: %s | resources: %s",
              bool(getattr(sys, "frozen", False)), app_directory(),
              resource_directory())
logging.debug("Occlusion model: %s",
              core.find_model(core.PersonSegmenter.MODEL_FILENAME) or "NOT FOUND")
logging.debug("ffmpeg: %s", core.find_binary("ffmpeg") or "NOT FOUND")

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
# in galileo_core so that the preview and the render share one implementation and
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

class BlendDialog(QDialog):
    """Live controls for how hard the creative is matched to the footage.

    Non-modal in effect: every slider updates the preview as it moves, because
    judging a blend from a number is impossible and the whole point is how it
    looks against the shot.
    """

    EFFECTS = [
        ("lighting", "Lighting",
         "Carry the surface's uneven light and shadow onto the creative."),
        ("colour", "Colour cast",
         "Tint towards the ambient light. Kept low by default: this is the one\n"
         "control that alters the creative's own colours, which an ad test is\n"
         "often measuring."),
        ("sharpness", "Softness",
         "Soften a too-crisp creative to match the focus of the surface."),
        ("grain", "Grain",
         "Add sensor noise matching the footage, so the insert is not\n"
         "conspicuously clean."),
        ("motion", "Motion blur",
         "Smear the creative when the camera pans, as the rest of the shot is."),
        ("highlights", "Keep highlights",
         "Let reflections on glass show through. Off for matte billboards."),
    ]

    def __init__(self, settings, on_change, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.on_change = on_change
        self.original = settings.to_dict()

        self.setWindowTitle("Blend into Footage")
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QDialog { background-color: #2A2A2A; }
            QLabel { color: #E8E8E8; font-size: 13px; }
            QCheckBox { color: #E8E8E8; font-size: 13px; }
            QSlider::groove:horizontal { background: #444; height: 5px;
                                         border-radius: 2px; }
            QSlider::handle:horizontal { background: #DDD; width: 14px;
                                         margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #00A000;
                                           border-radius: 2px; }
            QPushButton { background-color: #3C3C3C; color: white; border: none;
                          border-radius: 6px; padding: 7px 16px; }
            QPushButton:hover { background-color: #505050; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.enabled_box = QCheckBox("Match the creative to the footage")
        self.enabled_box.setChecked(settings.enabled)
        self.enabled_box.toggled.connect(self._on_enabled)
        layout.addWidget(self.enabled_box)

        note = QLabel("Measured from the pixels the creative covers.")
        note.setStyleSheet("color: #9A9A9A; font-size: 11px;")
        layout.addWidget(note)

        self.sliders = {}
        self.value_labels = {}
        for name, title, hint in self.EFFECTS:
            row = QHBoxLayout()
            label = QLabel(title)
            label.setFixedWidth(110)
            label.setToolTip(hint)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(getattr(settings, name) * 100))
            slider.setToolTip(hint)
            slider.valueChanged.connect(
                lambda value, key=name: self._on_slider(key, value))
            value_label = QLabel(f"{getattr(settings, name):.2f}")
            value_label.setFixedWidth(38)
            value_label.setStyleSheet("color: #9A9A9A; font-size: 12px;")

            row.addWidget(label)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            layout.addLayout(row)
            self.sliders[name] = slider
            self.value_labels[name] = value_label

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel |
                                   QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self._revert_and_reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._defaults)
        layout.addWidget(buttons, alignment=Qt.AlignRight)
        self._sync_enabled()

    def _on_enabled(self, checked):
        self.settings.enabled = checked
        self._sync_enabled()
        self.on_change()

    def _sync_enabled(self):
        for slider in self.sliders.values():
            slider.setEnabled(self.settings.enabled)

    def _on_slider(self, key, value):
        setattr(self.settings, key, value / 100.0)
        self.value_labels[key].setText(f"{value / 100.0:.2f}")
        self.on_change()

    def _defaults(self):
        fresh = blend.BlendSettings()
        for name, _, _ in self.EFFECTS:
            self.sliders[name].setValue(int(getattr(fresh, name) * 100))
        self.enabled_box.setChecked(True)
        self.fill_box.setChecked(morphlib.Morph().fill)

    def _revert_and_reject(self):
        for key, value in self.original.items():
            setattr(self.settings, key, value)
        self.on_change()
        self.reject()


class MorphDialog(QDialog):
    """Bend and tilt the creative, with a thumbnail that follows the sliders.

    Every control updates both the thumbnail here and the video preview behind,
    because the only way to judge whether a bend looks like the surface it is
    meant to sit on is to see it against that surface.
    """

    CONTROLS = [
        ("yaw", "Turn", -45, 45, "Turn the right edge away from the viewer."),
        ("pitch", "Tip", -45, 45, "Tip the top edge away from the viewer."),
        ("roll", "Rotate", -30, 30, "Rotate within the surface."),
        ("bow_h", "Bow across", -100, 100,
         "Curve about the vertical axis, as on a convex panel.\n"
         "Positive bulges the middle towards the viewer."),
        ("bow_v", "Bow down", -100, 100, "Curve about the horizontal axis."),
        ("shading", "Curve shading", 0, 100,
         "How much a bowed surface darkens as it turns away.\n"
         "This is what makes a bow read as a curve rather than as a\n"
         "squashed flat sheet. Drop it to 0 to leave the creative's\n"
         "own colours untouched."),
        ("perspective", "Perspective", 0, 100,
         "How strongly a turn foreshortens.\nLow is nearly a flat squash; "
         "high exaggerates depth."),
    ]
    # Stored 0..1, shown 0..100.
    SCALED = {"bow_h", "bow_v", "perspective", "shading"}

    def __init__(self, morph, creative, on_change, parent=None, preview=None):
        """
        Args:
            preview: called for a BGR picture of the creative as it currently
                sits on the footage, cropped to the area. Without one the
                artwork is shown on its own against a checkerboard, which says
                what the shape is but not whether it suits the surface -- and
                the surface is the only thing that can answer that.
        """
        super().__init__(parent)
        self.morph = morph
        self.creative = creative
        self.on_change = on_change
        self.preview = preview
        self.original = morph.to_dict()

        self.setWindowTitle("Shape the Creative")
        self.setMinimumWidth(520)
        self.setStyleSheet("""
            QDialog { background-color: #2A2A2A; }
            QLabel { color: #E8E8E8; font-size: 13px; }
            QCheckBox { color: #E8E8E8; font-size: 13px; }
            QSlider::groove:horizontal { background: #444; height: 5px;
                                         border-radius: 2px; }
            QSlider::handle:horizontal { background: #DDD; width: 14px;
                                         margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #6A8FD8;
                                           border-radius: 2px; }
            QPushButton { background-color: #3C3C3C; color: white; border: none;
                          border-radius: 6px; padding: 7px 16px; }
            QPushButton:hover { background-color: #505050; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(9)

        switches = QHBoxLayout()
        self.enabled_box = QCheckBox("Shape this creative")
        self.enabled_box.setChecked(morph.enabled)
        self.enabled_box.toggled.connect(self._on_enabled)
        switches.addWidget(self.enabled_box)

        self.fill_box = QCheckBox("Fill the area")
        self.fill_box.setChecked(morph.fill)
        self.fill_box.setToolTip(
            "Keep the creative covering every pixel of the tracked area,\n"
            "cropping whatever a turn pushes outside it.\n\n"
            "Turned off, a turn leaves transparent wedges at the corners --\n"
            "and those wedges are the tracked surface showing through.")
        self.fill_box.toggled.connect(self._on_fill)
        switches.addWidget(self.fill_box)
        switches.addStretch()
        layout.addLayout(switches)

        self.thumbnail = QLabel()
        self.thumbnail.setFixedHeight(210 if preview is not None else 150)
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setStyleSheet("background-color: #141414; border-radius: 6px;")
        layout.addWidget(self.thumbnail)

        self.sliders = {}
        self.value_labels = {}
        for name, title, low, high, hint in self.CONTROLS:
            row = QHBoxLayout()
            label = QLabel(title)
            label.setFixedWidth(96)
            label.setToolTip(hint)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(low, high)
            slider.setValue(self._to_slider(name))
            slider.setToolTip(hint)
            slider.valueChanged.connect(
                lambda value, key=name: self._on_slider(key, value))
            value_label = QLabel("")
            value_label.setFixedWidth(44)
            value_label.setStyleSheet("color: #9A9A9A; font-size: 12px;")

            row.addWidget(label)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            layout.addLayout(row)
            self.sliders[name] = slider
            self.value_labels[name] = value_label

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel |
                                   QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self._revert_and_reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).setText("Flat")
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._flatten)
        layout.addWidget(buttons, alignment=Qt.AlignRight)

        self._sync_enabled()
        self._refresh()

    def _to_slider(self, name):
        value = getattr(self.morph, name)
        return int(round(value * 100)) if name in self.SCALED else int(round(value))

    def _on_enabled(self, checked):
        self.morph.enabled = checked
        self._sync_enabled()
        self._refresh()

    def _on_fill(self, checked):
        self.morph.fill = checked
        self._refresh()

    def _sync_enabled(self):
        for slider in self.sliders.values():
            slider.setEnabled(self.morph.enabled)

    def _on_slider(self, key, value):
        setattr(self.morph, key, value / 100.0 if key in self.SCALED else float(value))
        self._refresh()

    def _flatten(self):
        fresh = morphlib.Morph()
        for name, _, _, _, _ in self.CONTROLS:
            self.sliders[name].setValue(
                int(round(getattr(fresh, name) * 100)) if name in self.SCALED
                else int(round(getattr(fresh, name))))
        self.enabled_box.setChecked(True)
        self.fill_box.setChecked(morphlib.Morph().fill)

    def _revert_and_reject(self):
        for key, value in self.original.items():
            setattr(self.morph, key, value)
        self.on_change()
        self.reject()

    def _refresh(self):
        for name, _, _, _, _ in self.CONTROLS:
            value = getattr(self.morph, name)
            self.value_labels[name].setText(
                f"{value:.2f}" if name in self.SCALED else f"{value:.0f}°")
        self._draw_thumbnail()
        self.on_change()

    def _draw_on_the_footage(self) -> bool:
        """Show the creative where it will actually be, on the shot itself."""
        try:
            picture = self.preview()
        except Exception:                      # never let a preview stop an edit
            logging.debug("shape preview failed", exc_info=True)
            return False
        if picture is None or picture.size == 0:
            return False

        height = self.thumbnail.height()
        width = max(1, int(picture.shape[1] * height / picture.shape[0]))
        limit = max(1, self.thumbnail.width() or 480)
        if width > limit:
            width, height = limit, max(1, int(picture.shape[0] * limit / picture.shape[1]))
        small = cv2.resize(picture, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
        self.thumbnail.setPixmap(QPixmap.fromImage(image.copy()))
        return True

    def _draw_thumbnail(self):
        if self.preview is not None and self._draw_on_the_footage():
            return
        if self.creative is None:
            return
        shaped = morphlib.apply_morph(self.creative, self.morph)
        shaped = core.to_bgra(shaped)

        height = 150
        scale = height / shaped.shape[0]
        width = max(1, int(shaped.shape[1] * scale))
        small = cv2.resize(shaped, (width, height), interpolation=cv2.INTER_AREA)

        # Checkerboard behind, so the transparent surround a tilt creates is
        # visible as transparency rather than reading as black artwork.
        board = np.zeros((height, width, 3), np.uint8)
        board[:] = 40
        step = 12
        for y in range(0, height, step):
            for x in range(0, width, step):
                if ((x // step) + (y // step)) % 2 == 0:
                    board[y:y + step, x:x + step] = 60

        alpha = small[:, :, 3:4].astype(np.float32) / 255.0
        blended = (board * (1 - alpha) + small[:, :, :3] * alpha).astype(np.uint8)
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
        self.thumbnail.setPixmap(QPixmap.fromImage(image.copy()))


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
        background_color = QColor("#00A000") if self._checked else QColor("#5A5A5A")
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

        # Latching modes get the bright fill so it is obvious which of them is
        # currently changing what a click does. This keyed off "Overlay", which
        # matches no icon, so Colourise and Curve were indicated at 2.3:1
        # contrast against the panel.
        if label_text in LeftColumn.STATEFUL_ICONS:
            self.selected_color = "#00A000"
        else:
            self.selected_color = self.default_selected_color

        self.base_color = self.default_base_color
        self.hover_color = self.default_hover_color

        # Nine tools have to fit the shortest window the app allows. At 62
        # high they needed 558px of column against the 480 a 1024x640 window
        # leaves, so Curve and Clear sat below the fold behind a scrollbar
        # barely a pixel wide -- present, but not findable.
        self.setFixedSize(78, 52)
        self.set_normal_style()

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(2, 4, 2, 4)
        self.layout.setSpacing(1)

        self.icon_label = QLabel(icon_char)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 20px; color: white;")

        self.meaning_label = QLabel(label_text)
        self.meaning_label.setAlignment(Qt.AlignCenter)
        self.meaning_label.setStyleSheet("font-size: 10px; color: #E0E0E0;")

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

class MagnifierPoint:
    """One handle worth magnifying, and what to call it."""

    CORNER = "corner"
    HANDLE = "handle"

    __slots__ = ("x", "y", "label", "kind", "title")

    def __init__(self, x, y, label="", kind=CORNER, title=None):
        """
        Args:
            label: a couple of characters, for a tile in a crowded grid.
            title: the full name, shown when this handle has the widget to
                itself and there is room to spell it out.
        """
        self.x = float(x)
        self.y = float(y)
        self.label = label
        self.kind = kind
        self.title = title or label

    @classmethod
    def coerce(cls, item, index):
        """Accept a bare ``(x, y)`` as well as a full point."""
        if isinstance(item, cls):
            return item
        x, y = item
        return cls(x, y, str(index + 1), cls.CORNER, f"Corner {index + 1}")


class MagnifierWidget(QWidget):
    """A zoomed view of the handles, for placing them exactly.

    Marking the area is the one part of the job where single pixels decide the
    outcome. The tracker seeds its features from whatever is enclosed, so a
    corner left a little inside the screen throws away the bezel detail that
    tracks best, and one left a little outside picks up the wall behind and
    drags the insert off during a pan. Neither is visible at video scale.

    So this shows *true pixels*, not a smoothed enlargement: past 6x it stops
    interpolating and draws the source grid, because the question being asked
    is "which pixel is the edge on", and a blurred answer is no answer. It also
    draws the region's own outline through each view, which is what makes the
    curve handles judgeable -- a Bezier handle is placed correctly when the
    curve it produces sits on the screen's edge, and that cannot be told from
    the handle's own position.
    """

    #: Above this magnification, interpolating between source pixels hides the
    #: very boundaries the user is trying to place a corner on. Four is where a
    #: source pixel is first big enough to be told apart from its neighbour;
    #: below it smoothing is the better picture, since nobody is counting
    #: pixels while looking for a handle.
    CRISP_ABOVE = 4.0
    #: Above this, source pixels are big enough on screen to rule off.
    GRID_ABOVE = 12.0

    MIN_ZOOM, MAX_ZOOM = 2.0, 32.0

    #: How many pixels of footage to show across the shorter side of a tile
    #: when choosing the magnification for itself.
    #:
    #: Magnification cannot sensibly be a fixed number, because a tile is
    #: anywhere from 50 to 500 pixels across depending on the size of the
    #: widget and how many handles are in it. A fixed 8x put nine pixels of
    #: footage in a default-sized tile: no context to align against, and
    #: blocks so large the picture reads as mush. Holding the amount of
    #: *footage* steady instead means a small tile stays legible and a large
    #: one earns real magnification.
    DEFAULT_SPAN = 28.0

    #: Asks the panel to fill the video stage with this widget, or to put it
    #: back. Twelve handles in a floating box a couple of hundred pixels wide
    #: leaves each one smaller than the thumbnail it replaced, which is worse
    #: than the four quadrants ever were.
    expandToggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(MagnifierWidget, self).__init__(parent)
        self.setStyleSheet("background-color: #121212; border: 2px solid #121212;")
        self.setMinimumSize(150, 150)

        # The base frame (np.ndarray, BGR format)
        self.base_frame = None

        #: :class:`MagnifierPoint` list, in the base frame's coordinate space.
        self.points = []
        #: The region, so its outline can be drawn through each view.
        self.region = None
        #: Which bend handle the keys are pointing at, as (edge, slot).
        self.selected_control = None

        self.zoom_factor = 8.0
        #: Let each tile pick its own magnification from its size. Scrolling
        #: sets a fixed one instead.
        self.auto_zoom = True
        # Kept for callers that still pass it; the view is centred exactly on
        # the point now, so there is no ROI to pad.
        self.buffer = 10

        #: Which point is selected, or -1. Selected points get a green mark.
        self.selected_index = -1
        #: Which point to show alone, filling the widget, or None for all of
        #: them. Set while a handle is being dragged.
        self.focus_index = None

        # For manual resizing
        self.resizing = False
        self.last_mouse_pos = None
        self.resize_handle_size = 20  # size of the "corner handle"
        self._grip_lit = False        # brighten it while the mouse is over it
        #: Filling the video stage rather than floating over a corner of it.
        self.expanded = False
        self._expand_lit = False
        #: One magnified view of the whole area, with every handle on it,
        #: instead of a tile each. A grid shows each handle closely but never
        #: shows the shape they make; this shows the shape, which is what
        #: tells you whether the outline is following the screen.
        self.whole_area = False
        #: The frame with the creatives composited into it. The whole-area view
        #: shows this one: it is a view of the insert as a whole, so what
        #: matters there is whether the advert sits on the screen properly. A
        #: tile is the opposite -- it exists to line one handle up against the
        #: real edge underneath, which the creative would cover -- so tiles
        #: keep showing base_frame.
        self.composited = None
        self._single_lit = False
        # Set here as well as on the first manual resize: update_magnifier_dimensions
        # reads it as soon as a second corner exists, which is long before the
        # user has had any reason to drag the magnifier.
        self.user_resized = False

        self.setMouseTracking(True)

    # -- data --------------------------------------------------------------

    @property
    def overlay_points(self):
        """The plain coordinates, for callers written against the old shape."""
        return [(p.x, p.y) for p in self.points]

    def sizeHint(self):
        # By default, just return current size
        return QSize(self.width(), self.height())

    def setData(
        self,
        base_frame: np.ndarray,
        overlay_points: list,
        zoom_factor: float = None,
        buffer: int = None,
        selected_index: int = None,
        region=None,
        focus_index=-1,
        selected_control=-1,
        composited=None,
    ):
        """
        Update the magnifier data and refresh.

        :param base_frame:     np.ndarray BGR image
        :param overlay_points: :class:`MagnifierPoint` list, or bare ``(x, y)``
        :param zoom_factor:    optional new zoom factor
        :param buffer:         accepted and ignored; kept for older callers
        :param selected_index: which handle is selected, or -1 for none
        :param region:         a ``core.Region`` whose outline to draw
        :param focus_index:    show this handle alone; None for all of them.
                               Left at -1 to mean "unchanged".
        :param selected_control: which bend handle is picked, as (edge, slot).
        :param composited:     the same frame with the creatives drawn in, used
                               by the whole-area view. None falls back to
                               ``base_frame``.
        """
        self.base_frame = base_frame
        self.composited = composited
        self.points = [MagnifierPoint.coerce(p, i)
                       for i, p in enumerate(overlay_points or [])]
        self.region = region

        if zoom_factor is not None:
            self.zoom_factor = float(np.clip(zoom_factor, self.MIN_ZOOM,
                                             self.MAX_ZOOM))
            self.auto_zoom = False       # an explicit number means fix it
        if buffer is not None:
            self.buffer = buffer
        if selected_index is not None:
            self.selected_index = selected_index
        if focus_index != -1:
            self.focus_index = focus_index
        if selected_control != -1:
            self.selected_control = selected_control

        self.update()

    # -- interaction -------------------------------------------------------

    def wheelEvent(self, event):
        """Zoom in and out. A fixed magnification cannot suit both finding a
        handle and placing it."""
        steps = event.angleDelta().y() / 120.0
        if steps:
            # Start from whatever is on screen, so the first notch nudges the
            # current view rather than jumping to some remembered number.
            tiles = self.tiles()
            current = self.zoom_for(tiles[0][0]) if tiles else self.zoom_factor
            self.auto_zoom = False
            self.zoom_factor = float(np.clip(current * (1.25 ** steps),
                                             self.MIN_ZOOM, self.MAX_ZOOM))
            self.update()
            event.accept()
            return
        super(MagnifierWidget, self).wheelEvent(event)

    def zoom_badge_rect(self) -> QRect:
        """Where the magnification is shown. Clicking it hands the choice back
        to the tiles, which is the only way out of a zoom scrolled somewhere
        unhelpful without guessing at the number it started from."""
        return QRect(6, self.height() - 20, 46, 15)

    def expand_button_rect(self) -> QRect:
        """Where the fill-the-stage button sits."""
        return QRect(self.width() - 27, 5, 22, 22)

    def single_button_rect(self) -> QRect:
        """Where the one-view-or-all button sits, beside it."""
        return QRect(self.width() - 52, 5, 22, 22)

    def set_whole_area(self, whole: bool):
        if bool(whole) == self.whole_area:
            return
        self.whole_area = bool(whole)
        self.update()

    def set_expanded(self, expanded: bool):
        if expanded == self.expanded:
            return
        self.expanded = bool(expanded)
        self.expandToggled.emit(self.expanded)
        self.update()

    def mouseDoubleClickEvent(self, event):
        """Double-click anywhere as well, since the button is small."""
        on_a_button = (self.expand_button_rect().contains(event.pos())
                       or self.single_button_rect().contains(event.pos()))
        if (event.button() == Qt.LeftButton and not on_a_button
                and not self.over_grip(event.x(), event.y())):
            self.set_expanded(not self.expanded)
            event.accept()
            return
        super(MagnifierWidget, self).mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.expand_button_rect().contains(event.pos()):
                self.set_expanded(not self.expanded)
                event.accept()
                return
            if self.single_button_rect().contains(event.pos()):
                self.set_whole_area(not self.whole_area)
                event.accept()
                return
            if not self.auto_zoom and self.zoom_badge_rect().contains(event.pos()):
                self.auto_zoom = True
                self.update()
                event.accept()
                return
            if self.over_grip(event.x(), event.y()):
                self.resizing = True
                self.last_mouse_pos = event.globalPos()
                self.user_resized = True   # This flag disables auto-resizing
                # Whatever pinned the size, a drag on the grip has to be able
                # to change it. Anything that fixes the size leaves the maximum
                # equal to the minimum, and then resize() below does nothing at
                # all -- no error, no movement, just a handle that seems dead.
                self.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
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
            over_grip = self.over_grip(event.x(), event.y())
            over_expand = self.expand_button_rect().contains(event.pos())
            over_single = self.single_button_rect().contains(event.pos())
            if over_expand or over_single:
                self.setCursor(Qt.PointingHandCursor)
            elif over_grip:
                self.setCursor(Qt.SizeFDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            lit = (over_grip, over_expand, over_single)
            if lit != (self._grip_lit, self._expand_lit, self._single_lit):
                self._grip_lit, self._expand_lit, self._single_lit = lit
                self.update()
        super(MagnifierWidget, self).mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._grip_lit or self._expand_lit or self._single_lit:
            self._grip_lit = self._expand_lit = self._single_lit = False
            self.update()
        super(MagnifierWidget, self).leaveEvent(event)

    def over_grip(self, x, y) -> bool:
        """Whether this position is on the corner that resizes the widget."""
        return (x >= self.width() - self.resize_handle_size
                and y >= self.height() - self.resize_handle_size)

    def mouseReleaseEvent(self, event):
        if self.resizing:
            self.resizing = False
            event.accept()
            return
        super(MagnifierWidget, self).mouseReleaseEvent(event)

    # -- layout ------------------------------------------------------------

    def tiles(self):
        """``(rect, point_index)`` for each view to draw.

        One handle being dragged gets the whole widget: that is the moment
        precision is wanted, and four thumbnails serve it worse than one clear
        view. Otherwise every handle gets a tile.
        """
        count = len(self.points)
        if not count:
            return []

        if self.whole_area and self.focus_index is None:
            if self.whole_area_point() is not None:
                return [(self.rect().adjusted(2, 2, -2, -2), -1)]

        focus = self.focus_index
        if focus is not None and 0 <= focus < count:
            return [(self.rect().adjusted(2, 2, -2, -2), focus)]

        columns, rows = self.grid_shape()
        width, height = self.width(), self.height()
        order = self._tile_order()
        out = []
        for slot in range(count):
            column, row = slot % columns, slot // columns
            x0 = width * column // columns
            x1 = width * (column + 1) // columns
            y0 = height * row // rows
            y1 = height * (row + 1) // rows
            out.append((QRect(x0, y0, x1 - x0, y1 - y0), order[slot]))
        return out

    def grid_shape(self):
        """``(columns, rows)`` for the overview."""
        if self.whole_area:
            return 1, 1
        count = len(self.points)
        if count <= 1:
            return 1, 1
        if count == 4 and all(p.kind == MagnifierPoint.CORNER for p in self.points):
            # The familiar arrangement, and worth keeping: each corner appears
            # in the part of the box where it actually sits on the video, so
            # the eye goes to the right tile without reading the labels.
            return 2, 2
        if any(p.kind == MagnifierPoint.HANDLE for p in self.points):
            # Curving is on: a row per edge, holding that edge's corner and its
            # two bend handles, so a row is one edge of the area.
            return 3, int(np.ceil(count / 3.0))
        columns = int(np.ceil(np.sqrt(count)))
        return columns, int(np.ceil(count / float(columns)))

    def _tile_order(self):
        """Which point each tile shows, for the spatial 2x2 case.

        Corners are clicked round the shape -- top-left, top-right,
        bottom-right, bottom-left -- while a grid runs left to right, top to
        bottom. Swapping the last two puts each corner in its own part of the
        box.
        """
        if self.grid_shape() == (2, 2) and len(self.points) == 4:
            return [0, 1, 3, 2]
        return list(range(len(self.points)))

    def area_bounds(self):
        """The marked area's extent, in frame coordinates, or None.

        Taken from the outline rather than the corners so a bend that swings
        wide of them is inside the view rather than cut off by it.
        """
        points = [(p.x, p.y) for p in self.points]
        if self.region is not None:
            try:
                points += [tuple(p) for p in core.region_boundary(self.region,
                                                                  samples=32)]
            except Exception:
                pass
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def whole_area_point(self):
        """A view of everything at once, centred on the marked area."""
        bounds = self.area_bounds()
        if bounds is None:
            return None
        left, top, right, bottom = bounds
        return MagnifierPoint((left + right) / 2.0, (top + bottom) / 2.0,
                              "", MagnifierPoint.CORNER, "Whole area")

    def point_at(self, index):
        """The point a tile shows. -1 is the whole-area view."""
        if index < 0:
            return self.whole_area_point()
        return self.points[index]

    def zoom_for(self, rect) -> float:
        """The magnification this tile should use."""
        if self.whole_area:
            bounds = self.area_bounds()
            if bounds is not None:
                # Room round the edges: a handle sitting exactly on the border
                # of the view is the one you cannot judge.
                margin = 1.35
                wide = max(1.0, (bounds[2] - bounds[0]) * margin)
                tall = max(1.0, (bounds[3] - bounds[1]) * margin)
                # Fits, whatever that takes. An area larger than the widget
                # cannot be both shown whole and magnified, and showing it
                # whole is the entire point of this view -- enlarging the
                # magnifier, or filling the stage with it, is what buys the
                # magnification back.
                return float(np.clip(min(rect.width() / wide,
                                         rect.height() / tall),
                                     0.05, self.MAX_ZOOM))
        if not self.auto_zoom:
            return self.zoom_factor
        shorter = max(1, min(rect.width(), rect.height()))
        return float(np.clip(shorter / self.DEFAULT_SPAN,
                             self.MIN_ZOOM, self.MAX_ZOOM))

    def tile_mapping(self, rect, point):
        """``(scale, offset_x, offset_y)`` taking frame coordinates to widget
        ones inside this tile, so that ``widget = frame * scale + offset``.

        The one definition of where the picture goes. The magnified image, the
        outline drawn over it and the crosshair all derive from this, which is
        what guarantees they agree: the handle lands exactly at the centre of
        its tile for every handle, including ones outside the frame entirely.
        """
        zoom = self.zoom_for(rect)
        return (zoom,
                rect.x() + rect.width() / 2.0 - point.x * zoom,
                rect.y() + rect.height() / 2.0 - point.y * zoom)

    def source_at(self, rect, point, x, y):
        """Which pixel of the frame lies under a position in this tile."""
        scale, offset_x, offset_y = self.tile_mapping(rect, point)
        return ((x - offset_x) / scale, (y - offset_y) / scale)

    def source_frame(self):
        """The picture this view magnifies.

        The whole-area view shows the creative in place, because that view is
        about the insert as a whole -- whether the advert sits on the screen
        and moves with it. A tile is the opposite: it is there to line one
        handle up against the real edge underneath, and the creative covers
        exactly that edge, so tiles stay on the untouched frame.
        """
        if self.whole_area and self.composited is not None:
            return self.composited
        return self.base_frame

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(18, 18, 18))

        if self.base_frame is None or not self.points:
            painter.setPen(QPen(QColor(130, 130, 130)))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Mark an area to magnify it")
            # Still offer the buttons: expanding and then clearing the shape
            # would otherwise leave the stage covered with no way back.
            self._draw_expand_button(painter)
            self._draw_single_button(painter)
            painter.end()
            return

        for rect, index in self.tiles():
            point = self.point_at(index)
            if point is not None:
                self._draw_tile(painter, rect, point, index)

        # The whole-widget border last, so no tile paints over it.
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        self._draw_zoom_badge(painter)
        self._draw_resize_grip(painter)
        self._draw_expand_button(painter)
        self._draw_single_button(painter)
        painter.end()

    def _draw_tile(self, painter, rect, point, index):
        if rect.width() < 8 or rect.height() < 8:
            return
        painter.save()
        painter.setClipRect(rect)

        zoom = self.zoom_for(rect)
        # The view is centred exactly on the handle and never slid back inside
        # the frame. Clamping it used to leave the crosshair drawn at the
        # middle of the tile while the picture behind it had been shifted, so
        # near the edge of frame the magnifier pointed at the wrong pixel --
        # precisely where placement is most fiddly. Anything past the edge is
        # simply shown as off-frame. The offsets are tile-local here, hence
        # subtracting the tile's own origin from the shared mapping.
        _, offset_x, offset_y = self.tile_mapping(rect, point)
        matrix = np.float32([[zoom, 0, offset_x - rect.x()],
                             [0, zoom, offset_y - rect.y()]])
        crisp = zoom >= self.CRISP_ABOVE
        patch = cv2.warpAffine(
            self.source_frame(), matrix, (rect.width(), rect.height()),
            flags=cv2.INTER_NEAREST if crisp else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(26, 20, 20))
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rect.width(), rect.height(),
                       3 * rect.width(), QImage.Format_RGB888)
        painter.drawImage(rect.topLeft(), image.copy())

        if zoom >= self.GRID_ABOVE:
            self._draw_pixel_grid(painter, rect, point, zoom)
        self._draw_geometry(painter, rect, point, zoom)
        if index >= 0:
            self._draw_crosshair(painter, rect, point, index)
        self._draw_label(painter, rect, point, index)

        # An outline, not a fill: drawRect uses whatever brush is current, and
        # the label's translucent black was still set, so this quietly washed
        # every tile over at a third of its brightness.
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.restore()

    def _draw_pixel_grid(self, painter, rect, point, zoom):
        """Rule off the source pixels, so an edge can be read off exactly."""
        painter.setPen(QPen(QColor(255, 255, 255, 55), 1))
        left = point.x - rect.width() / (2.0 * zoom)
        top = point.y - rect.height() / (2.0 * zoom)
        first_x = np.ceil(left)
        first_y = np.ceil(top)
        for i in range(int(rect.width() / zoom) + 2):
            x = rect.x() + (first_x + i - left) * zoom
            if rect.x() <= x <= rect.right():
                painter.drawLine(int(x), rect.y(), int(x), rect.bottom())
        for i in range(int(rect.height() / zoom) + 2):
            y = rect.y() + (first_y + i - top) * zoom
            if rect.y() <= y <= rect.bottom():
                painter.drawLine(rect.x(), int(y), rect.right(), int(y))

    def _draw_geometry(self, painter, rect, point, zoom):
        """The area's outline, and the handle lines, through this view.

        Without it a Bezier handle cannot be judged at all: what matters is
        where the *curve* ends up, not where the handle sits, and the curve is
        often nowhere near it.
        """
        if self.region is None:
            return

        scale, offset_x, offset_y = self.tile_mapping(rect, point)

        def to_tile(x, y):
            return (x * scale + offset_x, y * scale + offset_y)

        try:
            boundary = core.region_boundary(self.region, samples=48)
        except Exception:                      # a half-drawn or folded shape
            return

        painter.setPen(QPen(QColor(0, 230, 0, 200), 2))
        mapped = [to_tile(x, y) for x, y in boundary]
        for i in range(len(mapped)):
            x1, y1 = mapped[i]
            x2, y2 = mapped[(i + 1) % len(mapped)]
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        corners = self.region.corners
        # The corners themselves, so a tile shows its neighbours as well as
        # the handle it is centred on -- and, in the whole-area view, so every
        # handle is drawn where it actually is rather than one in the middle.
        painter.setBrush(Qt.NoBrush)
        picked = None
        if 0 <= self.selected_index < len(self.points):
            chosen = self.points[self.selected_index]
            picked = (chosen.x, chosen.y)
        for corner in corners:
            x, y = to_tile(*corner)
            here = (picked is not None
                    and abs(corner[0] - picked[0]) < 0.5
                    and abs(corner[1] - picked[1]) < 0.5)
            painter.setPen(QPen(QColor(0, 255, 0) if here
                                else QColor(255, 90, 90, 210), 2))
            size = 6 if here else 4
            painter.drawEllipse(int(x) - size, int(y) - size, size * 2, size * 2)

        if not self.region.curved:
            return

        controls = self.region.controls
        painter.setPen(QPen(QColor(120, 180, 255, 150), 1, Qt.DashLine))
        for edge in range(4):
            for slot in range(2):
                anchor = corners[edge] if slot == 0 else corners[(edge + 1) % 4]
                ax, ay = to_tile(*anchor)
                cx, cy = to_tile(*controls[edge, slot])
                painter.drawLine(int(ax), int(ay), int(cx), int(cy))

        # And the bend handles themselves, on the curve they are bending. A
        # handle is placed correctly when the curve it produces sits on the
        # screen's edge, which cannot be judged from a view that shows the
        # curve but not what is pulling it.
        for edge in range(4):
            for slot in range(2):
                x, y = to_tile(*controls[edge, slot])
                chosen = (edge, slot) == self.selected_control
                painter.setPen(QPen(QColor(0, 255, 255) if chosen
                                    else QColor(120, 180, 255), 2))
                painter.setBrush(QBrush(QColor(120, 180, 255, 110)))
                size = 5 if chosen else 4
                painter.drawRect(int(x) - size, int(y) - size, size * 2, size * 2)
        painter.setBrush(Qt.NoBrush)

    def _draw_crosshair(self, painter, rect, point, index):
        """The mark is always the true position of the handle.

        Drawn as a gap-centred cross rather than a solid one so the pixel being
        placed is never hidden by the thing pointing at it.
        """
        selected = index == self.selected_index
        if point.kind == MagnifierPoint.HANDLE:
            colour = QColor(0, 255, 255) if selected else QColor(120, 180, 255)
        else:
            colour = QColor(0, 255, 0) if selected else QColor(255, 60, 60)

        cx = rect.x() + rect.width() // 2
        cy = rect.y() + rect.height() // 2
        arm, gap = 12, 4

        # A dark under-stroke keeps it legible over pale footage. The coloured
        # stroke has to be the thicker part of the pair, or the outline swamps
        # it and every mark reads as black whatever it is meant to say.
        for width, shade in ((4, QColor(0, 0, 0, 150)), (2, colour)):
            painter.setPen(QPen(shade, width))
            painter.drawLine(cx - arm, cy, cx - gap, cy)
            painter.drawLine(cx + gap, cy, cx + arm, cy)
            painter.drawLine(cx, cy - arm, cx, cy - gap)
            painter.drawLine(cx, cy + gap, cx, cy + arm)

        painter.setBrush(Qt.NoBrush)
        for width, shade in ((3, QColor(0, 0, 0, 130)), (1, colour)):
            painter.setPen(QPen(shade, width))
            if point.kind == MagnifierPoint.HANDLE:
                painter.drawRect(cx - 4, cy - 4, 8, 8)
            else:
                painter.drawEllipse(cx - 4, cy - 4, 8, 8)

    def _draw_label(self, painter, rect, point, index):
        """Name the handle on a solid tab.

        Drawn on a tab rather than straight onto the picture because the
        picture is arbitrary footage: plain text was unreadable half the time
        and truncated by the tile the rest of it.
        """
        focused = self.focus_index is not None or self.whole_area
        text = point.title if focused else point.label
        if not text or rect.height() < 30:
            return

        metrics = painter.fontMetrics()
        text = metrics.elidedText(text, Qt.ElideRight, rect.width() - 12)
        tab = QRect(rect.x() + 3, rect.y() + 3,
                    metrics.width(text) + 10, metrics.height() + 2)
        selected = index == self.selected_index

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 165)))
        painter.drawRect(tab)
        painter.setPen(QPen(QColor(255, 255, 255) if selected
                            else QColor(205, 205, 205)))
        painter.drawText(tab, Qt.AlignCenter, text)

    def _draw_zoom_badge(self, painter):
        if self.height() < 40:
            return
        tiles = self.tiles()
        zoom = self.zoom_for(tiles[0][0]) if tiles else self.zoom_factor
        # Bottom left, out of the resize grip's way.
        box = self.zoom_badge_rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 140)))
        painter.drawRect(box)
        painter.setPen(QPen(QColor(150, 150, 150) if self.auto_zoom
                            else QColor(210, 210, 210)))
        # A decimal below 10x: the whole-area view often lands between one
        # and two, where rounding reported the same number at every size.
        shown = f"{zoom:.1f}x" if zoom < 10 else f"{zoom:.0f}x"
        painter.drawText(box, Qt.AlignCenter,
                         shown if self.auto_zoom else shown + " \u2022")

    def _draw_expand_button(self, painter):
        """Arrows out to fill the stage, arrows in to put it back."""
        box = self.expand_button_rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 190 if self._expand_lit else 140)))
        painter.drawRect(box)

        shade = QColor(245, 245, 245) if self._expand_lit else QColor(185, 185, 185)
        painter.setPen(QPen(shade, 2))
        painter.setBrush(Qt.NoBrush)
        left, top = box.x() + 5, box.y() + 5
        right, bottom = box.right() - 5, box.bottom() - 5
        span = 4
        if self.expanded:
            # Pointing inwards: this will shrink it back.
            painter.drawLine(left, top + span, left, top)
            painter.drawLine(left, top, left + span, top)
            painter.drawLine(right, bottom - span, right, bottom)
            painter.drawLine(right, bottom, right - span, bottom)
        else:
            painter.drawLine(left, top, left + span, top)
            painter.drawLine(left, top, left, top + span)
            painter.drawLine(right, bottom, right - span, bottom)
            painter.drawLine(right, bottom, right, bottom - span)
        painter.drawLine(left + 1, top + 1, right - 1, bottom - 1)

    def _draw_single_button(self, painter):
        """Four panes to go back to a tile each, one to show the whole area.

        Drawn as what pressing it does rather than as the state it is in,
        which is the only reading that tells someone who has not used it
        before what will happen.
        """
        box = self.single_button_rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 190 if self._single_lit else 140)))
        painter.drawRect(box)

        shade = QColor(245, 245, 245) if self._single_lit else QColor(185, 185, 185)
        painter.setPen(QPen(shade, 1))
        painter.setBrush(Qt.NoBrush)
        inner = box.adjusted(5, 5, -5, -5)
        if self.whole_area:
            middle_x = inner.x() + inner.width() // 2
            middle_y = inner.y() + inner.height() // 2
            painter.drawRect(inner)
            painter.drawLine(middle_x, inner.y(), middle_x, inner.bottom())
            painter.drawLine(inner.x(), middle_y, inner.right(), middle_y)
        else:
            painter.drawRect(inner)

    def _draw_resize_grip(self, painter):
        """The usual diagonal ribs in the corner.

        The corner has always resized the widget, but nothing said so: there
        was no mark of any kind, so the only way to find it was to already
        know. Three ribs is what every other resizable corner looks like.
        """
        if self.expanded:
            return          # it fills the stage; there is nothing to drag
        painter.setBrush(Qt.NoBrush)
        shade = QColor(235, 235, 235) if self._grip_lit else QColor(150, 150, 150)
        painter.setPen(QPen(shade, 2))
        right, bottom = self.width() - 4, self.height() - 4
        for offset in (4, 9, 14):
            painter.drawLine(right - offset, bottom, right, bottom - offset)

class Placement:
    """One insertion: an area tracked through the clip, and what goes in it.

    A single shot of a mall or an airport concourse routinely has several
    screens worth filling, and an A/B test wants different creatives in them.
    Everything that used to be a single set of attributes on the overlay lives
    here instead, one instance per insertion, each with its own tracker.
    """

    _counter = 0

    def __init__(self, name: str = None):
        Placement._counter += 1
        self.name = name or f"Placement {Placement._counter}"
        self.enabled = True

        # Geometry
        self.points = []
        self.curvature = np.zeros((4, 2, 2), np.float32)
        self.curved_enabled = False
        self.tracking_history = {}

        # Creative
        self.overlay_bgra = None
        self.overlay_source_path = None
        self.inserted_overlay_is_video = False
        self.overlay_video_cap = None
        self.overlay_video_path = None
        self.inserted_overlay_start_frame = 0

        # Look
        self.brightness = 0
        self.contrast = 1.0
        self.colourise_enabled = False
        self.colourise_factor = 0
        self.blend = blend.BlendSettings()
        # Bend and tilt applied to the creative itself, before it is warped
        # into the tracked area. Kept off the quad on purpose: the tracking
        # describes where the surface is and must not be nudged to fake the
        # look of the artwork on it.
        self.morph = morphlib.Morph()
        # Shaping a 1080p creative costs tens of milliseconds, and for a still
        # creative with settings nobody is dragging the answer is the same on
        # every frame of playback and of a render.
        self.shape_cache = morphlib.ShapeCache()

        # Tracking state, rebuilt whenever the shape is edited.
        self.feature_source = core.PlanarTracker.INTERIOR
        self.tracker = None
        self.kalman_filters = []
        self._previous_quad = None

    def has_overlay(self) -> bool:
        return self.overlay_bgra is not None or self.inserted_overlay_is_video

    def is_ready(self) -> bool:
        """True if this placement can actually draw something."""
        return self.enabled and len(self.points) == 4 and self.has_overlay()

    def reset_tracker(self):
        self.tracker = None
        self.kalman_filters = []
        self._previous_quad = None

    def release(self):
        if self.overlay_video_cap:
            self.overlay_video_cap.release()
            self.overlay_video_cap = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "points": [list(p) for p in self.points],
            "curvature": self.curvature.tolist(),
            "curved": self.curved_enabled,
            "tracking_history": core.history_to_lists(self.tracking_history),
            "overlay_path": self.overlay_source_path,
            "overlay_is_video": self.inserted_overlay_is_video,
            "overlay_start_frame": self.inserted_overlay_start_frame,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "colourise": self.colourise_enabled,
            "blend": self.blend.to_dict(),
            "morph": self.morph.to_dict(),
            "feature_source": self.feature_source,
        }

    @classmethod
    def from_dict(cls, data) -> "Placement":
        placement = cls(data.get("name"))
        placement.enabled = bool(data.get("enabled", True))
        placement.points = [tuple(p) for p in data.get("points", [])]
        if data.get("curvature") is not None:
            placement.curvature = np.array(data["curvature"], np.float32).reshape(4, 2, 2)
        placement.curved_enabled = bool(data.get("curved", False))
        placement.tracking_history = {
            int(k): [tuple(p) for p in v]
            for k, v in (data.get("tracking_history") or {}).items()}
        placement.overlay_source_path = data.get("overlay_path")
        placement.inserted_overlay_is_video = bool(data.get("overlay_is_video", False))
        placement.inserted_overlay_start_frame = data.get("overlay_start_frame", 0)
        placement.brightness = data.get("brightness", 0)
        placement.contrast = data.get("contrast", 1.0)
        placement.colourise_enabled = bool(data.get("colourise", False))
        placement.blend = blend.BlendSettings.from_dict(data.get("blend"))
        placement.morph = morphlib.Morph.from_dict(data.get("morph"))
        placement.feature_source = data.get("feature_source",
                                            core.PlanarTracker.INTERIOR)
        return placement

    def __repr__(self):
        return (f"Placement({self.name!r}, corners={len(self.points)}, "
                f"tracked={len(self.tracking_history)})")


def _active_property(attribute):
    """Expose an attribute of the active placement as if it were the overlay's.

    Nearly all of the GUI was written against a single insertion. Delegating
    rather than rewriting every call site keeps that code working unchanged and
    removes any chance of one path reading a stale copy -- the same reason the
    duplicated tracking history was collapsed into one store.
    """
    return property(lambda self: getattr(self.active, attribute),
                    lambda self, value: setattr(self.active, attribute, value))


class TrackingOverlay(QWidget):
    points_updated = pyqtSignal(int, list)
    shape_changed = pyqtSignal()
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super(TrackingOverlay, self).__init__(parent)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

        self.tracking_enabled = False
        self.drag_index = -1  # index of corner being dragged, or -1 if none
        self.selected_point_index = -1  # which corner is selected (via keys, etc.)
        self.drag_control = None      # (edge_index, control_index) while dragging
        # Which bend handle the keys are pointing at, as (edge, slot), or None
        # for one of the corners. Kept beside selected_point_index rather than
        # folded into it, the same way drag_control sits beside drag_index.
        self.selected_control = None

        # For hovering
        self.active_point_index = -1

        # Several insertions can be tracked at once -- a concourse shot often
        # has more than one screen worth filling. Editing always applies to the
        # active one; everything else is drawn dimmed and still renders.
        self.placements = [Placement()]
        self.active_index = 0

        # Widget config
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)

    # -- placements --------------------------------------------------------

    @property
    def active(self) -> Placement:
        """The placement that editing applies to."""
        if not self.placements:
            self.placements = [Placement()]
        self.active_index = max(0, min(self.active_index, len(self.placements) - 1))
        return self.placements[self.active_index]

    def add_placement(self, name: str = None) -> Placement:
        placement = Placement(name)
        self.placements.append(placement)
        self.active_index = len(self.placements) - 1
        self.selected_point_index = -1
        self.selected_control = None
        self.update()
        return placement

    def remove_placement(self, index: int) -> None:
        if not 0 <= index < len(self.placements) or len(self.placements) <= 1:
            return
        self.placements.pop(index).release()
        self.active_index = min(self.active_index, len(self.placements) - 1)
        self.update()

    def set_active(self, index: int) -> None:
        if 0 <= index < len(self.placements):
            self.active_index = index
            self.selected_point_index = -1
            self.selected_control = None
            self.shape_changed.emit()
            self.update()

    def ready_placements(self):
        """Those that can actually draw something this frame."""
        return [p for p in self.placements if p.is_ready()]

    # Everything below was written against a single insertion; these delegate
    # to the active placement so that code goes on working untouched.
    points = _active_property("points")
    curvature = _active_property("curvature")
    curved_enabled = _active_property("curved_enabled")
    tracking_history = _active_property("tracking_history")
    overlay_bgra = _active_property("overlay_bgra")
    overlay_source_path = _active_property("overlay_source_path")
    inserted_overlay_is_video = _active_property("inserted_overlay_is_video")
    overlay_video_cap = _active_property("overlay_video_cap")
    overlay_video_path = _active_property("overlay_video_path")
    inserted_overlay_start_frame = _active_property("inserted_overlay_start_frame")
    brightness = _active_property("brightness")
    contrast = _active_property("contrast")
    colourise_enabled = _active_property("colourise_enabled")
    colourise_factor = _active_property("colourise_factor")
    blend = _active_property("blend")
    morph = _active_property("morph")
    _previous_quad = _active_property("_previous_quad")

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
                    self.selected_control = hit
                    self.selected_point_index = -1
                    found_drag = True

            if found_drag:
                self.set_magnifier_visible(True)

            if not found_drag:
                if len(self.points) < 4:
                    self.points.append((rx, ry))
                    self.selected_point_index = len(self.points) - 1
                    self.update()
                    if len(self.points) == 4:
                        self.on_fourth_corner_placed()
            elif self.drag_index != -1:
                self.selected_control = None
                # Select what was actually grabbed. This used to snap the
                # selection to the last corner on every click, so pressing "1"
                # and then an arrow key moved corner 4 instead.
                self.selected_point_index = self.drag_index

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

    def set_magnifier_visible(self, visible: bool):
        """Ask the panel to show the magnifier for the duration of a drag."""
        mw = self.get_main_window()
        if mw and mw.central_panel:
            mw.central_panel.set_magnifier_temporarily_visible(visible)

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

    def commit_shape(self):
        """Record the current shape against the frame on screen.

        Every edit has to come through here. Previously only a mouse release
        did, so an arrow-key nudge -- the tool documented for precise corner
        placement -- changed what was drawn without ever changing what would be
        rendered.
        """
        mw = self.get_main_window()
        if not mw or not mw.central_panel:
            return
        panel = mw.central_panel
        if not panel.cap or not panel.cap.isOpened():
            return

        frame_index = panel.get_current_frame_index()
        if len(self.points) == 4:
            self.tracking_history[frame_index] = self.points[:]
        elif frame_index in self.tracking_history:
            del self.tracking_history[frame_index]
        mw.mark_dirty()

    def mouseReleaseEvent(self, event):
        was_dragging = (self.drag_index != -1 or self.drag_control is not None)
        self.drag_index = -1
        self.drag_control = None
        if was_dragging:
            self.set_magnifier_visible(False)

        self.commit_shape()
        self.shape_changed.emit()
        super(TrackingOverlay, self).mouseReleaseEvent(event)

    #: Which edge each key picks the bend handles of. The letters match the
    #: names the magnifier puts on them, so T goes to the tiles marked T1 and
    #: T2 without anything to learn.
    EDGE_KEYS = {Qt.Key_T: 0, Qt.Key_R: 1, Qt.Key_B: 2, Qt.Key_L: 3}

    def keyPressEvent(self, event):
        key = event.key()
        if key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5]:
            num = key - Qt.Key_0
            if num <= 4:
                self.selected_point_index = num - 1
            else:
                self.selected_point_index = 4
            self.selected_control = None

            # Immediately redraw so the newly selected corner turns green
            self.update()
            self.shape_changed.emit()
            self.selection_changed.emit()
            event.accept()
            return

        if key in self.EDGE_KEYS and self.curved_enabled and len(self.points) == 4:
            # Pressing the same edge again steps to its other handle, so four
            # keys reach all eight without a modifier to remember.
            edge = self.EDGE_KEYS[key]
            if self.selected_control and self.selected_control[0] == edge:
                slot = 1 - self.selected_control[1]
            else:
                slot = 0
            self.selected_control = (edge, slot)
            self.selected_point_index = -1
            self.update()
            self.shape_changed.emit()
            self.selection_changed.emit()
            event.accept()
            return

        # Anything not handled here must fall through to the parent, or
        # CentralPanel's D (delete shape) and C (copy from previous frame)
        # shortcuts never fire: this widget holds focus almost permanently, and
        # a bare return leaves the event marked accepted.
        arrows = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0),
                  Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}
        nothing_selected = (self.selected_point_index < 0
                            and self.selected_control is None)
        if key not in arrows or nothing_selected:
            event.ignore()
            super(TrackingOverlay, self).keyPressEvent(event)
            return

        # Shift for a coarse jump, Ctrl for sub-pixel: corner placement on a
        # distant billboard needs finer than one pixel per press.
        step = 1.0
        if event.modifiers() & Qt.ShiftModifier:
            step = 10.0
        elif event.modifiers() & Qt.ControlModifier:
            step = 0.25
        dx, dy = (v * step for v in arrows[key])

        if len(self.points) == 4:
            if self.selected_control is not None:
                # Nudging matters more here than on a corner: a bend handle is
                # a lever, so its effect along the curve is larger than the
                # distance it moves.
                edge, slot = self.selected_control
                cx, cy = self.current_region().controls[edge, slot]
                self.set_control_point(edge, slot, (cx + dx, cy + dy))
            elif self.selected_point_index < 4:
                px, py = self.points[self.selected_point_index]
                self.points[self.selected_point_index] = (px + dx, py + dy)
            else:
                self.points = [(px + dx, py + dy) for px, py in self.points]

        self.update()
        self.commit_shape()
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

    def refresh_preview(self):
        """Redraw the outline *and* the picture underneath it.

        update() repaints this widget, which carries the outline and the
        handles and nothing else; the creative is composited into the frame by
        the panel. Anything changing how the creative looks has to ask for
        both. Asking only for this one is why inserting a creative, or
        adjusting its brightness, appeared to do nothing until the video
        happened to decode another frame.
        """
        self.update()
        mw = self.get_main_window()
        if mw and mw.central_panel:
            mw.central_panel.refresh_display()

    def set_brightness(self, brightness: int):
        self.brightness = brightness
        self.refresh_preview()

    def set_contrast(self, contrast: float):
        self.contrast = contrast
        self.refresh_preview()

    def on_fourth_corner_placed(self):
        """Called once the shape is complete.

        It used to make the overlay transparent to the mouse the instant the
        fourth corner landed, so the corners became undraggable exactly when
        the user wanted to nudge them, with no visible reason. Placement
        already stops at four points, so the overlay stays live.
        """
        self.commit_shape()
        mw = self.get_main_window()
        if mw and mw.central_panel:
            # The shape is now valid, so tracking becomes available.
            mw.central_panel.update_tracking_availability()
            mw.check_for_a_playing_screen()

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
        self.refresh_preview()

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

        self.refresh_preview()

    def remove_inserted_overlay(self):
        self.overlay_bgra = None
        self.overlay_source_path = None
        self.inserted_overlay_is_video = False
        if self.overlay_video_cap:
            self.overlay_video_cap.release()
            self.overlay_video_cap = None
        self.overlay_video_path = None
        self.inserted_overlay_start_frame = 0
        self.refresh_preview()

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

    def styled_for(self, placement, base_frame, quad, motion=None) -> np.ndarray:
        """One placement's creative, adapted to the frame it is going into."""
        if placement.overlay_bgra is None:
            return None
        styled = placement.shape_cache.apply(placement.overlay_bgra,
                                             placement.morph)
        styled = core.apply_brightness_contrast(
            styled, placement.brightness, placement.contrast)
        if placement.colourise_enabled and base_frame is not None:
            styled = core.apply_colourise(styled, base_frame, quad)
        if base_frame is not None and placement.blend.enabled:
            styled = blend.photometric_match(styled, base_frame, quad,
                                             placement.blend, motion=motion)
        return styled

    def motion_for(self, placement, quad) -> tuple:
        """How far this placement's surface moved since the previous frame."""
        current = np.float32(quad) if quad is not None else None
        previous = placement._previous_quad
        placement._previous_quad = None if current is None else current.copy()
        if previous is None or current is None:
            return None
        delta = (current - previous).mean(axis=0)
        return (float(delta[0]), float(delta[1]))

    def styled_overlay(self, base_frame=None, quad=None, motion=None) -> np.ndarray:
        """The creative, adapted to the frame it is going into.

        The render calls the identical helper, so what is previewed is what is
        written to the file. Manual brightness/contrast are applied first, then
        the automatic photometric match measured from the footage itself.
        """
        if self.overlay_bgra is None:
            return None
        styled = self.active.shape_cache.apply(self.overlay_bgra, self.morph)
        styled = core.apply_brightness_contrast(
            styled, self.brightness, self.contrast)
        if self.colourise_enabled and base_frame is not None and quad is not None:
            styled = core.apply_colourise(styled, base_frame, quad)
        if base_frame is not None and quad is not None and self.blend.enabled:
            styled = blend.photometric_match(styled, base_frame, quad,
                                             self.blend, motion=motion)
        return styled

    def grain_for(self, base_frame, quad) -> float:
        """How much grain to lay over the insert, measured from the footage."""
        if base_frame is None or quad is None or not self.blend.enabled:
            return 0.0
        return blend.grain_sigma_for(base_frame, quad, self.blend)

    def motion_since(self, quad) -> tuple:
        """How far the surface moved since the last frame, for motion blur."""
        current = np.float32(quad) if quad is not None else None
        previous, self._previous_quad = self._previous_quad, (
            None if current is None else current.copy())
        if previous is None or current is None:
            return None
        delta = (current - previous).mean(axis=0)
        return (float(delta[0]), float(delta[1]))

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

        # Inactive placements first, dimmed and unlabelled, so it is obvious
        # which one an edit will apply to without hiding the others.
        for index, placement in enumerate(self.placements):
            if index == self.active_index or len(placement.points) != 4:
                continue
            region = core.Region(placement.points, placement.curvature,
                                 placement.curved_enabled)
            boundary = core.region_boundary(region, samples=32)
            display = [self.raw_to_display(x, y) for x, y in boundary]
            painter.setPen(QPen(QColor(120, 200, 120, 110), 1, Qt.DashLine))
            for j in range(len(display)):
                x1, y1 = display[j]
                x2, y2 = display[(j + 1) % len(display)]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            cx = sum(p[0] for p in display) / len(display)
            cy = sum(p[1] for p in display) / len(display)
            painter.setPen(QPen(QColor(190, 230, 190, 170), 1))
            painter.drawText(int(cx) - 30, int(cy), placement.name)

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
                    active = ((edge, slot) in (self.drag_control,
                                               self.selected_control))
                    painter.setPen(QPen(QColor(0, 255, 255) if active
                                        else QColor(120, 180, 255), 2))
                    painter.setBrush(QBrush(QColor(120, 180, 255, 150)))
                    size = 6 if active else 4
                    painter.drawRect(int(cx) - size, int(cy) - size,
                                     size * 2, size * 2)

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
        self.refresh_preview()

    def add_fifth_node(self):
        if len(self.points) == 4:
            avg_x = sum(p[0] for p in self.points) / 4.0
            avg_y = sum(p[1] for p in self.points) / 4.0
            self.points.append((avg_x, avg_y))  # 5th node

    def _color_transfer(self, source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """Kept as a thin alias; the implementation lives in galileo_core."""
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

        options_menu = QMenu("Options", self.menu)

        self.digital_screen_action = QAction(
            "Digital screen (track surroundings)", self, checkable=True)
        self.digital_screen_action.setToolTip(
            "Track the bezel and wall around the screen rather than its "
            "picture, which moves on its own")
        self.digital_screen_action.toggled.connect(self.parent.toggle_digital_screen)
        options_menu.addAction(self.digital_screen_action)

        self.occlusion_action = QAction(
            "Draw behind people", self, checkable=True)
        self.occlusion_action.setToolTip(
            "Let passers-by walk in front of the inserted creative")
        self.occlusion_action.toggled.connect(self.parent.toggle_occlusion)
        options_menu.addAction(self.occlusion_action)

        self.deflicker_action = QAction(
            "Steady the lighting", self, checkable=True)
        self.deflicker_action.setToolTip(
            "Even out pulsing from fluorescent or LED lighting")
        self.deflicker_action.toggled.connect(self.parent.toggle_deflicker)
        options_menu.addAction(self.deflicker_action)

        self.menu.addMenu(load_menu)
        self.menu.addMenu(save_menu)
        self.menu.addMenu(options_menu)
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
        # A missing file gives a null pixmap rather than raising, so the
        # except this used to sit behind could never fire. Hidden rather than
        # given placeholder text: the word "Logo" sitting beside the title is
        # more obviously missing than a title with nothing after it.
        logo = QPixmap(os.path.join(resource_directory(), "logo.png"))
        if logo.isNull():
            logging.debug("no logo.png alongside the application")
            self.logo_label.hide()
        else:
            self.logo_label.setPixmap(logo.scaled(40, 40, Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation))
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
    blendClicked = pyqtSignal()
    curveClicked = pyqtSignal(bool)
    clearClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super(LeftColumn, self).__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #1A1A1A;")

        layout = QVBoxLayout()
        # Nine tools at 52 high with 15 between them wanted 588px of column,
        # against the 580 a 1024x640 window leaves -- so Render still fell off
        # the bottom. Tighter margins and spacing give the whole set room with
        # a little to spare.
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(6)

        # Emoji render inconsistently across Windows builds and several of
        # these are hard to tell apart at 20px, so every tool carries a tooltip
        # saying what it does. There were none anywhere on this toolbar.
        # Ordered as the job is done: mark the area, then settle how the
        # creative looks on it, then finish. Render sat second, between two
        # marking tools, though it is the last thing anyone does.
        items = [
            ("🖱", "Draw", "Click four corners to mark the area to fill.\n"
                           "Drag a corner to adjust it."),
            ("〰", "Curve", "Bend the edges of the area around a curved\n"
                            "screen or pillar."),
            ("👀", "Frame", "Jump to a specific frame number."),
            ("🌞", "Brightness", "Brighten or darken the inserted creative."),
            ("⛅", "Contrast", "Raise or lower the creative's contrast."),
            ("🎨", "Colourise", "Match the creative's colours to the surface\n"
                                "it is covering."),
            ("🌫", "Blend", "Match the creative's lighting, softness, grain and\n"
                            "motion blur to the footage, so it does not look\n"
                            "pasted on."),
            ("❌", "Clear", "Remove the tracking from every frame."),
            ("📼", "Render", "Export the video with the creative composited in."),
        ]

        self.icons = []
        for (icon_char, text, hint) in items:
            icon_widget = IconWidget(icon_char, text)
            icon_widget.setToolTip(f"{text}\n\n{hint}")
            layout.addWidget(icon_widget, 0, Qt.AlignHCenter)
            icon_widget.mousePressEvent = self.make_icon_click_handler(icon_widget, icon_widget.mousePressEvent)
            self.icons.append(icon_widget)

        layout.addStretch()

        # Scroll rather than clip. If the column ever runs out of height again
        # the failure is a scrollbar the user can act on, not tools that
        # silently disappear off the bottom.
        inner = QWidget()
        inner.setStyleSheet("background-color: #1A1A1A;")
        inner.setLayout(layout)

        scroll = QScrollArea(self)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background-color: #1A1A1A; border: none; }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
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
            elif text == "Blend":
                if is_selected:
                    self.blendClicked.emit()
                icon_widget.set_selected(False)   # opens a dialog, not a mode
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
        # Whether the pass now running has actually written anything down, so
        # that ending it can account for the frames rather than mark a project
        # unsaved because tracking was switched off and straight back on.
        self.tracked_this_pass = False
        self.cap = None
        self.playing = False
        self.prev_frame = None
        # The frame as shown, creatives and all. Kept so the magnifier's
        # whole-area view can show the insert without compositing a second
        # time on every mouse move of a drag.
        self.display_frame = None
        # Lighting correction for footage shot under a flickering fitting.
        # Applied as each frame is decoded, before anything else sees it, so
        # the tracker, the blend measurements and the render all work from a
        # plate whose lighting holds still.
        self.deflicker = None
        #: What a quick look at the clip found when it was opened, so the
        #: option can say what it would be correcting.
        self.flicker_report = None
        self.fps = 30

        # The single source of truth for "which frame is on screen". Deriving
        # it from CAP_PROP_POS_MSEC in one place and CAP_PROP_POS_FRAMES in
        # another used to make them disagree by a frame, which quietly
        # misfiled tracking data and offset video creatives.
        self.current_frame_index = 0
        self.total_frames = 0

        # Person segmentation for occlusion, built on first use.
        self.segmenter = None
        self.occlusion_enabled = False

        # 1) The container for video + overlay.
        #
        # No layout manager here on purpose: layout_video_stage positions the
        # label and the overlay by hand so the picture keeps its own aspect
        # ratio. A layout would stretch it to fill, and judging whether an
        # inserted advert looks convincing on a squashed image is not possible.
        self.video_container = QWidget(self)
        self.video_container.setStyleSheet("background-color: #0D0D0D;")

        self.video_label = QLabel(self.video_container)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setScaledContents(True)

        # An empty black rectangle told a first-time user nothing at all. This
        # sits over the stage until a video is loaded and then gets out of the
        # way, naming the order of operations rather than leaving it to be
        # discovered inside an unlabelled hamburger menu.
        self.empty_state = QLabel(self.video_container)
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.empty_state.setText(
            "<div style='color:#E8E8E8; font-size:19px; font-weight:600;'>"
            "Load a base video to begin</div>"
            "<div style='color:#9A9A9A; font-size:13px; margin-top:14px;'>"
            "Menu (☰) &rarr; Load &rarr; Base Video<br><br>"
            "<span style='color:#7A7A7A;'>then</span><br><br>"
            "1 &nbsp;Mark the screen or billboard &mdash; Draw, or "
            "Find Target from Image<br>"
            "2 &nbsp;Load a creative and press Insert<br>"
            "3 &nbsp;Turn tracking on and play forward<br>"
            "4 &nbsp;Render"
            "</div>")
        self.empty_state.setStyleSheet("background-color: #0D0D0D;")

        self.tracking_overlay = TrackingOverlay(self.video_container)
        self.tracking_overlay.setFocusPolicy(Qt.StrongFocus)
        self.tracking_overlay.raise_()
        self.tracking_overlay.hide()
        self.tracking_overlay.shape_changed.connect(self.on_shape_changed)
        self.tracking_overlay.selection_changed.connect(self.update_magnifier)

        # Frame label
        self.frame_label = QLabel("Frame: 0", self)
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

        self.tracking_label = QLabel("Tracking off", self)
        self.tracking_label.setStyleSheet("""
            QLabel {
                color: #BBBBBB;
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

        self.magnifier_label = QLabel("Magnifier off", self)
        self.magnifier_label.setStyleSheet("""
            QLabel {
                color: #BBBBBB;
                font-size: 14px;
                background-color: #1A1A1A;
                padding: 5px;
                min-width: 150px;
            }
        """)
        self.magnifier_label.show()
        self.magnifier = MagnifierWidget(self.video_container)
        self.magnifier.hide()
        self._magnifier_geometry = None
        self.magnifier.expandToggled.connect(self.on_magnifier_expanded)

        # Playback controls
        self.controls_frame = QFrame(self)
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

        # Labelled, not lettered. "C" and "D" in circles were only legible to
        # someone who already knew, which is nobody on their first clip.
        self.del_btn = HoverButton("Delete")
        self.del_btn.setStyleSheet(self.button_style(radius=8))
        self.del_btn.setToolTip("Remove this frame's shape  (D)")
        self.del_btn.setFixedSize(72, 34)
        self.del_btn.clicked.connect(self.on_delete_shape)

        self.copy_btn = HoverButton("Copy")
        self.copy_btn.setStyleSheet(self.button_style(radius=8))
        self.copy_btn.setToolTip("Copy the shape from the previous frame  (C)")
        self.copy_btn.setFixedSize(72, 34)
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

        # A status strip below the transport, holding what used to float on
        # top of the picture.
        self.status_bar = QFrame()
        self.status_bar.setStyleSheet("background-color: #1A1A1A;")
        self.status_bar.setFixedHeight(46)
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(12, 4, 12, 4)
        status_layout.setSpacing(12)
        status_layout.addWidget(self.frame_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.tracking_label)
        status_layout.addWidget(self.switch)
        status_layout.addSpacing(16)
        status_layout.addWidget(self.magnifier_label)
        status_layout.addWidget(self.magnifier_switch)

        # Final panel layout. The transport and the status strip are real rows,
        # not free-floating children positioned against a phantom 16:9 box --
        # which is how the playback bar came to sit a third of the way down the
        # picture, across the very area being edited, stealing its clicks.
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_layout.addWidget(self.video_container, stretch=1)
        self.main_layout.addWidget(self.controls_frame)
        self.main_layout.addWidget(self.status_bar)
        self.setLayout(self.main_layout)

        self.slider.valueChanged.connect(self.on_slider_scrub)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider.valueChanged.connect(self.on_slider_value_changed)

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

    def button_style(self, radius: int = 20):
        """Shared look for the transport row.

        The radius is a parameter because the same style dressed both the round
        transport buttons and the two beside them; fixed at 20 it made a 72x34
        button look like a squashed circle. The fixed width and height went the
        same way -- they overrode whatever size the button had been given.
        """
        return """
            QPushButton {
                background: #3C3C3C;
                border: none;
                border-radius: %dpx;
                color: white;
            }""" % radius + """
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

        # Show this frame's shape if it has one, otherwise keep what is on
        # screen so the user can carry it to a frame that needs marking.
        if frame_index in self.tracking_history:
            self.tracking_overlay.points = self.tracking_history[frame_index][:]

        self.tracking_overlay.update()

        # (Optional) Set focus if you need it.
        if self.tracking_mode:
            self.tracking_overlay.setFocus()

    def toggle_play_pause(self):
        self.playing = not self.playing
        logging.debug("Toggling play: %s", self.playing)
        if self.playing:
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            interval_ms = max(1, int(1000 / self.fps))
            self.timer.start(interval_ms)
        else:
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()
            # Pausing is how most passes end: the user plays as far as they
            # want, stops, and looks at what was recorded.
            self.note_tracking_pass_ended()

        if self.tracking_mode:
            self.tracking_overlay.setFocus()

    def disable_auto_tracking(self):
        """Turns off the tracking mode and updates the UI switch."""
        self.tracking_mode = False
        self.switch.setChecked(False)
        self.refresh_tracking_label()
        self.note_tracking_pass_ended()

    def refresh_tracking_label(self):
        """Say whether tracking is on and, with several placements, on which.

        Naming it matters: only the active placement is followed, so a user who
        thinks a pass is covering every screen in the shot would come away
        believing work had happened that had not.
        """
        overlay = self.tracking_overlay
        several = len(overlay.placements) > 1
        if self.tracking_mode:
            colour = "limegreen"
            tip = ("Following this placement only. Switch placement and play "
                   "again to track another." if several else
                   "Following the marked area through the clip.")
        else:
            colour = "#BBBBBB"
            tip = "Nothing is being followed; shapes stay where they are."
        self.tracking_label.setStyleSheet(f"""
            QLabel {{
                color: {colour};
                font-size: 14px;
                background-color: #1A1A1A;
                padding: 5px;
            }}
        """)

        if not self.tracking_mode:
            text = "Tracking off"
        elif not several:
            text = "Tracking on"
        else:
            # Placement names are the user's own words and get long -- "Ryanair
            # concourse portrait" is a realistic one. Shorten it rather than let
            # it push the magnifier controls off the end of the strip.
            # fontMetrics() rather than font(): the size comes from the
            # stylesheet, so only the resolved font measures the text right.
            metrics = self.tracking_label.fontMetrics()
            text = "Tracking " + metrics.elidedText(overlay.active.name,
                                                    Qt.ElideRight, 170)
            tip = f"{tip}\nTracking: {overlay.active.name}"
        self.tracking_label.setText(text)
        self.tracking_label.setToolTip(tip)
        # The strip hands this label its minimum width and no more, so without
        # this a longer name was simply cut off mid-word.
        self.tracking_label.setMinimumWidth(
            max(150, self.tracking_label.sizeHint().width()))

    def on_shape_changed(self):
        # A hand edit invalidates the tracker's anchor, so drop it; it will be
        # rebuilt from the corrected shape on the next tracked frame.
        self.reset_tracker()
        if len(self.tracking_overlay.points) == 4 and self.prev_frame is not None:
            self.refresh_display()
        self.update_magnifier()

    def refresh_display(self):
        """Redraw the current frame with the creative composited in.

        Used when the shape or a creative setting changes while playback is
        paused, so the preview updates without having to step a frame.
        """
        if self.prev_frame is None:
            self.display_frame = None
            return
        display_frame = self.composite_placements(self.prev_frame)
        self.display_frame = display_frame

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    @property
    def tracker(self):
        """The active placement's tracker.

        Tracking state moved onto the placement so that several insertions can
        be followed at once without interfering; this keeps the panel-level
        name working for the code and tests written against a single one.
        """
        return self.tracking_overlay.active.tracker

    @tracker.setter
    def tracker(self, value):
        self.tracking_overlay.active.tracker = value

    @property
    def kalman_filters(self):
        return self.tracking_overlay.active.kalman_filters

    @kalman_filters.setter
    def kalman_filters(self, value):
        self.tracking_overlay.active.kalman_filters = value

    @property
    def feature_source(self):
        """Where the active placement's tracker looks for features.

        Per placement rather than global: one shot can hold a printed poster,
        whose own artwork is rigid, alongside a digital screen, whose picture
        is not.
        """
        return self.tracking_overlay.active.feature_source

    @feature_source.setter
    def feature_source(self, value):
        self.tracking_overlay.active.feature_source = value

    @property
    def tracking_history(self):
        """The one and only store of tracked shapes.

        This used to be a second dict kept "as a mirror" of the overlay's, and
        the two drifted apart: the preview read this one while the render, the
        project file and the AOI export read the overlay's. A shape drawn by
        hand was therefore invisible when you navigated back to its frame, yet
        still present in the exported video. Aliasing the property removes the
        possibility rather than relying on both being updated.
        """
        return self.tracking_overlay.tracking_history

    @tracking_history.setter
    def tracking_history(self, value):
        self.tracking_overlay.tracking_history = value

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
            # Running off the end of the clip ends the pass as surely as
            # switching tracking off does.
            self.note_tracking_pass_ended()
            return

        current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.current_frame_index = current_frame_index

        # Before anything else looks at it. Tracking, the blend's measurements
        # of the surrounding footage and the preview all have to see the same
        # corrected plate the render will, or the advert would be matched to
        # lighting that is not what ends up in the file.
        if self.deflicker is not None:
            frame = self.deflicker.apply(frame, current_frame_index)
        if not self.slider.isSliderDown():
            # Moving the slider to follow playback must not be mistaken for the
            # user scrubbing. Without this, setValue fired valueChanged, which
            # seeked back to this same frame and called read_frame again: every
            # frame was decoded and tracked twice, and the Kalman filters got
            # two predict steps for one frame of real motion, so the shape ran
            # steadily ahead of the surface during a camera move.
            self.slider.blockSignals(True)
            self.slider.setValue(current_frame_index)
            self.slider.blockSignals(False)

        overlay = self.tracking_overlay

        # --- TRACKING ---------------------------------------------------
        # The tracker follows the texture across the whole region and fits a
        # homography to it; the Kalman filters then smooth the result and
        # carry the shape through brief dropouts.
        if self.tracking_mode:
            # Only the placement being worked on is followed. Tracking them all
            # at once read as a convenience, but a second pass -- to follow a
            # screen added later, say -- re-tracked the first placement too and
            # overwrote its history from wherever its corners happened to sit,
            # throwing away corrections made by hand with nothing on screen to
            # say it had happened. The others are not frozen: they replay their
            # own recorded track, which is read and never written.
            active = overlay.active
            tracked = self.track_placement(active, frame, current_frame_index)
            self.replay_recorded_shapes(current_frame_index,
                                        being_tracked=active if tracked else None)
        else:
            # Tracking off: every placement shows whatever was recorded for
            # this frame.
            self.replay_recorded_shapes(current_frame_index)

        # If an overlay video is inserted, advance it to the matching frame.
        if overlay.inserted_overlay_is_video:
            overlay_index = current_frame_index - overlay.inserted_overlay_start_frame
            overlay.update_overlay_video_frame_by_index(overlay_index)

        # --- DISPLAY ----------------------------------------------------
        # Composite through the same code path the renderer uses, so the
        # preview is a true preview rather than a lookalike.
        self.prev_frame = frame.copy()
        display_frame = self.composite_placements(frame)
        self.display_frame = display_frame

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

    def video_aspect(self) -> float:
        """Width divided by height of the loaded video, 16:9 when none."""
        if self.prev_frame is not None:
            h, w = self.prev_frame.shape[:2]
            if h > 0:
                return w / float(h)
        if self.cap and self.cap.isOpened():
            w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if w and h:
                return w / float(h)
        return 16.0 / 9.0

    def layout_video_stage(self):
        """Letterbox the picture inside its container without distorting it.

        The label and the overlay are given *identical* geometry, matching the
        picture exactly. That is what keeps display_to_raw/raw_to_display
        correct with no offset term: mouse positions arrive relative to the
        overlay, whose origin is the picture's top-left corner, and whose size
        is the picture's size. Scaling the pixmap inside a larger label instead
        would silently break every click, every corner drag and the magnifier,
        with no visible symptom until the export was wrong.
        """
        available_w = max(1, self.video_container.width())
        available_h = max(1, self.video_container.height())
        aspect = self.video_aspect()

        width = available_w
        height = int(round(width / aspect))
        if height > available_h:
            height = available_h
            width = int(round(height * aspect))

        x = (available_w - width) // 2
        y = (available_h - height) // 2
        for widget in (self.video_label, self.tracking_overlay):
            widget.setGeometry(x, y, max(1, width), max(1, height))

        self.empty_state.setGeometry(0, 0, available_w, available_h)
        self.empty_state.setVisible(self.cap is None or not self.cap.isOpened())
        self.empty_state.raise_()

        if self.magnifier.expanded:
            self.magnifier.setGeometry(x, y, max(1, width), max(1, height))
        else:
            self.magnifier.move(*self.magnifier_corner(x, y, width, height))
        self.magnifier.raise_()
        self.tracking_overlay.raise_()

    def resizeEvent(self, event):
        super(CentralPanel, self).resizeEvent(event)
        self.main_layout.activate()
        self.layout_video_stage()

    def update_tracking_availability(self):
        """Only offer tracking once there is a shape to track.

        The switch used to accept a click with no shape drawn: it would turn
        green, say "Tracking Enabled", and record nothing for the whole clip,
        because read_frame gates on having four corners. The guard for this
        existed but was never connected to anything.
        """
        ready = (len(self.tracking_overlay.points) == 4
                 and self.cap is not None and self.cap.isOpened())
        self.switch.setEnabled(ready)
        self.switch.setToolTip(
            "Follow the marked area through the clip" if ready
            else "Mark all four corners of the area first")
        if not ready and self.switch.isChecked():
            self.switch.setChecked(False)
        # Called whenever the active placement changes, which is exactly when
        # the name in the tracking label goes stale.
        self.refresh_tracking_label()

    def on_tracking_toggled(self, on: bool):
        if on and len(self.tracking_overlay.points) != 4:
            self.switch.setChecked(False)
            QMessageBox.information(
                self, "Cannot Enable Tracking",
                "Mark all four corners of the area before turning tracking on.")
            return

        self.tracking_mode = on
        self.refresh_tracking_label()
        if not on:
            self.note_tracking_pass_ended()
            # The shape deliberately survives. Turning tracking off means "stop
            # following the surface", not "throw away what I marked" -- and
            # rewinding, scrubbing backwards and Go-to-Frame all switch tracking
            # off for you, so clearing here destroyed work without any warning
            # and with no undo.
            self.tracking_overlay.update()

    def track_with_optical_flow(self, old_frame, new_frame, old_points):
        """Advance a quad by one frame.

        Superseded by :class:`galileo_core.PlanarTracker`, which tracks the
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

    def composite_placements(self, frame):
        """Draw every ready placement into the frame.

        Occlusion is segmented once and reused: the model is the slow part and
        the mask is the same whichever insert is being drawn behind it.
        """
        overlay = self.tracking_overlay
        ready = overlay.ready_placements()
        if not ready:
            return frame

        occlusion = None
        if self.occlusion_enabled:
            occlusion = self.occlusion_mask(frame, None)

        out = frame
        for placement in ready:
            region = core.Region(placement.points, placement.curvature,
                                 placement.curved_enabled)
            if not core.is_valid_region(region):
                continue
            motion = overlay.motion_for(placement, placement.points)
            styled = overlay.styled_for(placement, frame, placement.points, motion)
            if styled is None:
                continue
            out = core.composite_region(out, styled, region, occlusion=occlusion,
                                        in_place=(out is not frame))
            sigma = blend.grain_sigma_for(frame, region.corners, placement.blend) \
                if placement.blend.enabled else 0.0
            if sigma > 0.1:
                h, w = out.shape[:2]
                out = blend.add_grain(out, sigma,
                                      core.quad_to_mask(region.corners, w, h),
                                      seed=self.current_frame_index)
        return out

    def apply_grain(self, composited, source_frame, region):
        """Lay matched sensor noise over the insert.

        Applied after the warp, in frame space: grain added to the creative
        beforehand would be resampled along with it and come out the wrong size
        for the shot.
        """
        overlay = self.tracking_overlay
        sigma = overlay.grain_for(source_frame, region.corners)
        if sigma <= 0.1:
            return composited
        h, w = composited.shape[:2]
        mask = core.quad_to_mask(region.corners, w, h)
        return blend.add_grain(composited, sigma, mask, seed=self.current_frame_index)

    def track_placement(self, placement, frame, frame_index) -> bool:
        """Follow one placement's surface into this frame and record it.

        Returns whether anything was recorded, so the caller knows which
        placement it has just written and can leave that one alone.
        """
        if not placement.enabled or len(placement.points) != 4:
            return False

        if not placement.kalman_filters:
            placement.kalman_filters = core.make_filters(placement.points)
        if placement.tracker is None and self.prev_frame is not None:
            placement.tracker = core.PlanarTracker(
                self.prev_frame, placement.points,
                feature_source=placement.feature_source)

        measurement = None
        if placement.tracker is not None:
            result = placement.tracker.track(frame)
            if result.ok:
                measurement = result.quad
            else:
                logging.debug("Tracking failed on frame %d for %s: %s",
                              frame_index, placement.name, result.reason)

        smoothed = core.smooth_quad(placement.kalman_filters, measurement)
        placement.points = [(float(x), float(y)) for x, y in smoothed]
        placement.tracking_history[frame_index] = placement.points[:]
        self.tracked_this_pass = True
        return True

    def note_tracking_pass_ended(self):
        """Account for the frames a tracking pass just recorded.

        A pass writes straight into the placement's history and nothing else
        noticed. The placement list -- the one readout of how far each
        placement has been followed, and the only one now that a pass covers a
        single placement -- went on showing the count from before it started;
        and the unsaved-work flag stayed clear, so tracking a whole clip and
        then closing threw the lot away without a prompt. A shape nudged by
        hand marked the project dirty; two hundred tracked frames did not.
        """
        if not self.tracked_this_pass:
            return
        self.tracked_this_pass = False
        window = self.tracking_overlay.get_main_window()
        if window is not None and hasattr(window, "placement_list"):
            # mark_dirty refreshes the list and the title together.
            window.mark_dirty()

    def replay_recorded_shapes(self, frame_index, being_tracked=None):
        """Put every other placement onto the shape recorded for this frame.

        Read-only by design: this is what stops a pass over one placement from
        altering another's history. A placement with nothing recorded here
        keeps the shape it has -- it has not been tracked at this frame, and
        moving it anyway would be a guess written down as measurement.
        """
        for placement in self.tracking_overlay.placements:
            if placement is being_tracked:
                continue
            # Its tracker is anchored to a frame this pass has moved away
            # from, so drop it rather than let it resume from a stale anchor;
            # it re-anchors from the current shape when the user comes back to
            # this placement. _previous_quad survives on purpose -- motion blur
            # still needs to know how far the shape moved between frames.
            placement.tracker = None
            placement.kalman_filters = []
            recorded = placement.tracking_history.get(frame_index)
            if recorded:
                placement.points = recorded[:]

    def occlusion_mask(self, frame, quad):
        """A mask of people in front of the surface, or None if switched off."""
        if not self.occlusion_enabled or frame is None:
            return None
        if self.segmenter is None:
            try:
                self.segmenter = core.PersonSegmenter()
            except (FileNotFoundError, IOError) as exc:
                logging.warning("Occlusion unavailable: %s", exc)
                self.occlusion_enabled = False
                return None
        try:
            return self.segmenter.mask(frame, quad)
        except cv2.error as exc:
            logging.warning("Segmentation failed on this frame: %s", exc)
            return None

    def reset_tracker(self):
        """Re-anchor every placement's tracking to the shapes as they stand.

        Called whenever the user edits the shape by hand: without this the
        tracker would keep propagating from the pre-edit corners and drag the
        correction straight back out again.
        """
        self.tracker = None
        self.kalman_filters = []
        for placement in self.tracking_overlay.placements:
            placement.reset_tracker()

    def on_magnifier_expanded(self, expanded: bool):
        """Fill the video stage with the magnifier, or put it back.

        Twelve handles in a floating box a couple of hundred pixels wide gives
        each one less room than the four quadrants had, which is worse than
        what it replaced. Filling the stage is what makes the crowded layouts
        usable, so the size it had before is remembered and restored.
        """
        magnifier = self.magnifier
        if expanded:
            self._magnifier_geometry = magnifier.geometry()
            magnifier.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
        elif getattr(self, "_magnifier_geometry", None) is not None:
            magnifier.setGeometry(self._magnifier_geometry)
        self.layout_video_stage()
        magnifier.raise_()
        self.update_magnifier(force=True)

    #: Smallest a single magnified view may be before it stops being one.
    MAGNIFIER_TILE = 96

    def magnifier_corner(self, x, y, width, height):
        """Which corner of the picture the magnifier should sit in.

        It used to sit in the top left whatever was underneath. Now that it is
        sized to hold twelve views it is large enough to cover the very area
        being worked on, which is the one thing it must not hide -- so it takes
        whichever corner the marked area reaches into least.
        """
        margin = 10
        box_w, box_h = self.magnifier.width(), self.magnifier.height()
        corners = {
            "top left": (x + margin, y + margin),
            "top right": (x + width - box_w - margin, y + margin),
            "bottom left": (x + margin, y + height - box_h - margin),
            "bottom right": (x + width - box_w - margin,
                             y + height - box_h - margin),
        }

        points = self.tracking_overlay.points
        if len(points) < 2:
            return corners["top left"]

        shape = [self.tracking_overlay.raw_to_display(px, py) for px, py in points]
        shape_box = (min(p[0] for p in shape) + x, min(p[1] for p in shape) + y,
                     max(p[0] for p in shape) + x, max(p[1] for p in shape) + y)

        def covered(position):
            left, top = position
            return self._overlap((left, top, left + box_w, top + box_h), shape_box)

        return min(corners.values(), key=covered)

    @staticmethod
    def _overlap(a, b) -> float:
        """Area shared by two boxes given as (left, top, right, bottom)."""
        wide = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        tall = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return wide * tall

    def update_magnifier_dimensions(self):
        shape_pts = self.tracking_overlay.points
        if len(shape_pts) < 2:
            return
        # Skip auto-resizing if the user has manually resized, or if it is
        # filling the stage and its size is not its own to choose.
        if (self.magnifier.resizing or self.magnifier.user_resized
                or self.magnifier.expanded):
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

        # Enough room for the views it actually has to draw. Sized for the
        # shape alone, switching curved edges on divided the same 150 pixels
        # twelve ways instead of four, leaving each handle a 50x37 thumbnail
        # -- smaller than the thing it was meant to magnify.
        columns, rows = self.magnifier.grid_shape()
        computed_w = max(computed_w, columns * self.MAGNIFIER_TILE)
        computed_h = max(computed_h, rows * self.MAGNIFIER_TILE)

        current_w = self.magnifier.width()
        current_h = self.magnifier.height()
        new_w = current_w
        new_h = current_h
        if computed_w > current_w:
            new_w = computed_w
        if computed_h > current_h:
            new_h = computed_h
        if (new_w != current_w) or (new_h != current_h):
            # resize, not setFixedSize: the latter pins the minimum and maximum
            # together, so the drag handle's own resize() silently did nothing
            # and the magnifier could not be enlarged by hand from the moment
            # it first sized itself -- which is as soon as two corners exist.
            self.magnifier.resize(new_w, new_h)

    def set_magnifier_temporarily_visible(self, visible: bool):
        """Show the magnifier while a corner is being dragged.

        Precise placement is exactly when a zoomed view is wanted, so it
        appears on press and goes away on release -- unless the user has turned
        it on properly, in which case it stays.
        """
        if self.magnifier_switch.isChecked():
            return          # already on for good; leave it alone
        self.magnifier.setVisible(visible)
        if visible:
            self.magnifier.raise_()
            self.update_magnifier(force=True)

    def update_magnifier(self, force: bool = False):
        if not force and not self.magnifier_switch.isChecked() \
                and not self.magnifier.isVisible():
            return

        if self.prev_frame is None:
            self.magnifier.setData(None, [], selected_index=-1)
            return

        overlay = self.tracking_overlay
        points, selected, focus = self.magnifier_points()
        # Only when a creative is actually in place; otherwise the composited
        # frame is just the frame again, and the whole-area view should say so
        # by falling back rather than holding a second reference to it.
        composited = (self.display_frame if overlay.ready_placements() else None)
        self.magnifier.setData(
            base_frame=self.prev_frame.copy(),
            overlay_points=points,
            selected_index=selected,
            region=overlay.current_region() if len(overlay.points) == 4 else None,
            focus_index=focus,
            selected_control=overlay.drag_control or overlay.selected_control,
            composited=composited,
        )
        # Sized after the views are set, not before: the grid it has to fit
        # depends on how many handles there are, and the widget only learns
        # that from the call above.
        self.update_magnifier_dimensions()

    def magnifier_points(self):
        """What the magnifier should show: ``(points, selected, focus)``.

        With curving on there are twelve handles to place, not four, and the
        eight that bend the edges were the ones that most needed magnifying --
        they are small, they sit on top of the outline, and a pixel of error in
        one is plainly visible as a curve that misses the screen. They were the
        only handles the magnifier never showed.
        """
        overlay = self.tracking_overlay
        corners = overlay.points[:]
        points = [MagnifierPoint(x, y, str(i + 1), MagnifierPoint.CORNER,
                                 f"Corner {i + 1}")
                  for i, (x, y) in enumerate(corners)]
        selected = overlay.selected_point_index
        focus = overlay.drag_index if overlay.drag_index != -1 else None

        if not (overlay.curved_enabled and len(corners) == 4):
            return points, selected, focus

        # A row per edge: its starting corner, then the two handles that bend
        # it, which is the order the grid reads them in.
        region = overlay.current_region()
        edge_names = ("Top", "Right", "Bottom", "Left")
        rows, dragged = [], overlay.drag_control
        for edge in range(4):
            row = [points[edge]]
            for slot in range(2):
                cx, cy = region.controls[edge, slot]
                initial = edge_names[edge][0]
                row.append(MagnifierPoint(
                    cx, cy, f"{initial}{slot + 1}", MagnifierPoint.HANDLE,
                    f"{edge_names[edge]} edge, bend {slot + 1}"))
            rows.append(row)

        # Corner i moved from position i to position 3i once the bend handles
        # were interleaved, so every index has to be carried across with it.
        ordered = [p for row in rows for p in row]
        chosen = dragged or overlay.selected_control
        if chosen is not None:
            edge, slot = chosen
            selected = edge * 3 + slot + 1
            # Filling the widget with it is right while it is being dragged.
            # Picking it with a key is how someone lines a handle up against
            # the rest of the shape, so the other views stay put.
            focus = selected if dragged is not None else focus
        else:
            if overlay.drag_index != -1:
                focus = overlay.drag_index * 3
            if 0 <= selected < 4:
                selected = selected * 3
        return ordered, selected, focus

    def load_video(self, file_path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(file_path)
        if self.cap.isOpened():
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setEnabled(True)
            self.slider.setRange(0, total_frames - 1)
            # Connections are made once, in __init__. Making them here meant
            # loading a second video doubled every one of them.
            self.slider.blockSignals(True)
            self.slider.setValue(0)
            self.slider.blockSignals(False)
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
        self.layout_video_stage()   # the aspect ratio is only known now
        logging.debug(f"Loaded {file_path}, FPS={self.fps}")

        self.tracking_overlay.reset()
        self.update_tracking_availability()
        self.scan_for_flicker()
        mw = self.window()
        if isinstance(mw, QMainWindow) and hasattr(mw, "refresh_title"):
            mw.refresh_title()
        if isinstance(mw, QMainWindow) and hasattr(mw, "refresh_deflicker_action"):
            mw.refresh_deflicker_action()

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
            self.magnifier_label.setText("Magnifier on")
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
            self.magnifier_label.setText("Magnifier off")
            self.magnifier_label.setStyleSheet("""
                QLabel {
                    color: #BBBBBB;
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

    # -- flickering lighting ------------------------------------------------

    def scan_for_flicker(self):
        """Take a quick look at the clip for pulsing lighting.

        A window from the middle rather than the whole clip: this runs on every
        video that is opened, and a full pass over a long one would be a stall
        on each load for footage that usually turns out to be fine. The full
        measurement only happens if the correction is actually switched on.
        """
        self.deflicker = None
        self.flicker_report = None
        path = getattr(self, "current_video_path", None)
        if not path:
            return None
        try:
            self.flicker_report = deflicker.scan(path)
        except cv2.error as exc:
            logging.warning("Could not scan %s for flicker: %s", path, exc)
        return self.flicker_report

    def set_deflicker(self, on: bool, progress=None):
        """Switch the lighting correction on or off.

        Measured over the whole clip and then applied to each frame as it is
        decoded, rather than by writing a corrected copy of the footage:
        nothing is re-encoded, so there is no generation loss and no second
        file the size of the original, and switching it off again is free.

        Returns the corrector, or None if there was nothing to correct or the
        measurement was cancelled.
        """
        if not on:
            self.deflicker = None
            self.refresh_display()
            return None

        path = getattr(self, "current_video_path", None)
        if not path:
            return None
        mode = "auto"
        if self.flicker_report is not None and self.flicker_report.kind != "steady":
            mode = self.flicker_report.kind
        self.deflicker = deflicker.Deflicker.measure(path, mode=mode,
                                                     progress=progress)
        # The frame on screen was decoded before this existed, so re-read it
        # rather than leave the preview a frame behind the setting.
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            self.read_frame()
        return self.deflicker


class RenderDialog(QDialog):
    """One dialog for the whole render, showing what is actually tracked.

    This replaces four chained prompts that asked for numbers with no context.
    The frame range defaulted to the entire clip regardless of how much had
    been tracked, and since the renderer leaves untracked frames untouched, the
    usual first-run result was a video in which the advert appeared on a single
    frame -- reported as "exported successfully".
    """

    def __init__(self, total_frames, tracked_range, video_size, fps,
                 suggested_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Render Video")
        self.setMinimumWidth(520)
        self.setStyleSheet("""
            QDialog { background-color: #2A2A2A; }
            QLabel { color: #E8E8E8; font-size: 13px; }
            QSpinBox, QComboBox, QLineEdit {
                background-color: #1E1E1E; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 4px; font-size: 13px;
            }
            QPushButton {
                background-color: #3C3C3C; color: white; border: none;
                border-radius: 6px; padding: 7px 16px; font-size: 13px;
            }
            QPushButton:hover { background-color: #505050; }
        """)

        self.video_size = video_size
        self.tracked_range = tracked_range
        self.fps = fps or 30.0

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        first, last = tracked_range
        summary = QLabel(
            f"<b>Tracked:</b> frames {first}–{last} "
            f"of {total_frames}  ({(last - first + 1) / self.fps:.1f}s)")
        layout.addWidget(summary)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Frames"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, max(0, total_frames - 1))
        self.start_spin.setValue(first)
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, max(0, total_frames - 1))
        self.end_spin.setValue(last)
        range_row.addWidget(self.start_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.end_spin)
        range_row.addStretch(1)
        layout.addLayout(range_row)

        # Resolution as a plain choice showing real pixel sizes, rather than
        # asking for a multiplier.
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Resolution"))
        self.scale_combo = QComboBox()
        for label, factor in (("Full", 1.0), ("Half", 0.5), ("Quarter", 0.25)):
            w = int(video_size[0] * factor)
            h = int(video_size[1] * factor)
            self.scale_combo.addItem(f"{label}  —  {w}×{h}", factor)
        scale_row.addWidget(self.scale_combo)
        scale_row.addStretch(1)
        layout.addLayout(scale_row)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Save to"))
        self.path_edit = QLineEdit(suggested_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        layout.addLayout(path_row)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #FFB454; font-size: 12px;")
        layout.addWidget(self.warning)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Render")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, alignment=Qt.AlignRight)

        for spin in (self.start_spin, self.end_spin):
            spin.valueChanged.connect(self._update_warning)
        self._update_warning()

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Rendered Video", self.path_edit.text(), "MP4 Files (*.mp4)")
        if path:
            self.path_edit.setText(path)

    def _update_warning(self):
        """Say plainly when the chosen range reaches past the tracking."""
        first, last = self.tracked_range
        start, end = self.start_spin.value(), self.end_spin.value()
        gaps = []
        if start < first:
            gaps.append(f"{start}–{first - 1}")
        if end > last:
            gaps.append(f"{last + 1}–{end}")
        if gaps:
            self.warning.setText(
                "⚠  No tracking on frames " + ", ".join(gaps) +
                ". The creative will not appear on them.")
        else:
            self.warning.setText("")

    def values(self):
        return (self.start_spin.value(), self.end_spin.value(),
                self.scale_combo.currentData(), self.path_edit.text().strip())


class MatchChoiceDialog(QDialog):
    """Pick which of several found instances to turn into placements.

    The same poster often appears on more than one panel in a concourse shot.
    Rather than silently taking the strongest, all of them are offered, ticked
    by default, with the evidence for each so a marginal one can be dropped.
    """

    def __init__(self, detections, parent=None):
        super().__init__(parent)
        self.detections = detections
        self.setWindowTitle("Several Matches Found")
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QDialog { background-color: #2A2A2A; }
            QLabel { color: #E8E8E8; font-size: 13px; }
            QCheckBox { color: #E8E8E8; font-size: 12px; }
            QPushButton { background-color: #3C3C3C; color: white; border: none;
                          border-radius: 6px; padding: 7px 16px; }
            QPushButton:hover { background-color: #505050; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(QLabel(
            f"<b>{len(detections)} instances of that target were found.</b>"))
        note = QLabel("Each ticked one becomes its own placement, with its own "
                      "tracking and creative.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9A9A9A; font-size: 11px;")
        layout.addWidget(note)

        self.boxes = []
        for index, detection in enumerate(detections, start=1):
            centre = np.float32(detection.quad).mean(axis=0)
            area = core.quad_area(detection.quad)
            box = QCheckBox(
                f"{index}.  {detection.n_inliers} features, "
                f"{detection.confidence:.0%} confidence  —  "
                f"{int(area):,} px at ({int(centre[0])}, {int(centre[1])})")
            box.setChecked(True)
            layout.addWidget(box)
            self.boxes.append(box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Create placements")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, alignment=Qt.AlignRight)

    def selected(self):
        return [detection for detection, box in zip(self.detections, self.boxes)
                if box.isChecked()]


class _LegacyPlacement:
    """Presents a single-insertion RenderSettings as if it were a Placement.

    PlacementSnapshot.__init__ is the one place that decides what a render
    needs from a placement. Building the older single-insertion snapshot field
    by field meant every addition had to be remembered twice, and a forgotten
    one surfaced as an AttributeError with the render thread already running.
    Going through the same constructor makes that impossible.
    """

    def __init__(self, settings):
        self.name = "Placement 1"
        self.tracking_history = settings.history or {}
        self.curvature = settings.curvature
        self.curved_enabled = settings.curved
        self.overlay_bgra = settings.overlay_bgra
        self.inserted_overlay_is_video = bool(settings.overlay_video_path)
        self.overlay_video_path = settings.overlay_video_path
        self.inserted_overlay_start_frame = settings.overlay_start_frame
        self.brightness = settings.brightness
        self.contrast = settings.contrast
        self.colourise_enabled = settings.colourise
        self.blend = settings.blend
        self.morph = settings.morph


class PlacementSnapshot:
    """One placement, detached from the GUI for the render thread."""

    def __init__(self, placement, scale):
        self.name = placement.name
        self.history = {int(k): [(float(x), float(y)) for x, y in v]
                        for k, v in placement.tracking_history.items()
                        if v is not None and len(v) == 4}
        self.curvature = np.array(placement.curvature, np.float32)
        self.curved = bool(placement.curved_enabled)
        self.creative = (placement.overlay_bgra.copy()
                         if placement.overlay_bgra is not None
                         and not placement.inserted_overlay_is_video else None)
        self.video_path = (placement.overlay_video_path
                           if placement.inserted_overlay_is_video else None)
        self.start_frame = placement.inserted_overlay_start_frame
        self.brightness = placement.brightness
        self.contrast = placement.contrast
        self.colourise = placement.colourise_enabled
        self.blend = blend.BlendSettings.from_dict(placement.blend.to_dict())
        self.morph = morphlib.Morph.from_dict(placement.morph.to_dict())
        self.shape_cache = morphlib.ShapeCache()
        self.scale = scale
        # Per-placement decode state, filled in by the worker.
        self.capture = None
        self.cursor = -1
        self.frame = None
        self.exhausted = False
        self.previous_quad = None
        self.dense = {}


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
                 colourise=False, include_audio=True, occlusion=False,
                 blend_settings=None, placements=None, morph=None,
                 deflicker_gains=None):
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
        # A flag, not a segmenter: a cv2.dnn.Net cannot be shared across
        # threads, so the worker builds its own.
        self.occlusion = occlusion
        self.blend = blend_settings or blend.BlendSettings.off()
        self.morph = morph or morphlib.Morph()
        # Every insertion to draw. The single-placement fields above are kept
        # so existing callers and tests go on working.
        self.placements = placements or []
        # The lighting correction the preview was working from. It has to
        # reach the file too, or what was tracked and blended against a steady
        # plate would be rendered onto a flickering one. Plain arrays, so it
        # crosses to the worker thread like everything else here.
        self.deflicker = deflicker_gains


class RenderWorker(QObject):
    """Composites the creative into the base video, off the GUI thread."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # Emits a final status message or error

    def __init__(self, settings: RenderSettings):
        super().__init__()
        self.settings = settings
        self.is_canceled = False

    def _prepare_placements(self, settings, caveats):
        """Work out what each placement covers, and open its creative video.

        Returns only those with tracking inside the chosen range, so a
        placement marked but never tracked simply does not draw rather than
        aborting the whole render.
        """
        snapshots = list(settings.placements)
        if not snapshots:
            # A settings object built the old way, carrying a single insertion.
            snapshots = [PlacementSnapshot(_LegacyPlacement(settings),
                                           settings.scale_factor)]

        usable = []
        for placement in snapshots:
            placement.dense = core.interpolate_tracking(
                placement.history, settings.start_frame, settings.end_frame)
            if not placement.dense:
                continue

            if placement.video_path:
                placement.capture = cv2.VideoCapture(placement.video_path)
                if placement.capture.isOpened():
                    placement.fps = (placement.capture.get(cv2.CAP_PROP_FPS)
                                     or settings.fps)
                else:
                    placement.capture = None
                    caveats.append(
                        f"the creative video for {placement.name} could not be "
                        "opened, so it was not inserted")
                    continue
            usable.append(placement)
        return usable

    def _draw_placement(self, placement, base_frame, frame_idx, settings,
                        occlusion, size):
        """Composite one placement into the frame, if it is on screen here."""
        quad = placement.dense.get(frame_idx)
        if quad is None:
            return base_frame

        creative = placement.creative
        if placement.capture is not None and not placement.exhausted:
            # Map by time, not by frame count, so a creative shot at a
            # different frame rate plays at the right speed.
            elapsed = frame_idx - placement.start_frame
            if elapsed >= 0:
                wanted = int(round(elapsed * getattr(placement, "fps", settings.fps)
                                   / max(settings.fps, 1e-6)))
                while placement.cursor < wanted:
                    got, next_frame = placement.capture.read()
                    if not got:
                        placement.exhausted = True
                        break
                    placement.frame = next_frame
                    placement.cursor += 1
                creative = placement.frame
        if creative is None:
            return base_frame

        region = core.Region(np.float32(quad) * placement.scale,
                             placement.curvature, placement.curved)
        if not core.is_valid_region(region):
            return base_frame

        styled = placement.shape_cache.apply(creative, placement.morph)
        styled = core.apply_brightness_contrast(
            styled, placement.brightness, placement.contrast)
        if placement.colourise:
            styled = core.apply_colourise(styled, base_frame, region.corners)

        grain = 0.0
        if placement.blend.enabled:
            motion = None
            if placement.previous_quad is not None:
                delta = (region.corners - placement.previous_quad).mean(axis=0)
                motion = (float(delta[0]), float(delta[1]))
            styled = blend.photometric_match(styled, base_frame, region.corners,
                                             placement.blend, motion=motion)
            placement.previous_quad = region.corners.copy()
            # Measure the footage's grain BEFORE the creative covers it, or the
            # sample is of the clean creative and no grain gets added at all.
            grain = blend.grain_sigma_for(base_frame, region.corners,
                                          placement.blend)

        base_frame = core.composite_region(base_frame, styled, region,
                                           in_place=True, occlusion=occlusion)
        if grain > 0.1:
            base_frame = blend.add_grain(
                base_frame, grain,
                core.quad_to_mask(region.corners, size[0], size[1]),
                seed=frame_idx)
        return base_frame

    def run(self):
        """The main rendering loop."""
        settings = self.settings
        base_cap = overlay_cap = out_writer = None
        # Bound before the try so the cleanup below can always run. It used to
        # be assigned inside, where a failure opening the base video left the
        # name unbound and the cleanup raised on its way out, hiding whatever
        # had actually gone wrong.
        placements = []
        try:
            base_cap = cv2.VideoCapture(settings.base_video_path)
            if not base_cap.isOpened():
                raise IOError("Could not open the base video for rendering.")

            scale = settings.scale_factor
            width = int(int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale)
            height = int(int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

            # Fill the gaps between tracked frames instead of holding the last
            # known position, which used to freeze the insert mid-shot.
            # Things that went wrong but did not stop the render. These have
            # to reach the user: each one changes what is in the file, and all
            # of them used to be reported as an unqualified success.
            caveats = []

            placements = self._prepare_placements(settings, caveats)
            if not placements:
                # No file gets written here, so this is a failure, not a
                # "Completed" with a note.
                self.finished.emit(
                    "Error: no tracking data covers the chosen frame range.")
                return

            duration = ((settings.end_frame - settings.start_frame + 1)
                        / max(settings.fps, 1e-6))
            out_writer = core.FrameWriter(
                settings.out_path, settings.fps, (width, height),
                audio_source=(settings.base_video_path
                              if settings.include_audio else None),
                audio_start=settings.start_frame / max(settings.fps, 1e-6),
                duration=duration)
            caveats.extend(out_writer.caveats)

            # Seek once, then read straight through. Seeking per frame is slow
            # and, on long-GOP codecs, lands on the nearest keyframe rather
            # than the frame asked for.
            if settings.start_frame > 0:
                base_cap.set(cv2.CAP_PROP_POS_FRAMES, settings.start_frame)

            segmenter = None
            if settings.occlusion:
                try:
                    segmenter = core.PersonSegmenter()
                except (FileNotFoundError, IOError) as exc:
                    logging.warning("Rendering without occlusion: %s", exc)
                    caveats.append("the person model is missing, so people "
                                   "walking in front were painted over")

            frame_counter = 0

            for frame_idx in range(settings.start_frame, settings.end_frame + 1):
                if self.is_canceled:
                    break

                ok, base_frame = base_cap.read()
                if not ok:
                    break
                # Before the resize: the per-row gains are one per row of the
                # source, so scaling first would leave them describing rows
                # that no longer exist.
                if settings.deflicker is not None:
                    base_frame = settings.deflicker.apply(base_frame, frame_idx)
                if scale != 1.0:
                    base_frame = cv2.resize(base_frame, (width, height),
                                            interpolation=cv2.INTER_AREA)

                # Segment once per frame and share the mask: the model is the
                # slow part and it does not depend on which insert is drawn.
                occlusion = None
                if segmenter is not None:
                    try:
                        occlusion = segmenter.mask(base_frame)
                    except cv2.error as exc:
                        logging.warning("Segmentation failed on frame %d: %s",
                                        frame_idx, exc)

                for placement in placements:
                    base_frame = self._draw_placement(
                        placement, base_frame, frame_idx, settings,
                        occlusion, (width, height))

                out_writer.write(base_frame)
                frame_counter += 1
                self.progress.emit(frame_counter)

            base_cap.release()
            base_cap = None

            if self.is_canceled:
                out_writer.abort()
                out_writer = None
                self.finished.emit("Canceled")
                return

            caveats.extend(c for c in out_writer.close()
                           if c not in caveats)
            out_writer = None

            if frame_counter == 0:
                self.finished.emit("Error: no frames were written.")
                return

            self.finished.emit("Completed" +
                               (f" ({'; '.join(caveats)})" if caveats else ""))

        except Exception as e:
            logging.exception("Render worker failed")
            self.finished.emit(f"Error: {e}")
        finally:
            # Every placement carrying a creative video holds a capture of its
            # own. These were released on the way out of a successful render
            # only, so a render that failed part-way -- the case most likely to
            # be retried, and retried again -- leaked one open file per
            # placement each time.
            for placement in placements:
                if getattr(placement, "capture", None) is not None:
                    placement.capture.release()
                    placement.capture = None
            for handle in (base_cap, overlay_cap):
                if handle is not None:
                    handle.release()
            if out_writer is not None:
                out_writer.abort()

    def cancel(self):
        self.is_canceled = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logging.debug("MainWindow initiated")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setMouseTracking(True)

        # No forced aspect ratio: calling resize() from inside resizeEvent
        # fought every attempt to size the window and pinned it to 19:9, an
        # ultrawide shape none of the target laptops have.
        self.setMinimumSize(1024, 640)

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

        # --- Placements -------------------------------------------------
        placements_title = QLabel("Placements")
        placements_title.setStyleSheet(
            "color: white; font-size: 15px; font-weight: bold;")
        self.overlay_layout.addWidget(placements_title)

        hint = QLabel("Each one tracks its own surface and holds its own "
                      "creative. Tick to include it in the render.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8A8A8A; font-size: 10px;")
        self.overlay_layout.addWidget(hint)

        self.placement_list = QListWidget()
        self.placement_list.setFixedHeight(110)
        # Elide a long name rather than growing a scrollbar under a list four
        # rows tall; the tooltip carries the whole thing either way.
        self.placement_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.placement_list.setTextElideMode(Qt.ElideRight)
        self.placement_list.setStyleSheet("""
            QListWidget { background-color: #222; color: #E8E8E8;
                          border: 1px solid #3A3A3A; border-radius: 6px;
                          font-size: 12px; }
            QListWidget::item { padding: 4px; }
            /* An accent rather than a flood. Pure green across the whole row
               shouted louder than anything else in a deliberately quiet
               window, and it is only saying which row is selected. */
            QListWidget::item:selected { background-color: #2E4A6B; color: white;
                                         border-left: 3px solid #6A9FD8; }
        """)
        self.placement_list.currentRowChanged.connect(self.on_placement_selected)
        self.placement_list.itemChanged.connect(self.on_placement_item_changed)
        self.overlay_layout.addWidget(self.placement_list)

        placement_buttons = QHBoxLayout()
        for label, tip, slot in (
                ("+ Add", "Add another placement to track and fill",
                 self.add_placement),
                ("Remove", "Delete the selected placement", self.remove_placement),
                ("Rename", "Rename the selected placement", self.rename_placement)):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.setStyleSheet("""
                QPushButton { background-color: #3C3C3C; color: white;
                              border: none; border-radius: 5px; padding: 4px;
                              font-size: 11px; }
                QPushButton:hover { background-color: #505050; }
            """)
            button.clicked.connect(slot)
            placement_buttons.addWidget(button)
        self.overlay_layout.addLayout(placement_buttons)

        library_title = QLabel("Library")
        # Matching Placements above it: two headings in the same column, one
        # centred and one not, reads as a mistake.
        library_title.setStyleSheet(
            "color: white; font-size: 15px; font-weight: bold;")
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
        self.dirty = False
        # What the last successful render covered, so the AOI export can
        # describe that file rather than the untouched source video.
        self.last_render = None
        self.pending_render = None
        self.last_output_dir = ""
        self.refresh_title()
        self.refresh_placement_list()

    # -- unsaved work ------------------------------------------------------

    # -- placements --------------------------------------------------------

    def refresh_placement_list(self):
        """Rebuild the list, keeping the active placement selected."""
        overlay = self.central_panel.tracking_overlay
        self.placement_list.blockSignals(True)
        self.placement_list.clear()
        for placement in overlay.placements:
            tracked = len(placement.tracking_history)
            mark = "\u25cf" if placement.has_overlay() else "\u25cb"
            # "0f" meant nothing to anyone who had not written it. Kept short
            # all the same: the panel is narrow, and a longer word only got
            # elided away again.
            count = "1 frame" if tracked == 1 else f"{tracked} frames"
            item = QListWidgetItem(f"{mark}  {placement.name}  \u00b7  {count}")
            item.setToolTip(
                f"{placement.name}\n"
                f"{tracked} tracked frames\n"
                + ("Creative loaded" if placement.has_overlay()
                   else "No creative inserted yet"))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if placement.enabled else Qt.Unchecked)
            self.placement_list.addItem(item)
        self.placement_list.setCurrentRow(overlay.active_index)
        self.placement_list.blockSignals(False)

    def on_placement_selected(self, row):
        if row < 0:
            return
        self.central_panel.tracking_overlay.set_active(row)
        self.central_panel.refresh_display()
        self.central_panel.update_tracking_availability()
        self.refresh_library_cards()

    def on_placement_item_changed(self, item):
        row = self.placement_list.row(item)
        overlay = self.central_panel.tracking_overlay
        if 0 <= row < len(overlay.placements):
            overlay.placements[row].enabled = item.checkState() == Qt.Checked
            self.central_panel.refresh_display()
            self.mark_dirty()

    def add_placement(self):
        self.central_panel.tracking_overlay.add_placement()
        self.refresh_placement_list()
        self.central_panel.update_tracking_availability()
        self.central_panel.enable_tracking_mode()
        self.mark_dirty()

    def remove_placement(self):
        overlay = self.central_panel.tracking_overlay
        if len(overlay.placements) <= 1:
            QMessageBox.information(
                self, "Placements",
                "There has to be at least one placement. Use Clear to empty "
                "this one instead.")
            return
        row = self.placement_list.currentRow()
        name = overlay.placements[row].name
        if QMessageBox.question(
                self, "Remove Placement",
                f"Remove {name}, including its tracking?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        overlay.remove_placement(row)
        self.refresh_placement_list()
        self.central_panel.refresh_display()
        self.mark_dirty()

    def rename_placement(self):
        overlay = self.central_panel.tracking_overlay
        row = self.placement_list.currentRow()
        if row < 0:
            return
        name, ok = QInputDialog.getText(self, "Rename Placement", "Name:",
                                        text=overlay.placements[row].name)
        if ok and name.strip():
            overlay.placements[row].name = name.strip()
            self.refresh_placement_list()
            self.mark_dirty()

    def mark_dirty(self):
        """Note that there is tracking work which is not on disk."""
        if not self.dirty:
            self.dirty = True
            self.refresh_title()
        if hasattr(self, "placement_list"):
            self.refresh_placement_list()

    def mark_clean(self):
        self.dirty = False
        self.refresh_title()

    def refresh_title(self):
        name = os.path.basename(getattr(self.central_panel, "current_video_path", "")
                                or "")
        parts = ["Galileo"]
        if name:
            parts.append(name)
        title = "  —  ".join(parts) + (" *" if self.dirty else "")
        self.title_bar.title_label.setText(title)
        self.setWindowTitle(title)

    def confirm_discard(self, action: str) -> bool:
        """Ask before throwing away tracking work. True means carry on.

        Tracking a clip is the expensive part of using this tool, and it was
        previously discarded without a word by loading another video, loading a
        project, or closing the window.
        """
        if not self.dirty:
            return True
        reply = QMessageBox.warning(
            self, "Unsaved Tracking",
            f"You have tracking work that has not been saved.\n\n{action}",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self.save_project()
            return not self.dirty
        return True

    def closeEvent(self, event):
        if self.confirm_discard("Close anyway?"):
            event.accept()
        else:
            event.ignore()

    def _connect_signals(self):
        """Connects signals from child widgets to main window slots."""
        # Left Column connections
        self.left_col.drawClicked.connect(self.on_draw_clicked)
        self.left_col.renderClicked.connect(self.render_video)
        self.left_col.goToFrameClicked.connect(self.on_goto_frame_clicked)
        self.left_col.brightnessClicked.connect(self.on_brightness_clicked)
        self.left_col.contrastClicked.connect(self.on_contrast_clicked)
        self.left_col.colouriseClicked.connect(self.on_colourise_clicked)
        self.left_col.blendClicked.connect(self.open_blend_dialog)
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
            self.mark_dirty()
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

        panel = self.central_panel
        total_frames = int(panel.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keys = sorted(overlay.tracking_history)
        tracked_range = (keys[0], keys[-1])
        video_size = (int(panel.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(panel.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        base = os.path.splitext(os.path.basename(panel.current_video_path or "video"))[0]
        creative = os.path.splitext(os.path.basename(
            overlay.overlay_source_path or "creative"))[0]
        directory = self.last_output_dir or os.path.dirname(
            panel.current_video_path or "")
        suggested = os.path.join(directory, f"{base}_{creative}.mp4")

        dialog = RenderDialog(total_frames, tracked_range, video_size,
                              panel.fps, suggested, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        start_frame, end_frame, scale_factor, out_path = dialog.values()
        if not out_path:
            QMessageBox.warning(self, "Render Error", "Choose where to save the video.")
            return
        if end_frame < start_frame:
            QMessageBox.warning(self, "Render Error",
                                "The end frame is before the start frame.")
            return
        self.last_output_dir = os.path.dirname(out_path)

        settings = self.build_render_settings(start_frame, end_frame,
                                              scale_factor, out_path)
        # Remember what was rendered so the AOI export can describe *this*
        # file rather than the untouched source.
        self.pending_render = {
            "out_path": out_path, "start_frame": start_frame,
            "end_frame": end_frame, "scale_factor": scale_factor,
            "fps": panel.fps,
            "width": int(video_size[0] * scale_factor),
            "height": int(video_size[1] * scale_factor),
        }

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
            occlusion=panel.occlusion_enabled,
            blend_settings=blend.BlendSettings.from_dict(overlay.blend.to_dict()),
            placements=[PlacementSnapshot(p, scale_factor)
                        for p in overlay.placements
                        if p.is_ready()],
            deflicker_gains=panel.deflicker,
        )

    def on_render_finished(self, status):
        """Slot to handle cleanup after rendering finishes, fails, or is canceled."""
        self.progress_dialog.close()
        self.render_thread.quit()
        self.render_thread.wait()

        if status.startswith("Completed"):
            pending = self.pending_render or {}
            out_path = pending.get("out_path")

            # "Completed" used to be claimed even when no file was written.
            if out_path and not os.path.exists(out_path):
                QMessageBox.critical(
                    self, "Render Failed",
                    f"No file was written to:\n{out_path}\n\n"
                    "See the log for details.")
                self.pending_render = None
                return

            self.last_render = pending
            detail = status[len("Completed"):].strip(" ()")
            frames = pending.get("end_frame", 0) - pending.get("start_frame", 0) + 1
            message = (f"Exported {frames} frames to:\n{out_path}\n\n"
                       f"{pending.get('width')}×{pending.get('height')}")
            if detail:
                # A caveat is a warning, not a success notice: the render may
                # have gone out with no audio, no creative, or the ad painted
                # over every pedestrian.
                QMessageBox.warning(
                    self, "Render Complete — with a caveat",
                    f"{message}\n\n⚠  {detail[0].upper()}{detail[1:]}.")
            else:
                QMessageBox.information(self, "Render Complete", message)
            self.pending_render = None
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
            "placements": [p.to_dict() for p in overlay.placements],
            "curvature": overlay.curvature.tolist(),
            "curved": overlay.curved_enabled,
            "brightness": overlay.brightness,
            "contrast": overlay.contrast,
            "colourise": overlay.colourise_enabled,
            "blend": overlay.blend.to_dict(),
            "overlay_path": overlay.overlay_source_path,
            "overlay_is_video": overlay.inserted_overlay_is_video,
            "overlay_start_frame": overlay.inserted_overlay_start_frame,
            # The setting, not the gains. Per-row gains for a long clip run to
            # millions of numbers, which would dwarf the rest of the file; they
            # are cheap enough to measure again from the footage on load.
            "deflicker": (self.central_panel.deflicker.mode
                          if self.central_panel.deflicker is not None else None),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.mark_clean()
        QMessageBox.information(self, "Saved", f"Project saved to:\n{path}")

    def load_project(self):
        if not self.confirm_discard("Loading a project will replace it."):
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "JSON Files (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)

        base_video = data.get("base_video", "")
        if base_video:
            self.central_panel.load_video(base_video)

        overlay = self.central_panel.tracking_overlay
        if data.get("placements"):
            for placement in overlay.placements:
                placement.release()
            overlay.placements = [Placement.from_dict(d) for d in data["placements"]]
            overlay.active_index = 0
            for placement in overlay.placements:
                path_ = placement.overlay_source_path
                if not path_ or not os.path.exists(path_):
                    continue
                if placement.inserted_overlay_is_video:
                    placement.overlay_video_path = path_
                else:
                    try:
                        placement.overlay_bgra = core.load_image_bgra(path_)
                    except IOError:
                        logging.warning("Missing creative for %s: %s",
                                        placement.name, path_)
            self.refresh_placement_list()

        history = {int(k): v for k, v in data.get("tracking_history", {}).items()}
        overlay.tracking_history = {k: [tuple(p) for p in v] for k, v in history.items()}
        self.central_panel.tracking_history = dict(overlay.tracking_history)

        if data.get("curvature") is not None:
            overlay.curvature = np.array(data["curvature"], np.float32).reshape(4, 2, 2)
        overlay.curved_enabled = bool(data.get("curved", False))
        overlay.brightness = data.get("brightness", 0)
        overlay.contrast = data.get("contrast", 1.0)
        overlay.colourise_enabled = bool(data.get("colourise", False))
        overlay.blend = blend.BlendSettings.from_dict(data.get("blend"))

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

        # Only the setting was saved, so the gains are measured again from the
        # footage. Done through the same handler the menu uses, which is what
        # puts a progress dialog on the pass rather than an unexplained pause.
        mode = data.get("deflicker")
        action = getattr(self.title_bar, "deflicker_action", None)
        if mode and base_video and action is not None:
            action.setChecked(True)      # fires toggle_deflicker

        current = self.central_panel.get_current_frame_index()
        points = overlay.tracking_history.get(current) or data.get("tracking_points", [])
        overlay.points = [tuple(p) for p in points]
        self.central_panel.reset_tracker()
        self.central_panel.update_tracking_availability()
        self.central_panel.refresh_display()
        overlay.update()
        self.mark_clean()

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
            self, "Select a picture or clip of the target", "",
            "Images and video (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi "
            "*.mkv *.mov *.m4v);;Images (*.png *.jpg *.jpeg *.bmp *.webp);;"
            "Video (*.mp4 *.avi *.mkv *.mov *.m4v)")
        if not ref_path:
            return

        try:
            # A clip of the target carries several angles and exposures, so it
            # is sampled into a handful of reference views rather than one.
            matcher = core.open_reference(ref_path)
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

        detections = []
        try:
            # Try the frame on screen first; it is what the user is looking at.
            if panel.prev_frame is not None:
                detections = matcher.locate_all(
                    panel.prev_frame,
                    frame_index=panel.get_current_frame_index())
            if not detections and not canceled["value"]:
                found = matcher.scan_video(panel.current_video_path,
                                           step=5, progress=report)
                if found is not None:
                    # Re-examine that frame properly: the scan stops at the
                    # first sighting, but the panel it found may not be alone.
                    capture = cv2.VideoCapture(panel.current_video_path)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, found.frame_index or 0)
                    ok, hit_frame = capture.read()
                    capture.release()
                    detections = (matcher.locate_all(
                        hit_frame, frame_index=found.frame_index)
                        if ok else [found])
        except (IOError, cv2.error) as exc:
            progress.close()
            QMessageBox.warning(self, "Detect Error", str(exc))
            return
        progress.close()

        if canceled["value"]:
            return
        if not detections:
            QMessageBox.information(
                self, "Not Found",
                "Could not find that target in the video.\n\n"
                "This matches flat, detailed things -- a billboard, a screen, a "
                "poster. Try a sharper, straighter-on picture of it, cropped to "
                "just the target.")
            return

        # A reference that was a wide shot rather than a crop of the target
        # matches confidently and returns the corners of the whole scene. Say
        # so, rather than dropping a scene-sized quad on the user.
        reference_frame = panel.prev_frame
        if reference_frame is not None and detections:
            coverage = core.frame_coverage(detections[0], reference_frame.shape)
            if coverage > 0.5:
                if QMessageBox.question(
                        self, "Check the Reference",
                        f"The match covers {coverage:.0%} of the picture, which "
                        "usually means the reference was a wide shot rather "
                        "than a crop of the target.\n\n"
                        "Crop the image, or film the clip framed on the screen "
                        "or billboard itself, and try again.\n\n"
                        "Use this match anyway?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No) != QMessageBox.Yes:
                    return

        chosen = detections
        if len(detections) > 1:
            dialog = MatchChoiceDialog(detections, parent=self)
            if dialog.exec_() != QDialog.Accepted:
                return
            chosen = dialog.selected()
            if not chosen:
                return

        frame_index = chosen[0].frame_index
        if frame_index is not None and frame_index != panel.get_current_frame_index():
            panel.jump_to_frame(frame_index)

        overlay = panel.tracking_overlay
        current = panel.get_current_frame_index()
        created = []
        for index, detection in enumerate(chosen):
            # The first match fills the active placement if it is still blank,
            # so the common single-target case does not leave an empty one
            # behind; the rest each get their own.
            if index == 0 and not overlay.active.points:
                placement = overlay.active
            else:
                placement = overlay.add_placement()
            placement.points = [(float(x), float(y)) for x, y in detection.quad]
            placement.tracking_history[current] = placement.points[:]
            placement.reset_tracker()
            created.append(placement.name)

        overlay.set_active(overlay.placements.index(
            next(p for p in overlay.placements if p.name == created[0])))
        panel.enable_tracking_mode()
        panel.refresh_display()
        overlay.update()
        self.refresh_placement_list()
        self.mark_dirty()

        summary = "\n".join(
            f"  {name}: {d.n_inliers} matching features, {d.confidence:.0%} confidence"
            for name, d in zip(created, chosen))
        QMessageBox.information(
            self, "Target Found",
            f"Found {len(chosen)} on frame {frame_index}:\n\n{summary}\n\n"
            "Give each one a creative, adjust corners if needed, then turn "
            "tracking on.")

    def read_ahead(self, count: int = 14):
        """A short run of frames from where playback is, for inspection.

        Read through a capture of its own so the one driving playback is left
        exactly where the user left it.
        """
        panel = self.central_panel
        path = getattr(panel, "current_video_path", None)
        if not path:
            return []
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return []
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, panel.get_current_frame_index())
            frames = []
            for _ in range(count):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            return frames
        finally:
            capture.release()

    def check_for_a_playing_screen(self):
        """Warn if the area just marked is a screen showing its own picture.

        This is the one mistake that ruins an export without ever looking like
        a mistake: the tracker follows the advert being played instead of the
        panel playing it, and ends hundreds of pixels off with barely a
        complaint. The setting that avoids it has always been there, under
        Options, but only helps someone who already knows to look for it.
        """
        panel = self.central_panel
        overlay = panel.tracking_overlay
        if (len(overlay.points) != 4
                or panel.feature_source != core.PlanarTracker.INTERIOR):
            return                      # nothing marked, or already switched

        frames = self.read_ahead()
        if len(frames) < 4:
            return                      # too near the end of the clip to judge
        quads = [overlay.points] * len(frames)
        try:
            verdict = core.looks_like_a_playing_screen(frames, quads)
        except cv2.error as exc:        # never block marking an area over this
            logging.debug("screen check skipped: %s", exc)
            return

        logging.debug("screen check: %r", verdict)
        if not verdict.independent:
            return

        answer = QMessageBox.question(
            self, "This Looks Like a Playing Screen",
            "The picture inside the area you marked is moving independently "
            "of the panel it is on, by up to "
            f"{verdict.worst_gap:.0f} pixels a frame.\n\n"
            "That is what a digital screen looks like: the advert it is "
            "playing changes, so following the picture follows the advert "
            "rather than the screen, and the insert wanders off the panel "
            "without the tracking reporting anything wrong.\n\n"
            "Track the bezel and wall around it instead?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self.title_bar.digital_screen_action.setChecked(True)

    def toggle_digital_screen(self, enabled: bool):
        """Choose where the tracker looks for the features it follows.

        A printed billboard's own artwork is fixed to it, so tracking the
        inside of the region is ideal. A digital screen's picture is not: it
        changes and slides about independently of the panel, so features found
        on it follow the advert being played rather than the screen playing it,
        and the insert wanders off. For those, track the bezel and wall around
        it instead.
        """
        panel = self.central_panel
        panel.feature_source = (core.PlanarTracker.SURROUND if enabled
                                else core.PlanarTracker.INTERIOR)
        panel.reset_tracker()
        logging.debug("Tracking features from: %s", panel.feature_source)

        if enabled:
            QMessageBox.information(
                self, "Digital Screen Mode",
                "Tracking will follow the bezel and wall around the screen "
                "rather than the picture on it.\n\n"
                "Leave room around the area when you mark it -- that "
                "surrounding detail is what the tracking now depends on.")

    def toggle_occlusion(self, enabled: bool):
        """Let people walking in front of the surface stay in front of it."""
        panel = self.central_panel

        if enabled and not core.PersonSegmenter.is_available():
            QMessageBox.warning(
                self, "Model Not Found",
                "The person-segmentation model is missing.\n\n"
                "Run fetch_model.py to download it (about 6 MB), or place "
                f"{core.PersonSegmenter.MODEL_FILENAME} in a 'models' folder "
                "beside the application.")
            self.title_bar.occlusion_action.setChecked(False)
            return

        panel.occlusion_enabled = enabled
        if not enabled:
            panel.segmenter = None
        panel.refresh_display()

    # -- flickering lighting ------------------------------------------------

    def refresh_deflicker_action(self):
        """Say in the menu what the clip's lighting is actually doing.

        The option is worth nothing if you have to guess whether you need it:
        an office fitting pulsing at a rate the frame rate half-hides is
        exactly the sort of thing that is easier to measure than to see.
        """
        action = getattr(self.title_bar, "deflicker_action", None)
        if action is None:
            return
        report = self.central_panel.flicker_report
        loaded = bool(getattr(self.central_panel, "current_video_path", ""))
        action.setEnabled(loaded)
        if report is None:
            action.setToolTip("Even out pulsing from fluorescent or LED lighting"
                              if loaded else "Open a video first")
        elif report.kind == "steady":
            action.setToolTip("This clip's lighting is already steady "
                              f"({report.whole:.1f}% frame to frame)")
        elif report.kind == "rows":
            action.setToolTip(
                f"Banding found: {report.band_size:.1f}% across the rows, "
                "crawling up the picture. Correcting it works row by row.")
        else:
            beat = f" at about {report.hz:.0f}Hz" if report.hz else ""
            action.setToolTip(
                f"Flicker found: {report.whole:.1f}% frame to frame{beat}.")

    def toggle_deflicker(self, enabled: bool):
        """Even out lighting that pulses, before anything else sees the frame."""
        panel = self.central_panel
        if not enabled:
            panel.set_deflicker(False)
            return

        if not getattr(panel, "current_video_path", ""):
            self.title_bar.deflicker_action.setChecked(False)
            return

        total = panel.total_frames or 0
        dialog = QProgressDialog("Measuring the lighting…", "Cancel", 0,
                                 max(total, 1), self)
        dialog.setWindowTitle("Steady the lighting")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(400)

        def progress(done, count):
            dialog.setMaximum(max(count, done, 1))
            dialog.setValue(done)
            QApplication.processEvents()
            return not dialog.wasCanceled()

        try:
            fixer = panel.set_deflicker(True, progress=progress)
        finally:
            # Read this before closing: closing a QProgressDialog rejects it,
            # which sets wasCanceled, so asking afterwards always says the
            # user cancelled and the "nothing to correct" notice never showed.
            cancelled = dialog.wasCanceled()
            dialog.close()

        if fixer is None:
            # Either nothing to correct or the user cancelled; either way the
            # tick has to come back off rather than claim work that is not
            # being done.
            self.title_bar.deflicker_action.setChecked(False)
            if not cancelled:
                QMessageBox.information(
                    self, "Steady the lighting",
                    "This clip's lighting is already steady, so there is "
                    "nothing to even out.")
            return

        self.mark_dirty()
        kind = ("banding, row by row" if fixer.mode == "rows"
                else "whole-frame flicker")
        logging.info("Lighting steadied (%s) over %d frames.", kind, len(fixer))
        self.title_bar.deflicker_action.setToolTip(
            f"Steadying {kind} across {len(fixer)} frames. "
            "Untick to see the footage as shot.")

    def open_blend_dialog(self):
        """Tune how the creative is matched to the shot, with a live preview."""
        panel = self.central_panel
        if panel.prev_frame is None:
            QMessageBox.warning(self, "Blend", "Load a base video first.")
            return
        overlay = panel.tracking_overlay
        if not overlay.has_overlay() or len(overlay.points) != 4:
            QMessageBox.warning(
                self, "Blend",
                "Mark the area and insert a creative first -- the match is "
                "measured from the pixels the creative covers.")
            return

        dialog = BlendDialog(overlay.blend, panel.refresh_display, parent=self)
        dialog.exec_()
        panel.refresh_display()
        self.mark_dirty()

    def open_morph_dialog(self):
        """Bend and tilt the creative itself, with a live preview.

        The shape belongs to the artwork, not to the tracked area: the quad
        says where the surface is and stays as the tracker measured it, while
        this changes how the creative sits on that surface.
        """
        panel = self.central_panel
        overlay = panel.tracking_overlay
        if overlay.overlay_bgra is None:
            QMessageBox.warning(
                self, "Shape",
                "Insert a creative first -- there is nothing to shape yet.")
            return

        def on_the_footage():
            """The creative as it currently sits on the shot, cropped to it."""
            frame = panel.prev_frame
            if frame is None or len(overlay.points) != 4:
                return None
            composited = panel.composite_placements(frame)
            height, width = composited.shape[:2]
            # A generous margin: how a shape suits a panel is partly a
            # question about what surrounds the panel.
            bounds = core.quad_bounds(overlay.points, width, height, pad=60)
            if bounds is None:
                return None
            x0, y0, x1, y1 = bounds
            return composited[y0:y1, x0:x1]

        dialog = MorphDialog(overlay.morph, overlay.overlay_bgra,
                             panel.refresh_display, parent=self,
                             preview=on_the_footage)
        dialog.exec_()
        panel.refresh_display()
        self.mark_dirty()

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
        if not self.confirm_discard("Loading tracking points will replace it."):
            return
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

    def _write_aoi_file(self, path, placement, fps, video_w, video_h,
                        first_frame, last_frame, scale):
        """Write one placement's AOI geometry.

        Returns ``(rows written, frames skipped)``. This is the only writer of
        the format: the single-placement case used to have a second copy of
        the whole thing, which had already drifted -- it labelled every file
        "aoi1" instead of naming the placement, so which advert a file
        described depended on how many there had been.
        """
        frame_dict = core.interpolate_tracking(placement.tracking_history,
                                               first_frame, last_frame)
        origin = first_frame if first_frame is not None else 0
        rows, skipped = [], 0
        for frame_idx in sorted(frame_dict):
            corners = frame_dict[frame_idx]
            # An unrenderable shape draws no advert, so it must not claim an
            # area of interest either.
            if not core.is_valid_quad(corners):
                skipped += 1
                continue
            scaled = [(float(x) * scale, float(y) * scale) for x, y in corners]
            rows.append([
                f"{(frame_idx - origin) / fps:.3f}",
                f"{(frame_idx - origin + 1) / fps:.3f}",
                "0; 0; 0", "0; 0; 0", "0; 0; 0", "0; 0; 0",
                "None", "None", "None", "None", "0", "None", "None", "None",
                ";".join(f"{v:.4f}" for point in scaled for v in point),
            ])

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=",")
            writer.writerow([f"name:{placement.name}", "csv_type_3",
                             "width:None", "height:None",
                             f"stim_width:{video_w}", f"stim_height:{video_h}"])
            writer.writerow([
                "START", "END", "topleft_coords_XYZ_m", "topright_coords_XYZ_m",
                "bottomleft_coords_XYZ_m", "bottomright_coords_XYZ_m",
                "top_angle_deg", "left_angle_deg", "bottom_angle_deg",
                "right_angle_deg", "average_dist_from_corners",
                "disc_viewable_anagle", "disc_deflection_angle", "disc_diam",
                "points"])
            writer.writerows(rows)
        return len(rows), skipped

    def save_aoi_geometry(self):
        import csv
        if not self.central_panel.cap or not self.central_panel.cap.isOpened():
            QMessageBox.warning(self, "Export Error", "No base video is loaded.")
            return

        # The CSV has to describe the *rendered* stimulus, not the source
        # footage. The render starts at start_frame, may be scaled down, and
        # covers only part of the clip; exporting absolute source times and
        # full-resolution coordinates produced a file that parsed perfectly and
        # was wrong in three independent ways, so every fixation would be
        # mapped to the wrong place with no visible symptom.
        render = self.last_render
        if render:
            fps = render["fps"] or 30
            video_w, video_h = render["width"], render["height"]
            first_frame, last_frame = render["start_frame"], render["end_frame"]
            scale = render["scale_factor"]
            default_name = os.path.splitext(
                os.path.basename(render["out_path"]))[0] + "_aoi.csv"
            default_dir = os.path.dirname(render["out_path"])
        else:
            reply = QMessageBox.question(
                self, "No Render Yet",
                "Nothing has been rendered in this session, so the AOI geometry "
                "can only describe the full-length, full-resolution source "
                "video.\n\nIf the stimulus you show participants is a trimmed or "
                "scaled render, this CSV will not line up with it.\n\n"
                "Export against the source anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            fps = self.central_panel.fps or 30
            video_w = int(self.central_panel.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_h = int(self.central_panel.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first_frame, last_frame, scale = None, None, 1.0
            default_name = "aoi.csv"
            default_dir = self.last_output_dir or ""

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save AOI Geometry", os.path.join(default_dir, default_name),
            "CSV Files (*.csv)")
        if not save_path:
            return

        # One AOI file per placement: an eye-tracking analysis needs to tell
        # the adverts apart, so they cannot share a single area of interest.
        placements = [pl for pl in self.central_panel.tracking_overlay.placements
                      if pl.enabled and pl.tracking_history]
        if len(placements) > 1:
            written = []
            stem, ext = os.path.splitext(save_path)
            for placement in placements:
                safe = "".join(c if c.isalnum() or c in "-_ " else "_"
                               for c in placement.name).strip().replace(" ", "_")
                target = f"{stem}_{safe}{ext or '.csv'}"
                rows, dropped = self._write_aoi_file(
                    target, placement, fps, video_w, video_h,
                    first_frame, last_frame, scale)
                note = f"  ({rows} frames" + (f", {dropped} skipped)" if dropped
                                              else ")")
                written.append(os.path.basename(target) + note)
            QMessageBox.information(
                self, "Export Complete",
                "AOI geometry written, one file per placement:\n\n"
                + "\n".join(written))
            return

        # And one placement through the same writer, rather than a second copy
        # of the format. Keeping two is how they drifted apart: the copy here
        # headed every file "aoi1" instead of naming the placement, so which
        # advert a file described depended on how many there had been.
        only = (placements[0] if placements
                else self.central_panel.tracking_overlay.active)
        written, skipped = self._write_aoi_file(
            save_path, only, fps, video_w, video_h,
            first_frame, last_frame, scale)

        if not written:
            QMessageBox.warning(
                self, "Nothing Exported",
                "No usable tracking data, so an empty AOI file was written.\n\n"
                "Track the area across the frames you intend to show before "
                "exporting.")
            return

        detail = f"{written} frames, {video_w}\u00d7{video_h}"
        if render:
            detail += f"\nMatched to: {os.path.basename(render['out_path'])}"
        else:
            detail += "\n\n\u26a0  Describes the full-resolution source video."
        if skipped:
            detail += f"\n{skipped} frame(s) skipped: the shape was unrenderable."

        QMessageBox.information(self, "Export Complete",
                                f"AOI geometry written to:\n{save_path}\n\n{detail}")
        return

    def upload_base_video(self):
        if not self.confirm_discard("Loading another video will replace it."):
            return
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

        # Shaping belongs to the artwork rather than to the tracked area, so it
        # sits on the creative's own card. It only does anything once this
        # creative is the one in the video, hence the disabled start.
        shape_btn = QPushButton("Shape")
        shape_btn.setObjectName("shapeBtn")
        shape_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A6FA5; color: white; font-size: 12px;
                border: none; border-radius: 5px; padding: 5px;
            }
            QPushButton:hover { background-color: #5C86C4; }
            QPushButton:disabled { background-color: #3A3A3A; color: #6E6E6E; }
        """)
        shape_btn.setFixedSize(55, 25)
        shape_btn.setCursor(QCursor(Qt.PointingHandCursor))
        shape_btn.setEnabled(False)
        shape_btn.setToolTip("Insert this creative to bend or tilt it.")

        # Add the buttons to the horizontal layout.
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(select_btn)
        button_layout.addWidget(shape_btn)
        button_layout.addStretch()

        # Add the entire horizontal button layout to the main vertical layout.
        overlay_layout.addLayout(button_layout)

        # Name the creative, not its file type. Labelling by extension meant an
        # A/B/C test set showed as three identical cards reading "PNG". It gets
        # its own row: three buttons leave a filename no useful width.
        name = os.path.basename(file_path)
        ext_label = QLabel(name)
        ext_label.setStyleSheet("color: white; font-size: 11px;")
        # Its own font decides the height. A round 18 clipped the descenders,
        # so "ad.png" read as "ad.pnq".
        ext_label.setFixedHeight(ext_label.fontMetrics().height() + 2)
        ext_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ext_label.setToolTip(file_path)
        metrics = ext_label.fontMetrics()
        ext_label.setText(metrics.elidedText(name, Qt.ElideMiddle, 190))
        overlay_layout.addWidget(ext_label)
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
            """Take this creative out of whichever placements are holding it."""
            overlay = self.central_panel.tracking_overlay
            source = self.card_source(overlay_widget)
            active = overlay.active
            for placement in overlay.placements:
                if source is not None and placement.overlay_source_path == source:
                    overlay.set_active(overlay.placements.index(placement))
                    overlay.remove_inserted_overlay()
            overlay.set_active(overlay.placements.index(active))
            self.refresh_library_cards()
            if self.central_panel.tracking_mode:
                overlay.setFocus()

        def toggle_insert():
            """Put this creative into the placement being worked on, or take
            it out again -- and only that placement.

            It used to work off one global "the inserted creative", so filling
            a second placement first appeared to un-insert the first: the card
            still read "Inserted" because it was describing a different
            placement, and the click that should have filled the second one
            merely reset the label. It took two clicks, and left the library
            describing whichever placement had been touched last.
            """
            overlay = self.central_panel.tracking_overlay
            if overlay_widget.inserted:
                overlay.remove_inserted_overlay()
            else:
                frame_index = self.central_panel.get_current_frame_index()
                if overlay_widget.inserted_is_video and overlay_widget.video_path:
                    overlay.insert_video_overlay(
                        start_frame_index=frame_index,
                        video_path=overlay_widget.video_path)
                    overlay.update_overlay_video_frame_by_index(0)
                else:
                    # Prefer the file path: reading it directly keeps the
                    # creative's alpha channel intact.
                    overlay.insert_image_overlay(
                        overlay_widget.image_path or overlay_widget.pixmap,
                        frame_index)
                overlay.show()
            self.refresh_library_cards()
            if self.central_panel.tracking_mode:
                overlay.setFocus()

        select_btn.clicked.connect(toggle_insert)
        shape_btn.clicked.connect(self.open_morph_dialog)

    def library_cards(self):
        """Every creative card in the library panel."""
        for index in range(self.overlay_layout.count()):
            item = self.overlay_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None and hasattr(widget, "inserted_is_video"):
                yield widget

    @staticmethod
    def card_source(card):
        """The file a card stands for, whichever kind it is."""
        return card.video_path if card.inserted_is_video else card.image_path

    def refresh_library_cards(self):
        """Show which creative fills the placement being worked on.

        The library describes one placement at a time -- the active one -- so
        switching placements has to redraw all of it. Without that the labels
        described whichever placement was filled last, and said "Insert" over a
        creative that was in the video.
        """
        overlay = self.central_panel.tracking_overlay
        holder = overlay.active.overlay_source_path
        for card in self.library_cards():
            source = self.card_source(card)
            card.inserted = source is not None and source == holder
            select = card.findChild(QPushButton, "selectBtn")
            shape = card.findChild(QPushButton, "shapeBtn")
            if select is not None:
                select.setText("Inserted" if card.inserted else "Insert")
                select.setStyleSheet(
                    "QPushButton { background-color: %s; color: white; "
                    "font-size: 14px; border: none; border-radius: 5px; "
                    "padding: 5px; } QPushButton:hover { background-color: %s; }"
                    % (("limegreen", "green") if card.inserted
                       else ("green", "darkgreen")))
            if shape is not None:
                shape.setEnabled(card.inserted)
                shape.setToolTip(
                    "Bend and tilt this creative before it is tracked into place."
                    if card.inserted else "Insert this creative to bend or tilt it.")
            card.setStyleSheet(
                "background-color: %s; border-radius: 10px; padding: 5px;"
                % ("#00A000" if card.inserted else "#2A2A2A"))
        if self.inserted_overlay_widget is not None:
            self.inserted_overlay_widget = None

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

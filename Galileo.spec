# PyInstaller build for the Galileo Insertion Tool.
#
# Produces a self-contained folder that runs on a machine with no Python and no
# admin rights: the user unzips it and double-clicks the executable.
#
#     pip install -r requirements-build.txt
#     pyinstaller Galileo.spec
#
# Build on the platform you are shipping to -- PyInstaller cannot cross-compile,
# so a Windows .exe has to be built on Windows.
#
# Drop an ffmpeg binary next to the executable in dist/Galileo/ to give renders
# their audio; the app looks there before it looks at PATH.

import os
import sys

# Bundle the person-segmentation model if it has been fetched, so occlusion
# works for whoever receives the build without them downloading anything.
# Run fetch_model.py before building to include it.
DATAS = []
_models = os.path.join(os.path.abspath(SPECPATH), "models")
if os.path.isdir(_models) and any(f.endswith(".onnx") for f in os.listdir(_models)):
    DATAS.append((_models, "models"))
    print("  spec: bundling models/ for occlusion support")
else:
    print("  spec: no models/ found -- occlusion will be unavailable in this "
          "build (run fetch_model.py first to include it)")

# Qt is the main source of bloat and of duplicate-library clashes. WebEngine
# alone is hundreds of megabytes and nothing here uses it.
EXCLUDES = [
    "PyQt5.QtWebEngine", "PyQt5.QtWebEngineWidgets", "PyQt5.QtWebEngineCore",
    "PyQt5.QtWebKit", "PyQt5.QtWebKitWidgets", "PyQt5.QtQml", "PyQt5.QtQuick",
    "PyQt5.QtQuick3D", "PyQt5.QtBluetooth", "PyQt5.QtNfc", "PyQt5.QtDesigner",
    "PyQt5.QtHelp", "PyQt5.QtLocation", "PyQt5.QtPositioning", "PyQt5.QtSql",
    "PyQt5.QtTest", "PyQt5.QtXmlPatterns", "PyQt5.Qt3DCore",
    "PyQt5.QtMultimedia", "PyQt5.QtMultimediaWidgets",
    # Other GUI toolkits, and heavy scientific packages nothing here imports.
    "PySide2", "PySide6", "PyQt6", "tkinter", "matplotlib", "scipy", "pandas",
    "IPython", "jupyter", "notebook", "pytest", "setuptools", "pip",
]

analysis = Analysis(
    ["Galileo_Insertion_Tool_1.0.0.py"],
    pathex=[],
    binaries=[],
    datas=DATAS,
    # Imported normally and picked up automatically; listed so a packaging
    # regression shows up as a loud PyInstaller warning rather than a broken
    # executable discovered by whoever received it.
    hiddenimports=["galileo_core", "galileo_blend", "galileo_morph",
                   "galileo_deflicker"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

# What the hooks collect wholesale but the app never touches. Translations are
# Qt's own UI strings in three dozen languages; the plugin families below back
# Qt modules this app does not ship. Every file dropped here is one less for a
# first-run antivirus scan of the unzipped folder to chew through.
_PRUNE = (
    "Qt5/translations/",
    "Qt5/plugins/mediaservice/",
    "Qt5/plugins/audio/",
    "Qt5/plugins/playlistformats/",
    "Qt5/plugins/sqldrivers/",
    "Qt5/plugins/position/",
    "Qt5/plugins/sensors/",
    "Qt5/plugins/sensorgestures/",
    "Qt5/plugins/texttospeech/",
    "Qt5/plugins/gamepads/",
    "Qt5/plugins/webview/",
    "Qt5/plugins/bearer/",
)


def _kept(entries):
    return [entry for entry in entries
            if not any(fragment in entry[0].replace("\\", "/")
                       for fragment in _PRUNE)]


analysis.binaries = _kept(analysis.binaries)
analysis.datas = _kept(analysis.datas)

pyz = PYZ(analysis.pure)

# Paints within moments of the double-click, while Python is still unpacking
# itself -- on a cold Windows start with an antivirus scanning the folder, the
# gap this covers is several blank seconds. The app closes it once the window
# has actually painted. Not supported by PyInstaller on macOS, and building it
# needs tkinter *in the build environment* (to harvest Tcl/Tk for the
# bootloader; the tkinter exclude above only keeps the module out of the app).
splash = None
if sys.platform != "darwin":
    try:
        import tkinter                      # noqa: F401
    except ImportError:
        print("  spec: tkinter unavailable in this build environment -- "
              "building WITHOUT the startup splash. Use a Tk-enabled Python "
              "(python.org installers have it; on Debian/Ubuntu install "
              "python3-tk) to restore it.")
    else:
        splash = Splash(
            os.path.join(os.path.abspath(SPECPATH), "splash.png"),
            binaries=analysis.binaries,
            datas=analysis.datas,
        )

# Debug symbols are dead weight on the platforms that carry them; stripping is
# a no-op on Windows.
STRIP = sys.platform != "win32"

exe = EXE(
    pyz,
    analysis.scripts,
    *([splash] if splash is not None else []),
    [],
    exclude_binaries=True,
    name="Galileo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=STRIP,
    # UPX compression is what most often gets a bundle flagged by corporate
    # antivirus, and it slows startup. Not worth the saved megabytes here.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

collect = COLLECT(
    exe,
    *([splash.binaries] if splash is not None else []),
    analysis.binaries,
    analysis.datas,
    strip=STRIP,
    upx=False,
    upx_exclude=[],
    name="Galileo",
)

# On macOS, wrap the result in a .app so it behaves like a normal application.
if sys.platform == "darwin":
    app = BUNDLE(
        collect,
        name="Galileo.app",
        icon=None,
        bundle_identifier="com.kabiri.galileo",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": "Not used.",
        },
    )

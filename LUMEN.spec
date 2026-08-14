# PyInstaller build for the LUMEN Insertion Tool.
#
# Produces a self-contained folder that runs on a machine with no Python and no
# admin rights: the user unzips it and double-clicks the executable.
#
#     pip install -r requirements-build.txt
#     pyinstaller LUMEN.spec
#
# Build on the platform you are shipping to -- PyInstaller cannot cross-compile,
# so a Windows .exe has to be built on Windows.
#
# Drop an ffmpeg binary next to the executable in dist/LUMEN/ to give renders
# their audio; the app looks there before it looks at PATH.

import sys

block_cipher = None

# Qt is the main source of bloat and of duplicate-library clashes. WebEngine
# alone is hundreds of megabytes and nothing here uses it.
EXCLUDES = [
    "PyQt5.QtWebEngine", "PyQt5.QtWebEngineWidgets", "PyQt5.QtWebEngineCore",
    "PyQt5.QtWebKit", "PyQt5.QtWebKitWidgets", "PyQt5.QtQml", "PyQt5.QtQuick",
    "PyQt5.QtQuick3D", "PyQt5.QtBluetooth", "PyQt5.QtNfc", "PyQt5.QtDesigner",
    "PyQt5.QtHelp", "PyQt5.QtLocation", "PyQt5.QtPositioning", "PyQt5.QtSql",
    "PyQt5.QtTest", "PyQt5.QtXmlPatterns", "PyQt5.Qt3DCore",
    # Other GUI toolkits, and heavy scientific packages nothing here imports.
    "PySide2", "PySide6", "PyQt6", "tkinter", "matplotlib", "scipy", "pandas",
    "IPython", "jupyter", "notebook", "pytest", "setuptools", "pip",
]

analysis = Analysis(
    ["LUMEN_Insertion_Tool_1.0.0.py"],
    pathex=[],
    binaries=[],
    # lumen_core is imported normally and picked up automatically; listed here
    # so the build fails loudly if it ever goes missing rather than shipping a
    # broken executable.
    datas=[],
    hiddenimports=["lumen_core"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="LUMEN",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
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
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LUMEN",
)

# On macOS, wrap the result in a .app so it behaves like a normal application.
if sys.platform == "darwin":
    app = BUNDLE(
        collect,
        name="LUMEN.app",
        icon=None,
        bundle_identifier="com.kabiri.lumen",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": "Not used.",
        },
    )

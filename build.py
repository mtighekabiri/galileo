#!/usr/bin/env python3
"""Build LUMEN into a standalone folder that needs no Python installed.

    python build.py

Produces dist/LUMEN/ plus a zip beside it, ready to hand to a user who will
unzip it and double-click the executable. See BUILD.md for the details.

Builds for the machine it runs on: PyInstaller cannot cross-compile, so run
this on Windows to get a Windows build.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_ENV = os.path.join(ROOT, ".buildenv")
DIST = os.path.join(ROOT, "dist")
SPEC = os.path.join(ROOT, "LUMEN.spec")


def env_python(env_dir: str) -> str:
    if sys.platform == "win32":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def run(command, **kwargs):
    printable = " ".join(os.path.basename(c) if os.path.sep in c else c
                         for c in command)
    print(f"  $ {printable}")
    subprocess.run(command, check=True, **kwargs)


def make_build_env(reuse: bool) -> str:
    """A dedicated virtual environment for the build.

    Not merely tidiness: if opencv-python and opencv-python-headless are both
    present, PyInstaller bundles both, and the full build's private copy of Qt
    collides with PyQt5's at startup on the user's machine. Building from a
    known-clean environment is what rules that out.
    """
    if os.path.isdir(BUILD_ENV) and not reuse:
        print("Removing the previous build environment...")
        shutil.rmtree(BUILD_ENV)

    if not os.path.isdir(BUILD_ENV):
        print("Creating a clean build environment...")
        venv.EnvBuilder(with_pip=True, clear=True).create(BUILD_ENV)

    python = env_python(BUILD_ENV)
    print("Installing build dependencies...")
    run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "--quiet", "-r",
         os.path.join(ROOT, "requirements-build.txt")])
    return python


def verify_no_duplicate_opencv(python: str) -> None:
    result = subprocess.run(
        [python, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True, check=True)
    installed = result.stdout.lower()
    if "opencv-python==" in installed and "opencv-python-headless==" in installed:
        sys.exit(
            "Both opencv-python and opencv-python-headless are installed in the\n"
            "build environment. Bundling both ships two conflicting copies of Qt\n"
            "and the packaged app will fail to start. Remove opencv-python.")


def check_bundle() -> None:
    internal = os.path.join(DIST, "LUMEN", "_internal")
    stray = os.path.join(internal, "opencv_python.libs")
    if os.path.isdir(stray):
        print("\n  WARNING: the bundle contains opencv_python.libs, which carries a\n"
              "  second copy of Qt and can stop the app starting. Rebuild from a\n"
              "  clean environment (delete .buildenv and re-run).")
    if not os.path.isdir(internal):
        return
    total = sum(os.path.getsize(os.path.join(root, name))
                for root, _, files in os.walk(os.path.join(DIST, "LUMEN"))
                for name in files)
    print(f"  Bundle size: {total / 1e6:.0f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-env", action="store_true",
                        help="keep the existing .buildenv instead of rebuilding it")
    parser.add_argument("--no-zip", action="store_true",
                        help="leave dist/LUMEN/ unzipped")
    args = parser.parse_args()

    if not os.path.exists(SPEC):
        sys.exit(f"Cannot find {SPEC}")

    python = make_build_env(args.reuse_env)
    verify_no_duplicate_opencv(python)

    print("Running PyInstaller...")
    run([python, "-m", "PyInstaller", SPEC, "--noconfirm",
         "--distpath", DIST, "--workpath", os.path.join(ROOT, "build")],
        cwd=ROOT)

    check_bundle()

    if not args.no_zip:
        tag = f"{platform.system().lower()}-{platform.machine().lower()}"
        archive = os.path.join(DIST, f"LUMEN-{tag}")
        print("Zipping...")
        shutil.make_archive(archive, "zip", DIST, "LUMEN")
        print(f"\nDone: {archive}.zip")
    else:
        print(f"\nDone: {os.path.join(DIST, 'LUMEN')}")

    print("\nBefore handing it over, drop an ffmpeg binary next to the executable\n"
          "if you want rendered videos to keep their audio (see BUILD.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

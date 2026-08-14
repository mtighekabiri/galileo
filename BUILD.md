# Packaging Galileo so it runs without Python

The tool ships as a **self-contained folder**. A user unzips it somewhere they
can write — Desktop, Documents, a USB stick, a network share — and
double-clicks `Galileo.exe`. Nothing is installed, nothing touches the registry,
and no administrator is involved. Python, PyQt5, OpenCV and NumPy all travel
inside the folder.

## Before you build

```bash
python fetch_model.py
```

Downloads the person-segmentation model (about 6 MB) into `models/`, which the
spec then bundles so occlusion works for whoever receives the build. Skip it and
everything else still works, but **Draw behind people** will be unavailable to
them with no way to switch it on.

## Build it

Build **on the operating system you are shipping to**. PyInstaller cannot
cross-compile, so a Windows `.exe` must be built on a Windows machine and a
macOS `.app` on a Mac.

```bash
python build.py
```

That creates a clean virtual environment, installs the dependencies into it,
runs PyInstaller, and zips the result to `dist/Galileo-<platform>.zip`.

To drive PyInstaller yourself:

```bash
python -m venv .buildenv
.buildenv/Scripts/activate        # Windows;  source .buildenv/bin/activate elsewhere
pip install -r requirements-build.txt
pyinstaller Galileo.spec
```

The result is `dist/Galileo/`, with `Galileo.exe` at the top.

### Build in a clean environment — this one matters

If both `opencv-python` and `opencv-python-headless` are installed in the
environment you build from, PyInstaller bundles **both**, and the full
`opencv-python` carries its own copy of the Qt libraries. Two Qt builds in one
folder is exactly the conflict that produces *"could not load the Qt platform
plugin xcb/windows"* at startup — on the end user's machine, where it is
hardest to diagnose.

`build.py` uses a fresh virtual environment for this reason. If you build by
hand, do the same, and check afterwards that
`dist/Galileo/_internal/opencv_python.libs` does **not** exist; only
`opencv_python_headless.libs` should be there.

## Hand it over

Zip `dist/Galileo/` and distribute the zip.

Tell users to **unzip before running**. Launching the `.exe` from inside
Windows' built-in zip viewer appears to work, but the supporting libraries are
never extracted and it fails with an opaque error.

Expect roughly **350 MB unzipped, 195 MB zipped**. Most of that is Qt and
OpenCV; there is no meaningful way to shrink it while keeping both. It fits
comfortably on a share or a memory stick.

## Audio in the packaged build

OpenCV cannot write audio, so renders are silent unless `ffmpeg` is available.
The app looks for `ffmpeg` **next to its own executable** before checking
`PATH`, so:

> Download a static `ffmpeg.exe` and drop it into `dist/Galileo/` beside
> `Galileo.exe` before zipping.

Renders then keep their audio, still with nothing installed. Without it,
rendering succeeds and the completion message says the file is silent.

## Occlusion in the packaged build

`models/` is bundled into the application automatically when present at build
time; the app finds it whether it was shipped inside the bundle or dropped into
a `models` folder beside the executable afterwards.

To check what a build picked up, look at the first lines of its log — they
record whether the model and ffmpeg were found.

## Windows warnings on first run

An unsigned executable downloaded or copied from elsewhere will trip
SmartScreen: *"Windows protected your PC"* → **More info** → **Run anyway**.
This is normal for unsigned software and says nothing about the file.

Two ways to avoid putting users through it:

- **Distribute over an internal share** rather than email or the web. Files
  copied from a network location usually carry no mark-of-the-web and are not
  flagged.
- **Code-sign the executable**, which removes the warning properly. This needs
  a certificate from your IT department — the one thing here that involves
  them, and only once, not per user.

The spec deliberately disables UPX compression, which is the single most common
reason corporate antivirus quarantines a PyInstaller bundle.

## Where the log goes

`app_debug.log` is written to a per-user folder, not next to the executable, so
the app still starts from a read-only share:

| Platform | Location |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Galileo\app_debug.log` |
| macOS | `~/Library/Application Support/Galileo/app_debug.log` |
| Linux | `~/.local/share/Galileo/app_debug.log` |

Ask for that file when a user reports a problem.

## Alternatives considered

| Option | Why not |
| --- | --- |
| **MSI / setup.exe installer** | Installing per-machine is the thing that needs an administrator. A zipped folder sidesteps the whole problem. |
| **PyInstaller `--onefile`** | One tidy `.exe`, but it unpacks ~350 MB to a temp folder on *every* launch, so startup is slow, and single-file bundles are flagged by antivirus far more often. |
| **Nuitka** | Compiles to C and can be faster, but the build is fussier and gains little for a GUI app that spends its time inside OpenCV and Qt. |
| **Embedded Python + `.bat`** | Works without admin, but exposes the source tree and is easy for a user to break by moving files. |
| **Web app** | Would remove client install entirely, but means uploading footage to a server and rebuilding the whole interface — a different project, not a packaging change. |

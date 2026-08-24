#!/usr/bin/env python3
"""Download the models used for occlusion.

    python fetch_model.py

Fetches both into ./models/, skipping whichever is already there:

* **PP-HumanSeg** (about 6 MB) segments people, so the creative can be drawn
  *behind* somebody walking in front of the screen -- Options → Draw behind
  people.
* **MiDaS v2.1 small** (about 64 MB) estimates how far away everything in the
  frame is, so anything standing in front of the surface -- railings,
  lampposts, signs, a passing bus -- can be drawn in front of the creative
  too, whatever it is -- Options → Draw behind obstructions.

Both run through OpenCV's own DNN module, so nothing else needs installing.
Neither is committed to the repository, so run this once after cloning, and
before packaging a build if you want occlusion available to the people you
hand it to. Either model works without the other; each enables its own menu
item.

Models:
  PP-HumanSeg, OpenCV Model Zoo (Apache-2.0).
  https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg
  MiDaS v2.1 small, Intel ISL (MIT). Ranftl et al., "Towards Robust Monocular
  Depth Estimation", TPAMI 2020.  https://github.com/isl-org/MiDaS
"""

import hashlib
import os
import sys
import urllib.error
import urllib.request

MODELS = [
    {
        "filename": "human_segmentation_pphumanseg_2023mar.onnx",
        # The Zoo keeps model weights in Git LFS, so the plain github.com/raw
        # URL returns a text pointer rather than the file.
        # media.githubusercontent.com serves the real object.
        "url": ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                "models/human_segmentation_pphumanseg/"
                "human_segmentation_pphumanseg_2023mar.onnx"),
        "expected_bytes": 6163938,
        # Anything under this is a Git LFS pointer or an error page, not a
        # model. Well clear of the real size, and well above either.
        "min_bytes": 1_000_000,
        "size_hint": "about 6 MB",
        "feature": "Draw behind people",
    },
    {
        "filename": "midas_v21_small_256.onnx",
        # Released as model-small.onnx; saved under a name that says which
        # model and which input size it is once it is sitting in models/.
        "url": ("https://github.com/isl-org/MiDaS/releases/download/v2_1/"
                "model-small.onnx"),
        "expected_bytes": 66764249,
        "min_bytes": 30_000_000,
        "size_hint": "about 64 MB",
        "feature": "Draw behind obstructions",
    },
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download(url: str, destination: str, filename: str) -> None:
    print(f"Downloading {filename} ...")

    def report(count, block_size, total):
        if total > 0:
            done = min(100, count * block_size * 100 // total)
            print(f"\r  {done:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print()


def fetch(model: dict) -> bool:
    """Fetch one model if it is not already there. True when it is ready."""
    filename = model["filename"]
    destination = os.path.join(MODELS_DIR, filename)

    if (os.path.exists(destination)
            and os.path.getsize(destination) > model["min_bytes"]):
        print(f"Already present: {destination}")
    else:
        try:
            download(model["url"], destination, filename)
        except (urllib.error.URLError, OSError) as exc:
            print(f"\nDownload failed: {exc}", file=sys.stderr)
            print("\nIf this machine is behind a proxy or offline, fetch the "
                  "file manually from:\n  " + model["url"]
                  + f"\nand save it as:\n  {destination}", file=sys.stderr)
            return False

    size = os.path.getsize(destination)
    if size < model["min_bytes"]:
        print(f"\nThe downloaded file is only {size} bytes, which is too small "
              "to be the model -- it is most likely a Git LFS pointer or an "
              "error page. Delete it and try again.", file=sys.stderr)
        return False
    if size != model["expected_bytes"]:
        print(f"Note: expected {model['expected_bytes']} bytes but got {size}; "
              "the publisher may have released a new revision.")

    digest = hashlib.sha256()
    with open(destination, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    try:
        import cv2
        cv2.dnn.readNetFromONNX(destination)
    except Exception as exc:  # noqa: BLE001 - report whatever went wrong
        print(f"\nThe file downloaded but OpenCV could not load it: {exc}",
              file=sys.stderr)
        return False

    print(f"Ready: {destination}")
    print(f"  {size / 1e6:.1f} MB, sha256 {digest.hexdigest()[:16]}..., "
          "loads correctly")
    return True


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Each model is fetched on its own account: one of them failing -- a dead
    # mirror, a proxy in the way -- must still leave the other usable, since
    # each lights up a menu item by itself.
    ready = [model for model in MODELS if fetch(model)]

    print()
    if ready:
        print("Available in the tool now:")
        for model in ready:
            print(f"  Options -> {model['feature']}")
    failed = [model for model in MODELS if model not in ready]
    for model in failed:
        print(f"Still missing: {model['filename']} ({model['size_hint']}), "
              f"so 'Options -> {model['feature']}' will stay unavailable.",
              file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

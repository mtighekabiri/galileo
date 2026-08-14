#!/usr/bin/env python3
"""Download the person-segmentation model used for occlusion.

    python fetch_model.py

Fetches PP-HumanSeg (about 6 MB) into ./models/. With it in place, the tool can
draw the creative *behind* people who walk in front of the screen instead of
over them.

The model runs through OpenCV's own DNN module, so nothing else needs
installing. It is not committed to the repository, so run this once after
cloning, and before packaging a build if you want occlusion available to the
people you hand it to.

Model: PP-HumanSeg from the OpenCV Model Zoo (Apache-2.0).
https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg
"""

import hashlib
import os
import sys
import urllib.request

MODEL_NAME = "human_segmentation_pphumanseg_2023mar.onnx"

# The Zoo keeps model weights in Git LFS, so the plain github.com/raw URL
# returns a text pointer rather than the file. media.githubusercontent.com
# serves the real object.
URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
       "models/human_segmentation_pphumanseg/" + MODEL_NAME)

EXPECTED_BYTES = 6163938
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download(destination: str) -> None:
    print(f"Downloading {MODEL_NAME} ...")

    def report(count, block_size, total):
        if total > 0:
            done = min(100, count * block_size * 100 // total)
            print(f"\r  {done:3d}%", end="", flush=True)

    urllib.request.urlretrieve(URL, destination, reporthook=report)
    print()


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    destination = os.path.join(MODELS_DIR, MODEL_NAME)

    if os.path.exists(destination) and os.path.getsize(destination) > 1_000_000:
        print(f"Already present: {destination}")
    else:
        try:
            download(destination)
        except (urllib.error.URLError, OSError) as exc:
            print(f"\nDownload failed: {exc}", file=sys.stderr)
            print("\nIf this machine is behind a proxy or offline, fetch the file "
                  "manually from:\n  " + URL + f"\nand save it as:\n  {destination}",
                  file=sys.stderr)
            return 1

    size = os.path.getsize(destination)
    if size < 1_000_000:
        print(f"\nThe downloaded file is only {size} bytes, which is too small to "
              "be the model -- it is most likely a Git LFS pointer or an error "
              "page. Delete it and try again.", file=sys.stderr)
        return 1
    if size != EXPECTED_BYTES:
        print(f"Note: expected {EXPECTED_BYTES} bytes but got {size}; the Zoo may "
              "have published a new revision.")

    digest = hashlib.sha256()
    with open(destination, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    try:
        import cv2
        cv2.dnn.readNetFromONNX(destination)
        loaded = "loads correctly"
    except Exception as exc:  # noqa: BLE001 - report whatever went wrong
        print(f"\nThe file downloaded but OpenCV could not load it: {exc}",
              file=sys.stderr)
        return 1

    print(f"\nReady: {destination}")
    print(f"  {size / 1e6:.1f} MB, sha256 {digest.hexdigest()[:16]}..., {loaded}")
    print("\nOcclusion is now available in the tool (the 'Occlude' toggle).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download a FEW frames from the 12 GB Teledyne FLIR ADAS v2 dataset.

Usage:
    python tools/fetch_flir_samples.py --list            # what is inside? (downloads only the table of contents)
    python tools/fetch_flir_samples.py --get 20          # 20 still frames (16-bit + FLIR's 8-bit)
    python tools/fetch_flir_samples.py --video 60        # 60 consecutive video frames

By downloading you accept the dataset terms: https://www.flir.com/oem/adas/adas-dataset-agree/
The data is NOT ours: it stays in data/ (ignored by git) and is never uploaded.
"""

import argparse
import time
import zipfile
from collections import Counter
from pathlib import PurePosixPath, Path

from thermalpipe.datasets import SplitZipReader

URL = "https://adas-dataset-v2.flirconservator.com/dataset/parts/FLIR_ADAS_v2.zip.{:02d}"
N_PARTS = 12
OUT = Path("data/flir")


def summarize(names):
    """Count files per folder and extension."""
    groups = Counter()
    examples = {}
    for name in names:
        p = PurePosixPath(name)
        if name.endswith("/"):
            continue
        key = (str(p.parent), p.suffix.lower())
        groups[key] += 1
        examples.setdefault(key, p.name)
    for (folder, ext), count in sorted(groups.items()):
        print(f"{count:7d} x {ext or '(none)':6s} in {folder:55s} e.g. {examples[(folder, ext)]}")


def raw_files(names, top):
    """16-bit TIFFs inside <top>/analyticsData, sorted."""
    return sorted(n for n in names if f"{top}/analyticsData/" in n
                  and n.lower().endswith((".tif", ".tiff")))


def matching_jpeg(names_set, tiff_name):
    """FLIR's own 8-bit version of the same frame, if it exists."""
    p = PurePosixPath(tiff_name)
    for ext in (".jpg", ".jpeg"):
        candidate = str(p.parent.parent / "data" / (p.stem + ext))
        if candidate in names_set:
            return candidate
    return None


def extract(z, members, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in members:
        (out_dir / PurePosixPath(name).name).write_bytes(z.read(name))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--get", type=int, default=0, help="number of still frames")
    parser.add_argument("--video", type=int, default=0, help="number of consecutive video frames")
    args = parser.parse_args()

    start = time.time()
    reader = SplitZipReader([URL.format(i) for i in range(N_PARTS)])
    print(f"Dataset size: {reader.size / 1e9:.1f} GB (we will download only a tiny part)")
    with zipfile.ZipFile(reader) as z:
        names = z.namelist()
        names_set = set(names)
        print(f"Table of contents: {len(names)} entries")

        if args.list:
            summarize(names)

        if args.get:
            tiffs = raw_files(names, "images_thermal_val")
            step = max(1, len(tiffs) // args.get)
            chosen = tiffs[::step][:args.get]                  # spread over the whole set
            jpegs = [j for j in (matching_jpeg(names_set, t) for t in chosen) if j]
            extract(z, chosen, OUT / "val" / "raw16")
            extract(z, jpegs, OUT / "val" / "flir8")
            print(f"Saved {len(chosen)} raw 16-bit frames and {len(jpegs)} FLIR 8-bit frames")

        if args.video:
            tiffs = raw_files(names, "video_thermal_test")
            videos = {}
            for t in tiffs:                                     # group frames by video id
                video_id = PurePosixPath(t).name.split("-frame-")[0]
                videos.setdefault(video_id, []).append(t)
            longest = max(videos.values(), key=len)             # the video with most frames
            longest.sort(key=lambda t: int(PurePosixPath(t).name.split("-frame-")[1].split("-")[0]))
            chosen = longest[:args.video]                       # consecutive frames of ONE video
            extract(z, chosen, OUT / "video" / "raw16")
            print(f"Saved {len(chosen)} consecutive frames of one video "
                  f"({len(videos)} videos in total, longest has {len(longest)} frames)")

    print(f"Downloaded {reader.bytes_downloaded / 1e6:.1f} MB in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
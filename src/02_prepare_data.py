"""Build 6-frame radar sequences with the next 5-minute image as target."""
from datetime import datetime, timedelta
from pathlib import Path
import re

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/radar_images"
DEST = ROOT / "data/processed/sequences/radar_sequences.npz"
SIZE = (64, 64)  # modest size for a first CPU experiment
FRAMES = 6
INTERVAL = timedelta(minutes=5)


def timestamp(path):
    # Match a timestamp anywhere in the original or prefixed filename.
    patterns = [(r"(?<!\d)(\d{8})[_-]?(\d{4})(?!\d)", "%Y%m%d%H%M"),
                (r"(\d{4}-\d{2}-\d{2})[_ T](\d{2})[:-](\d{2})", "%Y-%m-%d%H%M")]
    for pattern, fmt in patterns:
        match = re.search(pattern, path.stem)
        if match:
            try:
                return datetime.strptime("".join(match.groups()), fmt)
            except ValueError:
                pass
    return None


def load_image(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("L").resize(SIZE), dtype=np.float32) / 255.0


def main():
    files = sorted((p for p in SOURCE.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}), key=lambda p: p.name) if SOURCE.exists() else []
    dated = sorted(((timestamp(p), p) for p in files if timestamp(p) is not None), key=lambda pair: pair[0])
    print(f"Found {len(files)} images; {len(dated)} have readable timestamps.")
    if len(dated) < FRAMES + 1:
        raise SystemExit("Need at least 7 timestamped radar images. Check filenames and extraction.")

    # Deduplicate timestamps before making sequences.
    by_time = {}
    for time, path in dated:
        by_time.setdefault(time, path)
    times = sorted(by_time)
    sequences, targets, starts = [], [], []
    for start in times:
        needed = [start + i * INTERVAL for i in range(FRAMES + 1)]
        if not all(time in by_time for time in needed):
            continue
        images = [load_image(by_time[time]) for time in needed]
        sequences.append(images[:-1])
        targets.append(images[-1])
        starts.append(start.isoformat())

    if not sequences:
        raise SystemExit("No uninterrupted seven-frame periods found at 5-minute intervals.")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEST, X=np.array(sequences, dtype=np.float32),
                        y=np.array(targets, dtype=np.float32), starts=np.array(starts))
    print(f"Saved {len(sequences)} sequences to {DEST}")


if __name__ == "__main__":
    main()

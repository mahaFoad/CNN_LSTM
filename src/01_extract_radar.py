"""Extract image members of downloaded radar ZIP files."""
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/radar_archives"
DEST = ROOT / "data/processed/radar_images"
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archives = sorted(SOURCE.glob("*.zip"))
    if not archives:
        print(f"No ZIP files found in {SOURCE}")
        return

    # The checksum lets repeated copies of the same image share one output file.
    seen = {hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
    total = 0
    for archive in archives:
        try:
            with ZipFile(archive) as zip_file:
                for member in zip_file.infolist():
                    if member.is_dir() or Path(member.filename).suffix.lower() not in EXTENSIONS:
                        continue
                    content = zip_file.read(member)
                    checksum = hashlib.sha256(content).hexdigest()
                    if checksum in seen:
                        continue
                    # Include archive name to prevent accidental name collisions.
                    filename = f"{archive.stem}_{Path(member.filename).name}"
                    target = DEST / filename
                    if target.exists():
                        print(f"Name collision: {target.name}; skipped")
                        continue
                    target.write_bytes(content)
                    seen.add(checksum)
                    total += 1
        except BadZipFile:
            print(f"Not a valid ZIP: {archive.name}")
    print(f"Extracted {total} new images to {DEST}")


if __name__ == "__main__":
    main()

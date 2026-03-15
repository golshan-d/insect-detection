"""
ingest.py — Team 893C
Simple ingestion tool to copy WAV files into the recordings folder
"""

import argparse
import shutil
from pathlib import Path


def ingest(source, dest):

    source = Path(source)
    dest = Path(dest)

    dest.mkdir(exist_ok=True)

    wav_files = list(source.glob("*.wav"))

    print("Found", len(wav_files), "WAV files")

    copied = 0

    for file in wav_files:

        target = dest / file.name

        if target.exists():
            print("Skipping", file.name)
            continue

        shutil.copy2(file, target)
        print("Copied", file.name)
        copied += 1

    print("\nCopied", copied, "new files")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True,
                        help="Folder containing WAV files")

    parser.add_argument("--dest", default="./recordings",
                        help="Destination recordings folder")

    args = parser.parse_args()

    ingest(args.source, args.dest)


if __name__ == "__main__":
    main()
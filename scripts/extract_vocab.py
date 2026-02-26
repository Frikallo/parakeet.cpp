#!/usr/bin/env python3
"""Extract SentencePiece vocabulary from a NeMo archive.

Usage:
    # Extract vocab to text file:
    python scripts/extract_vocab.py path/to/model.nemo -o vocab.txt

    # Dump vocab entries:
    python scripts/extract_vocab.py path/to/model.nemo --dump
"""

import argparse
import tarfile
import tempfile
from pathlib import Path


def find_vocab_in_archive(nemo_path):
    """Find and extract _tokenizer.vocab from .nemo tar archive."""
    nemo_path = Path(nemo_path)

    # Direct vocab file
    if nemo_path.suffix == ".vocab":
        return nemo_path.read_text()

    # Directory with vocab inside
    if nemo_path.is_dir():
        for pattern in ("**/tokenizer.vocab", "**/_tokenizer.vocab",
                        "**/vocab.txt"):
            matches = list(nemo_path.glob(pattern))
            if matches:
                return matches[0].read_text()
        raise ValueError(f"No vocab file found in {nemo_path}")

    # .nemo tar archive
    if nemo_path.suffix == ".nemo":
        tmpdir = tempfile.mkdtemp()
        with tarfile.open(nemo_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith("tokenizer.vocab"):
                    tar.extract(member, tmpdir, filter="data")
                    return (Path(tmpdir) / member.name).read_text()

    raise ValueError(f"Cannot find vocab in {nemo_path}")


def parse_vocab(text):
    """Parse SentencePiece .vocab format (piece<tab>score per line)."""
    pieces = []
    for line in text.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        pieces.append(parts[0])
    return pieces


def main():
    parser = argparse.ArgumentParser(
        description="Extract vocab from NeMo archive")
    parser.add_argument("input", help="Path to .nemo archive or vocab file")
    parser.add_argument("-o", "--output", default="vocab.txt",
                        help="Output vocab file (default: vocab.txt)")
    parser.add_argument("--dump", action="store_true",
                        help="Print vocab entries instead of saving")
    args = parser.parse_args()

    text = find_vocab_in_archive(args.input)
    pieces = parse_vocab(text)

    print(f"Found {len(pieces)} vocab entries")

    if args.dump:
        for i, piece in enumerate(pieces):
            print(f"  [{i:4d}] model_id={i+1:4d}  {repr(piece)}")
    else:
        # Write plain text file (one piece per line, no scores)
        with open(args.output, "w") as f:
            for piece in pieces:
                f.write(piece + "\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

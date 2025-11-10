"""Prepare Critical Role transcripts."""

import json
from pathlib import Path

import fire


def main(
    data_dir: str,
    output_dir: str,
) -> None:
    """Load cleaned transcripts, and output text format."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in data_dir.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sub_dir = None
        if file_path.stem.startswith("C1"):
            sub_dir = "campaign1"
        elif file_path.stem.startswith("C2"):
            sub_dir = "campaign2"
        (output_dir / sub_dir).mkdir(parents=True, exist_ok=True)
        with open(
            output_dir / sub_dir / f"{file_path.stem}.txt", "w", encoding="utf-8"
        ) as f:
            for turn in data["TURNS"]:
                for name in turn["NAMES"]:
                    f.write(f"{name.capitalize()}: {' '.join(turn['UTTERANCES'])}\n")


if __name__ == "__main__":
    fire.Fire(main)

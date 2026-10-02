"""Inspect existing MIDI artifacts; optionally synthesize local WAV previews."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from musicgen.midi import read_midi, summarize, render_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Requires NumPy; writes simple synthesized previews")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs"/"previews")
    args = parser.parse_args()
    if args.render: args.out.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT/"examples").glob("*.mid")):
        notes = read_midi(f)
        print(f.name, json.dumps(summarize(notes)))
        if args.render:
            render_wav(notes, args.out/f"{f.stem}.wav")
            print("Saved", args.out/f"{f.stem}.wav")


if __name__ == "__main__": main()

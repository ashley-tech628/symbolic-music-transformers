"""Validate that the summary matches the preserved notebook's saved outputs."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from musicgen.history import extract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true",help="Rebuild summary from saved output, without training")
    args=parser.parse_args()
    summary=extract(ROOT)
    target=ROOT/"results/historical/summary.json"
    if args.write:
        target.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    elif json.loads(target.read_text(encoding="utf-8"))!=summary:
        raise ValueError("Summary does not match the saved notebook output")
    print("PASS: saved output and summary match; no model inference was performed.")
    print("Final logged harmonizer token accuracy:",summary["task2_final_logged_epoch"]["token_acc"])
    print("Final logged LM perplexity:",summary["task1_final_logged_epoch"]["test_ppl"])


if __name__=="__main__":main()

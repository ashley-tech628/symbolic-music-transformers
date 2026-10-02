"""Run the notebook-derived research pipeline in a new output directory."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent.parent
REQUIRED={"torch":"torch","numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib","scipy":"scipy","music21":"music21"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["smoke","full"],default="smoke")
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--epochs",type=int,help="Override epochs for both models")
    parser.add_argument("--out",type=Path,default=ROOT/"outputs/new-run")
    parser.add_argument("--check-only",action="store_true",help="Validate source and list missing dependencies; do not train")
    args=parser.parse_args()
    if args.epochs is not None and args.epochs<1:parser.error("--epochs must be positive")
    source=ROOT/"research/experiment.py"
    ast.parse(source.read_text(encoding="utf-8"))
    missing=[m for m in REQUIRED if importlib.util.find_spec(m) is None]
    print("Profile:",args.profile,"Device:",args.device,"Missing dependencies:",missing or "none")
    if args.check_only:return 0
    if missing:
        parser.error("Install requirements.txt before training. No experiment has started.")
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):parser.error("Output directory is not empty; choose a new --out")
    out.mkdir(parents=True,exist_ok=True)
    os.environ.update(MUSICGEN_ROOT=str(ROOT),MUSICGEN_PROFILE=args.profile,MUSICGEN_DEVICE=args.device,MUSICGEN_SEED=str(args.seed),MPLBACKEND="Agg")
    if args.epochs is not None:os.environ["MUSICGEN_EPOCHS"]=str(args.epochs)
    else:os.environ.pop("MUSICGEN_EPOCHS",None)
    versions={m:importlib.metadata.version(dist) for m,dist in REQUIRED.items()}
    metadata={"profile":args.profile,"device_requested":args.device,"seed":args.seed,"epochs_override":args.epochs,
              "python":sys.version,"dependencies":versions,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"status":"running"}
    def save():
        (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    save();previous=Path.cwd();sys.path.insert(0,str(ROOT))
    try:
        os.chdir(out)
        runpy.run_path(str(source),run_name="__main__")
    except BaseException as error:
        metadata["status"]="failed";metadata["error_type"]=type(error).__name__;save();raise
    else:
        metadata["status"]="completed";save()
    finally:
        os.chdir(previous)
    print("Run saved in",out)
    return 0


if __name__=="__main__":raise SystemExit(main())

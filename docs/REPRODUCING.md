# Reproducing and extending the project

Run commands from the repository root with Python 3.10 or newer.

## Inspect existing evidence

```bash
python scripts/audit_history.py
python scripts/preview_samples.py
python -m unittest discover -s tests -v
```

These use the standard library. Architecture tests require PyTorch and otherwise skip explicitly. The audit reads saved notebook outputs without executing cells. It verifies consistency, not the truth or reproducibility of the historical experiment.

## Render listening previews

```bash
python -m pip install -r requirements-preview.txt
python scripts/preview_samples.py --render --out examples/audio
```

Open `examples/listen.html` locally. Simple additive synthesis converts preserved MIDI to WAV; no model checkpoint is required.

## Run a new experiment

Use a dedicated virtual environment and install `requirements.txt`. For GPU execution install a PyTorch build compatible with your machine. The dependency ranges are current packaging choices, not a recovered historical environment.

```bash
python -m pip install -r requirements.txt
python scripts/run_experiment.py --check-only
python scripts/run_experiment.py --profile smoke --device cpu --out outputs/smoke
```

Preflight reports missing modules and parses the experiment source; a successful preflight exit does not mean dependencies are present or training works. Smoke mode reduces the configuration but still trains both models and runs the sequential research pipeline. It may take substantial time on CPU.

```bash
python scripts/run_experiment.py --profile full --device cuda --seed 42 --out outputs/full
```

Use a new empty output directory for each run. `--epochs 1` overrides both training stages. New runs export model state dictionaries with configuration/vocabulary information, split identifiers, figures, summaries, MIDI, and run metadata. These newly added checkpoint exports are not recovered historical weights. A standalone checkpoint-loading inference interface is not included.

Alternatively install a notebook frontend/kernel separately and open `notebooks/music_generation.ipynb` from the repository. Its default configuration is smoke mode. Run cells in order in a fresh kernel. The script is derived from the same cleaned cell sources and is preferable for recording isolated runs.

## Reproduction boundary

Training and inference have not been rerun during packaging. Missing original weights, environment lockfile, prediction logs and cleared execution counters prevent a claim of exact reproduction. Seeded execution is not a guarantee of identical results across devices. Treat the monitored held-out split and baseline comparisons as described in EVALUATION.md. Validate a configured smoke run before investing in a full run.

# Packaging validation

Validated locally on 2026-10-02 using Python 3.12.14.

- Standard-library artifact suite: **9 passed**.
- PyTorch architecture suite: **3 skipped**, because PyTorch was unavailable.
- Saved-output audit: extracted results match the committed summary and source hash.
- MIDI inspection: unconditional sample has 289 notes across 4 note-bearing tracks, 60 seconds, pitches 42–79; conditioned sample has 102 notes across 4 note-bearing tracks, 39 seconds, pitches 45–72.
- Both WAV previews were rendered from the preserved MIDI using simple synthetic tones.
- Python sources and cleaned notebook code cells were syntax checked.

The available environment lacks PyTorch, music21, SciPy and Matplotlib. No new neural training/inference, checkpoint reload, GPU execution, complete pipeline run or hosted GitHub Actions execution was verified. Tests validate selected artifact invariants, not musical quality or historical accuracy claims.

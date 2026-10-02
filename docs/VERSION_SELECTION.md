# Version selection and source evidence

Selected: `assignment2/assignment2.ipynb` in `source-manifest.json`.

The manifest uses logical source labels to avoid publishing personal filesystem paths. `assignment2/` refers to the local course-submission folder; `claude-project/` refers to the separate project-working folder. The exact original notebook bytes are retained in `results/historical/source_notebook.ipynb`, with SHA-256 checked by tests.

| Candidate | Saved evidence | Decision |
|---|---|---|
| `assignment2.ipynb` | 22 code cells contain output; execution counters cleared; 30-epoch large-run logs and displayed tables | Main historical source |
| `FINAL (5).ipynb` | Executed cells and older output; final harmonizer accuracy about 23.3% | Earlier experiment, not mixed into the selected result |
| `fixed.ipynb` | Executed cells and earlier outputs | Alternate earlier experiment |
| `FINAL_v3`, `FINAL_v2`, `FINAL_IMPROVED`, `UPGRADED` | No saved code outputs | Not evidence of the newer metrics on their own |

The selection uses completeness of traceable artifacts, not an assertion that the filename or the highest score proves the best model. `workbook.html` and PDF exports may help presentation, but a notebook with structured saved outputs is more suitable for automated evidence extraction.

The selected notebook records 220 chorales, 176/44 source train/evaluation split, 5 training transpositions, 30 epochs, and an NVIDIA A100-SXM4-40GB. The exact runtime dependency versions, original checkpoint files, and per-token prediction logs were not recovered from these selected folders.

Two MIDI files were copied byte-for-byte from the same submission folder. Their hashes are in the manifest. The notebook later overwrites the unconditional sample after reranking: an earlier plot/table must not automatically be attributed to the final MIDI. The conditioned MIDI includes the provided soprano rather than generating all four voices from scratch.

The cleaned notebook has cleared outputs and added run-artifact export. The historical notebook is separate and unchanged. Model classes were extracted into a shared module without changing their bodies. See `ATTRIBUTION.md` for the packaging boundary.

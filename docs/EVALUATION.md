# Evaluation: what the results do and do not establish

## Evidence level

`scripts/audit_history.py` reads notebook JSON and saved output; it never executes notebook code. It parses epoch dictionaries and HTML result tables, verifies the extracted summary against the preserved evidence, and reports the exact final logged quantities. Displayed DataFrame cells are rounded and retained as strings. This checks transcription/provenance, not model correctness, training reproducibility, or independent scoring from per-token predictions.

The final harmonizer log is **0.3175197669579692** token accuracy; the displayed table rounds it to **0.318**. Lookup is displayed as **0.053**. Earlier notebooks contain different scores; they are not merged.

## Split protocol

The historical experiment splits source chorale entries before augmenting training sequences, avoiding direct transposed copies crossing that split. However, chorale variants/duplicate arrangements were not verified to be grouped. The set called `test` is evaluated repeatedly during training and discussed during development. A future final test should be separated from validation and used only after choices are frozen.

## Baseline comparability

The Transformer maps unseen tokens to `<UNK>` and its harmonization dataset skips short tail windows with fewer than four targets. The lookup baseline compares raw harmony strings over complete held-out sequences. These differences can affect denominators and correctness. The stored 31.8% versus 5.3% is a historical comparison, not a fully controlled sixfold-quality claim. Align targets, masking, windows and unknown-token policy before reporting a stronger conclusion.

Teacher forcing supplies previous true harmony tokens. Autoregressive generation supplies the model's own preceding tokens. Their metrics measure different operating conditions; high teacher-forced token accuracy alone does not establish good generated music.

The unconditional LM and bigram baseline should also be compared with aligned EOS, unknown-token and window conventions. A displayed difference in perplexity is not automatically a statistically meaningful architecture advantage.

## Corrections to the original narrative

- **Pitch-class distribution:** the saved table has Transformer JS 0.071 and unigram JS 0.006. The claim that Transformer is best among all systems on this metric is false. A histogram-matching baseline can still have poor temporal structure.
- **Bigram length:** the shown bigram draw has seven events. That single draw does not establish that a bigram model cannot produce longer music.
- **Memorization:** zero exact 8-gram overlap on a small sample does not prove non-memorization or originality. Short outputs can make this statistic uninformative.
- **OOV:** 20.4% is a fraction of distinct test chord types, not the fraction of test token positions. The implementation maps unseen tokens to `<UNK>`, so that percentage is not a direct hard floor on perplexity.
- **Beat embeddings:** a similarity plot is an observation about learned vectors, not a controlled demonstration that beat embeddings improved musical quality.
- **Sample identity:** later reranking overwrites the unconditioned MIDI. Early sample metrics are not necessarily metrics of the downloadable file.

## Remaining implementation limits

The fixed eight-step beat embedding assumes a 4/4-style periodicity; real chorales may use other meters. Input windows restart position indices, and the implementation does not preserve original bar phase as a separate feature. The soft harmonic preferences favor a simplified pitch-class system and are not a complete harmony-rule checker.

The inherited top-p-style filter drops the first token beyond the cumulative threshold rather than including it, so it differs from conventional nucleus sampling. Its repetition adjustment divides logits by a penalty, which can increase negative repeated logits. These historical heuristics are retained for traceability, not endorsed as canonical implementations; changing them should produce a separately labeled experiment.

Consonance/parallel-motion checks are simplified interval statistics, not expert evaluation. Aggregate musicality results should use multiple samples, equal length budgets, confidence intervals, and listening comparisons. No original learned weights are shipped, so the bundled MIDI previews cannot be regenerated directly without a new training run.

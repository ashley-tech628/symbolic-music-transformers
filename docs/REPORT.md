# Symbolic music generation and harmonization

**Author: Ashley Liu (Xinying Liu).** Independent project; authorship confirmed by the author. This technical synthesis accompanies the portfolio edition and distinguishes historical output from new packaging work.

## Objective

Explore two complementary sequence-modeling tasks on four-part chorales. Unconditional generation learns a distribution over joint SATB chord events. Conditioned harmonization receives the full soprano melody and generates the remaining three voices. This second task is one-to-many: an exact match to a reference harmonization is only one useful measure.

## Data representation

Scores are quantized to an eighth-note grid. Each time step contains soprano, alto, tenor and bass pitches or rests. Keys are normalized toward C major or A minor, and training-only shifts provide augmentation. Joint chord tokens model all four voices; melody tokens and three-voice harmony tokens support the conditioned task. Vocabularies are derived from training sequences.

The selected historical output records 220 source chorales, split into 176 training and 44 evaluation entries, with training augmentation producing 880 sequences. Splitting before augmentation is a useful safeguard, but near-duplicate arrangements still need grouping. The fixed periodic beat representation also simplifies the underlying meter.

## Architectures

The unconditional model is a decoder-only Transformer with causal self-attention, a learned per-head relative-position bias, periodic beat embeddings, pre-normalized residual blocks, and embedding/output weight tying. The implementation is inspired by relative-position modeling; it is not claimed to exactly reproduce a particular published Music Transformer implementation.

The harmonizer uses a bidirectional encoder over the supplied melody and a causal decoder over harmony tokens. Cross-attention lets harmony generation depend on future melody notes, while preventing access to future generated harmony. Both networks use cross-entropy training, AdamW, a cosine learning-rate schedule, and gradient clipping.

## Generation and diagnostics

The research pipeline explores filtered stochastic sampling, temperature, soft voice-leading preferences, best-of-N reranking, and beam search. Music21 writes separate voice parts into MIDI, merging repeated events into sustained notes. Baselines include unigram and bigram generators, a soprano-to-common-harmony lookup, and randomly initialized networks.

Diagnostics include predictive perplexity, teacher-forced token accuracy, repetition, voice crossing, interval-based consonance, pitch-class distribution distance, and n-gram overlap. They must be interpreted together: an excellent pitch histogram can coexist with poor transitions, and a reference mismatch can still sound plausible.

## Historical observations

The final stored harmonizer epoch reports loss 4.063884, perplexity 58.199937 and token accuracy 0.3175198. The displayed lookup accuracy is 0.053. The unconditional final logged perplexity is 485.769806. These quantities can be extracted consistently from the preserved notebook, but no per-token prediction file or original checkpoint was recovered for independent rescoring.

The saved Transformer sample has pitch-class JS 0.071; unigram is lower at 0.006. Thus the original “best among all systems” distribution claim is not supported. The demonstrated bigram sample is short, but a single example cannot establish a general limitation. The companion evaluation document records these and other protocol limits.

## Portfolio engineering work

The portfolio edition extracts shared architecture classes, separates preserved history from a clean rerun notebook, adds a command-line research runner, records new-run configurations/splits/dependencies, and saves weights for future runs. An offline artifact audit and MIDI parser verify evidence consistency and sample timing without invoking training. Lightweight synthetic WAV rendering makes existing MIDI artifacts easy to hear.

The packaging runtime lacks the full ML stack, so no new training or inference result is claimed. Architecture tests are provided but were skipped locally; hosted CI is configured to run them with CPU PyTorch after publication. Nine standard-library artifact tests passed locally.

## Next controlled experiment

Freeze a train/validation/final-test split grouped by chorale identity, align baseline tokenization and masks, save per-token predictions and exact environment versions, and evaluate several seeds. Compare models and sampling priors separately, with equal generation budgets and multiple listening examples. Treat current musical rules and overlap measures as diagnostics rather than guarantees.

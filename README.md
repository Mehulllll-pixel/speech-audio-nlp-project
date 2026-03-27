Speech & Audio NLP Projects
A collection of independent research projects focused on Hindi speech recognition, ASR post-processing, and evaluation frameworks.

Projects
1. Hindi ASR Fine-tuning — Whisper on Conversational Speech
Notebook: hindi_asr_finetuning_whisper.ipynb
Fine-tuned OpenAI's Whisper-small model on ~10 hours of conversational Hindi audio data. Covers the full pipeline: audio preprocessing, segment extraction, feature engineering, training configuration, and evaluation.
Key results:
ModelFLEURS WERDomain WERWhisper-small (baseline)85.03%76.90%Whisper-small (fine-tuned)56.33%46.10%

33.8% relative WER reduction on the FLEURS benchmark
40% reduction on the in-domain held-out set
Best checkpoint selected at step 400 using load_best_model_at_end=True
Includes a bottom-up error taxonomy across 25 sampled utterances (phonetic substitution, word boundary errors, hallucination/repetition, named entity errors, morphological errors)


2. ASR Output Cleanup Pipeline
Notebook: asr_output_cleanup_pipeline.ipynb
A rule-based post-processing pipeline for raw Whisper ASR output in Hindi. Addresses two systematic issues common in conversational Hindi transcriptions.
Components:

Number normalization — converts Hindi number words to digits, including compound numbers (e.g. तीन सौ पच्चीस → 325), with idiom protection to avoid destroying fixed phrases
English word detection — two-tier system tagging both Roman-script English tokens and Devanagari-script English loanwords (e.g. जॉब, कंप्यूटर)


3. Hindi Spelling Classifier (~1,77,000 words)
Notebook: hindi_spelling_classifier.ipynb
A 5-layer classification pipeline to label spelling correctness across a large conversational Hindi vocabulary — without relying on a standard Hindi dictionary (which would miss many valid colloquial forms).
Pipeline layers:

Script pre-filter (punctuation-fused tokens, noise)
Structural impossibility detection (double matra, stutters)
High-frequency word list lookup
Transliterated English loanword detection
Character bigram language model (self-supervised, trained on the dataset itself)

Results:

1,77,508 unique words classified
1,32,024 (74.4%) correctly spelled
16,596 words flagged as low-confidence — recommended for human review
Manual review of 50-word sample reveals 32% false negative rate in the low-confidence bucket


4. Lattice-Based WER Evaluation Framework
Notebook: lattice_wer_evaluation.ipynb
A fairer alternative to standard WER that accounts for valid spelling variants, compound-word splits, filler word optionality, and script-mixing in multilingual speech. Built for evaluating multiple ASR models against a shared reference set.
How it works:

Aligns each model output to the reference using word-level Levenshtein edit distance
Builds a per-utterance lattice where positions that ≥3/6 models agree on are treated as valid alternatives
Penalises only genuinely incorrect outputs, not surface-form variation

Results across 6 models, 45 utterances:
ModelStandard WERLattice WERReductionModel H0.03310.022133.3%Model i0.00610.00610.0% (genuine errors)Model k0.10180.085815.7%Model l0.10660.09619.8%Model m0.19560.172411.9%Model n0.10320.081221.3%
5/6 models were unfairly penalised by rigid reference matching. Model i is correctly unchanged — its errors were genuine, not shared by other models.

Tech Stack

Python, PyTorch, HuggingFace Transformers & Datasets
OpenAI Whisper
librosa, jiwer
Kaggle (notebooks 1–3), Google Colab (notebook 4)


Author
Mehul — AI/ML research projects in speech, NLP, and audio processing.

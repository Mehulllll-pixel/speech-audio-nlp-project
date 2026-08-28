# Speech & Audio NLP Projects

> A collection of independent research projects focused on Hindi speech recognition, ASR post-processing, and NLP evaluation frameworks.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch\&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface\&logoColor=black)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai\&logoColor=white)

---

## Projects

### 1. Hindi ASR Fine-tuning — Whisper on Conversational Speech

**Notebook:** [`hindi_asr_finetuning_whisper.ipynb`](./hindi_asr_finetuning_whisper.ipynb)

Fine-tuned OpenAI's `whisper-small` model on ~10 hours of conversational Hindi audio. Covers the full pipeline — audio preprocessing, segment extraction, feature engineering, training, and evaluation.

#### Results

| Model                      | FLEURS WER | Domain WER |   Improvement   |
| :------------------------- | :--------: | :--------: | :-------------: |
| Whisper-small (baseline)   |   85.03%   |   76.90%   |        —        |
| Whisper-small (fine-tuned) |   56.33%   |   46.10%   | ↓ 33.8% / 40.0% |

#### Highlights

* Best checkpoint selected at step 400 via `load_best_model_at_end=True`
* Systematic error taxonomy across 25 sampled utterances
* Proposed fixes for top 3 error types with code

<details>
<summary>Error taxonomy breakdown</summary>

| Error Type                 | Frequency | Root Cause                             |
| :------------------------- | :-------: | :------------------------------------- |
| Phonetic Substitution      |    40%    | Aspiration/manner confusion (झ→ज, स→श) |
| Word Boundary Error        |    24%    | Polysyllabic word segmentation         |
| Hallucination / Repetition |    16%    | Decoder loop on long inputs            |
| Number / Named Entity      |    12%    | OOV numerals and foreign proper nouns  |
| Morphological Error        |     8%    | Wrong grammatical form of correct root |

</details>

---

### 2. ASR Output Cleanup Pipeline

**Notebook:** [`asr_output_cleanup_pipeline.ipynb`](./asr_output_cleanup_pipeline.ipynb)

A rule-based post-processing pipeline for raw Whisper ASR output in Hindi. Addresses two systematic issues common in conversational Hindi transcriptions.

#### Components

| Module                     | What it does                                                                                                                                |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| **Number Normalization**   | Converts Hindi number words → digits. Handles compound numbers (तीन सौ पच्चीस → `325`). Protects idioms like *दो-चार* from being converted. |
| **English Word Detection** | Two-tier tagger for Roman-script tokens (`[EN]interview[/EN]`) and Devanagari loanwords (`[EN:LOAN]जॉब[/EN:LOAN]`)                          |

---

### 3. Hindi Spelling Classifier (~1,77,000 words)

**Notebook:** [`hindi_spelling_classifier.ipynb`](./hindi_spelling_classifier.ipynb)

A 5-layer pipeline to classify spelling correctness across a large conversational Hindi vocabulary — without relying on a standard dictionary (which misses valid colloquial forms).

#### Pipeline Architecture

```text
Input word
    │
    ▼
Layer 1 → Script pre-filter          (punctuation-fused tokens, noise)
    │
    ▼
Layer 2 → Structural impossibility   (double matra, stutters, ellipsis)
    │
    ▼
Layer 3 → High-frequency word list   (top ~200 Hindi function words)
    │
    ▼
Layer 4 → Transliterated English     (nukta chars, loanword root patterns)
    │
    ▼
Layer 5 → Character bigram LM        (self-supervised, trained on dataset)
    │
    ▼
Output: correct / incorrect + confidence
```

#### Results

| Metric                              |       Value      |
| :---------------------------------- | :--------------: |
| Total words classified              |     1,77,508     |
| Correctly spelled                   | 1,32,024 (74.4%) |
| Incorrectly spelled                 |  45,484 (25.6%)  |
| Low-confidence bucket               |   16,596 (9.4%)  |
| False negative rate (manual review) |        32%       |

> The low-confidence bucket is recommended for **human review** rather than auto-labelling.

---

### 4. Lattice-Based WER Evaluation Framework

**Notebook:** [`lattice_wer_evaluation.ipynb`](./lattice_wer_evaluation.ipynb)

A fairer alternative to standard WER that accounts for valid spelling variants, compound-word splits, filler word optionality, and script-mixing in multilingual speech.

#### Why standard WER is unfair

Standard WER penalises models for producing **valid alternatives** that differ from the reference, such as:

* Spelling variants: `सब्ज़ी` vs `सब्जी`
* Compound splits: `इन सब से` vs `इनसबसे`
* Script mixing: digit `14` vs word `चौदह`
* Optional fillers: `हम्म` being correctly omitted

#### Results — 6 models, 45 utterances, threshold = 3/6

| Model   | Standard WER | Lattice WER |      Reduction     |
| :------ | :----------: | :---------: | :----------------: |
| Model H |    0.0331    |    0.0221   |       ↓ 33.3%      |
| Model i |    0.0061    |    0.0061   | — (genuine errors) |
| Model k |    0.1018    |    0.0858   |       ↓ 15.7%      |
| Model l |    0.1066    |    0.0961   |       ↓ 9.8%       |
| Model m |    0.1956    |    0.1724   |       ↓ 11.9%      |
| Model n |    0.1032    |    0.0812   |       ↓ 21.3%      |

> 5/6 models were unfairly penalised by rigid reference matching. Model i is correctly **unchanged** — its errors were genuine.

---

## Tech Stack

* **Frameworks:** PyTorch, HuggingFace Transformers & Datasets
* **Models:** OpenAI Whisper-small (244M params)
* **Audio:** librosa, ffmpeg
* **Evaluation:** jiwer, custom Levenshtein
* **Environments:** Kaggle Notebooks, Google Colab

---

## Author

**Mehul Garg**

[Portfolio](https://mehul-garg.netlify.app/) · [GitHub](https://github.com/Mehulllll-pixel) · [LinkedIn](https://www.linkedin.com/)

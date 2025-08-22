# Grammatical Error Correction (GEC) — BART & T5 (with PEFT/LoRA)

**Short summary**

This repository implements a Grammatical Error Correction (GEC) system using pretrained sequence-to-sequence models (BART and T5) and parameter-efficient fine-tuning (LoRA / PEFT). It contains code to parse M2 files, build (source, target) training pairs, fine-tune models, run batched inference, and evaluate with **sacrebleu (BLEU)** and **exact-match** metrics.

> The results shown in this README were obtained on a validation split held out from the provided training M2 file. BLEU is reported using `sacrebleu` (corpus BLEU). Exact match is the percentage of predictions that are string-identical to the references after simple normalization (strip).

---

## Table of contents

* [Repository layout](#repository-layout)
* [Quick start](#quick-start)
* [Data preparation (M2)](#data-preparation-m2)
* [Training](#training)
* [Inference / Submission](#inference--submission)
* [Evaluation](#evaluation)
* [Experiments & Results (validation)](#experiments--results-validation)
* [Observations & Analysis](#observations--analysis)
* [Recommendations / Best settings](#recommendations--best-settings)
* [Notes / Implementation details](#notes--implementation-details)
* [Contact / License](#contact--license)

---

## Quick start

1. Create virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# typical packages: torch, transformers, datasets, peft, accelerate, sacrebleu, tqdm, pandas
```

2. Prepare data (convert M2 to train/val pairs using provided parser):

```bash
# Example: run data prep inside the script or use provided parser utilities
python -c "from src.m2_parser import M2Parser; print('Use M2Parser.parse_m2_file and apply_corrections to build pairs')"
```

3. Train a model (examples below in the **Training** section).

---

## Data preparation (M2)

* Use the provided `M2Parser` class to parse the `.m2` file. The parser produces entries of the form:

```json
{"source": "original sentence", "corrections": [ {start_idx, end_idx, error_type, correction}, ... ] }
```

* For each parsed source sentence, apply corrections with `M2Parser.apply_corrections(...)` to build the target (corrected) sentence.

* Shuffle and split into train / validation sets (the experiments reported here used a 90/10 split or similar).

**Note:** the provided parser currently only appends sentences if the M2 entry contains corrections. If you want to include no-op (already-correct) sentences, modify the parser to also keep `noop` examples.

---

## Training

Two example scripts are provided:

* `src/bart_gec.py` — fine-tune BART (supports LoRA/PEFT)
* `src/t5_gec.py` — fine-tune T5 (supports LoRA/PEFT)

**Example command (BART)**:

```bash
python src/bart_gec.py --train --m2_file data/train.m2 --model_path outputs/bart_experiment_1
```

**Example command (T5)**:

```bash
python src/t5_gec.py --train --m2_file data/train.m2 --model_path outputs/t5_experiment_1
```

Both scripts accept CLI flags to:

* `--train` : run training
* `--m2_file` : path to the `.m2` training file
* `--model_path` : directory to save the model / adapter
* `--evaluate` : evaluate via `--source_file` and `--reference_file`
* `--correct` : run inference on a file of input sentences

Training scripts implement:

* tokenization (source and target),
* dynamic padding via a collator,
* mixed precision training (`torch.cuda.amp`) optional,
* LoRA/PEFT wrapper (only update small adapter parameters),
* saving of adapter + tokenizer + minimal metadata for reproducibility.

**Key hyperparameters** are exposed in a `GECConfig` class inside each script: learning rate, batch size, `max_length`, `num_beams`, number of epochs, LoRA params (`r`, `alpha`, `dropout`), and training stabilization options (gradient accumulation, `fp16`, `max_grad_norm`).

---

## Inference / Submission

To run batched corrections (write each corrected sentence on a new line):

```bash
python src/t5_gec.py --correct --input_file data/test.src.txt --output_file corrected.txt --model_path outputs/t5_experiment_1
```

To create a submission CSV from a file with a `source` column:

```bash
python src/t5_gec.py --submission_file data/test_for_kaggle.csv --predictions outputs/test_predictions.csv --model_path outputs/t5_experiment_1
```

---

## Evaluation

The repo supports two metrics:

* **Exact match accuracy** — proportion of predictions exactly equal to the reference (after `.strip()`).
* **BLEU** (corpus BLEU) via `sacrebleu`.

Evaluation example:

```bash
python src/t5_gec.py --evaluate --source_file data/val.src.txt --reference_file data/val.ref.txt --model_path outputs/t5_experiment_1
```

---

## Experiments & Results (validation)

We conducted controlled experiments to study how changing hyperparameters (primarily learning rate and number of epochs) affects model performance. Results below are computed on the held-out validation dataset.

**Table 2 — BLEU Scores and exact match scores on validation dataset**

| Model | Learning Rate | Epochs | Exact match (%) | BLEU Score |
| ----: | :-----------: | :----: | --------------: | ---------: |
|  BART |      5e-5     |    2   |           15.89 |      77.58 |
|  BART |      5e-5     |    3   |           16.57 |      77.85 |
|  BART |      5e-5     |    4   |           17.98 |      77.90 |
|  BART |      5e-5     |    5   |           19.20 |      78.44 |
|    T5 |      5e-5     |    2   |           14.13 |      76.82 |
|    T5 |      5e-5     |    3   |           16.33 |      77.50 |
|    T5 |      5e-5     |    4   |           17.32 |      77.85 |
|    T5 |      5e-4     |    6   |       **21.80** |  **79.40** |
|    T5 |      5e-4     |    7   |           21.10 |      79.40 |

> Table 2: BLEU Scores and exact match scores on validation dataset with different hyperparameters.

---

## Observations & Analysis

1. **Epochs improve performance (up to a point).**
   Increasing epochs for both BART and T5 improved BLEU and exact match (e.g., BART improved from BLEU 77.58 → 78.44 and exact 15.89 → 19.20 when moving from 2 → 5 epochs).

2. **T5 is more sensitive to learning rate.**
   At `lr=5e-5` T5 follows a similar improvement curve as BART. However, when T5 was trained with a larger learning rate (`5e-4`) for more epochs (6 and 7), it showed the **best** validation BLEU (79.40) and the **highest exact match** (21.80 at epoch 6). This suggests T5 can obtain stronger performance with a more aggressive learning rate if training is controlled and stabilized (LoRA, fp16, gradient clipping).

3. **Diminishing / fluctuating returns at higher epochs.**
   The T5 run at `lr=5e-4` had very similar BLEU at epoch 6 and 7 (79.40 both), and a small drop in exact match at epoch 7 (21.10 vs 21.80). This indicates plateauing or mild overfitting; monitor validation metrics and use early stopping.

4. **BART vs T5 (final):**

   * Best BART run (5e-5, 5 epochs): BLEU 78.44, exact 19.20.
   * Best T5 run (5e-4, 6 epochs): BLEU 79.40, exact 21.80.
     In our experiments, **T5 (with higher LR)** slightly outperforms BART on validation BLEU and exact match. Note: higher LR can be less stable and may need gradient clipping and smaller batch sizes.

5. **LoRA helps memory and experimentation speed.**
   Using LoRA/PEFT keeps the number of trainable parameters small and allows more experiments and longer fine-tuning runs on the same hardware.

6. **Metric caveats.**

   * **Exact match** is strict; many valid corrections differ in small tokenization details. Higher BLEU with moderate exact match suggests the model often produces close/correct corrections but not always exact token-for-token matches.
   * **BLEU** rewards n-gram overlap — useful for fluency and similarity but not perfect for GEC semantics. Consider GLEU or the M2 scorer for more targeted GEC evaluation.

---

## Recommendations / Best settings (from these experiments)

* **Best validation performance:** T5 with learning rate `5e-4` and **6 epochs** (BLEU 79.40, exact 21.80). Use *LoRA/PEFT* with LoRA `r=8`, `alpha=16`, `dropout=0.1`.
* **Stable baseline:** BART with `lr=5e-5`, `epochs=4–5` (BLEU \~77.9–78.4).
* **Practical training tips:**

  * Use `fp16` mixed precision if GPU supports it to save memory.
  * Apply gradient clipping (`max_grad_norm≈1.0`) and optionally gradient accumulation to increase effective batch size.
  * Use `num_beams=3–5` for generation; more beams may improve BLEU but increases inference time.
  * Save adapter + tokenizer and a `train_metadata.json` file containing reproducible hyperparameters.

---

## How to reproduce these experiments (example commands)

**Train BART (example):**

```bash
python src/bart_gec.py \
  --train --m2_file data/train.m2 \
  --model_path outputs/bart_lr5e-5_epochs5 \
  --learning_rate 5e-5 --num_epochs 5
```

**Train T5 (best run):**

```bash
python src/t5_gec.py \
  --train --m2_file data/train.m2 \
  --model_path outputs/t5_lr5e-4_epochs6 \
  --learning_rate 5e-4 --num_epochs 6 --use_lora True
```

**Evaluate:**

```bash
python src/t5_gec.py \
  --evaluate --source_file data/val.src.txt --reference_file data/val.ref.txt \
  --model_path outputs/t5_lr5e-4_epochs6
```

**Batch inference (create `corrected.txt`):**

```bash
python src/t5_gec.py \
  --correct --input_file data/test.src.txt --output_file corrected.txt \
  --model_path outputs/t5_lr5e-4_epochs6
```

---

## Notes & implementation details

* **Tokenization & labels:** targets are tokenized and label padding token ids are replaced with `-100` so that the loss ignores padding positions.
* **M2 specifics:** The parser supports corrections that specify token spans (start/end). Corrections are applied in descending index order to avoid index shifts.
* **PEFT / LoRA:** The code wraps the base model using `peft.get_peft_model(...)` and saves adapters with `model.save_pretrained(path)`. Loading recovers the adapter and attaches it to the base model.
* **Metrics:** BLEU computed using `sacrebleu` (corpus BLEU). Exact match is strict string equality after trimming whitespace.

---

## Future work / extensions

* Add **M2 scorer** or GLEU for more GEC-specific evaluation.
* Evaluate on held-out test sets (Kaggle submission) and build a small error taxonomy (spelling, article, preposition, verb form).
* Compare LoRA vs full fine-tuning vs Adapters (quantify speed / memory tradeoffs).
* Examine model outputs and compute precision/recall for error detection + correction.

---

## Contact / license

* Author / entry: **2024AIZ8309** — replace with your name/ID when publishing.
* License: choose a license for your repo (e.g., MIT). Add `LICENSE` file if desired.

---

If you want, I can also:

* produce a `requirements.txt` pinned to known-compatible versions,
* create a `train.sh` example to run hyperparameter sweeps, or
* generate a small Jupyter notebook to visualize errors and per-type breakdowns.

#!/usr/bin/env python3
# bart_gec.py
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
from tqdm import tqdm
from datasets import Dataset
from sacrebleu.metrics import BLEU

# PEFT / LoRA imports
try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
    # Provide placeholders to avoid NameErrors later
    get_peft_model = None
    LoraConfig = None
    TaskType = None
    PeftModel = None
    PeftConfig = None

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW


@dataclass
class GECConfig:
    output_dir: str = "./gec_bart_outputs"
    cache_dir: str = "./cache"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # training hyperparams
    train_batch_size: int = 8
    eval_batch_size: int = 8
    batch_correct_batch_size: int = 8
    max_length: int = 128
    num_beams: int = 5
    learning_rate: float = 5e-4
    num_epochs: int = 5

    # LoRA/PEFT defaults
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # other
    seed: int = 42
    val_ratio: float = 0.1


class M2Parser:
    """Parser for M2 formatted GEC data."""

    @staticmethod
    def parse_m2_file(filename: str) -> List[Dict]:
        data = []
        current_sentence = {}
        source_sentence = None
        corrections = []

        with open(filename, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    continue
                line = line.strip()

                if line.startswith('S '):
                    if source_sentence is not None and corrections:
                        current_sentence = {'source': source_sentence, 'corrections': corrections}
                        data.append(current_sentence)
                    source_sentence = line[2:]
                    corrections = []

                elif line.startswith('A '):
                    if "noop" in line:
                        continue
                    parts = line[2:].split("|||")
                    if len(parts) >= 3:
                        try:
                            start_idx = int(parts[0].split()[0])
                            end_idx = int(parts[0].split()[1])
                        except Exception:
                            # malformed index: skip correction
                            continue
                        error_type = parts[1]
                        correction = parts[2]
                        corrections.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'error_type': error_type,
                            'correction': correction
                        })

        if source_sentence is not None and corrections:
            current_sentence = {'source': source_sentence, 'corrections': corrections}
            data.append(current_sentence)
        return data

    @staticmethod
    def apply_corrections(source: str, corrections: List[Dict]) -> str:
        tokens = source.split()
        sorted_corrections = sorted(corrections, key=lambda x: (x['start_idx'], x['end_idx']), reverse=True)
        for correction in sorted_corrections:
            start_idx = correction['start_idx']
            end_idx = correction['end_idx']
            corrected_text = correction['correction']
            if start_idx < len(tokens):
                end_idx = min(end_idx, len(tokens))
                del tokens[start_idx:end_idx]
                if corrected_text.strip():
                    corrected_tokens = corrected_text.split()
                    for i, tok in enumerate(corrected_tokens):
                        tokens.insert(start_idx + i, tok)
        return ' '.join(tokens)


class GECorrector:
    """BART-based GEC with optional LoRA + manual DataLoader training + per-epoch generation eval."""

    def __init__(self, config: GECConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model_name = "facebook/bart-large"

        # tokenizer & model placeholders
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.config.cache_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            # ensure a pad token exists
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=self.config.cache_dir)

        # LoRA / PEFT wrap if requested and available
        if getattr(config, "use_lora", False) and PEFT_AVAILABLE:
            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj"],  # common for bart attention projections
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.model = get_peft_model(base_model, lora_cfg)
            # print trainable count if available
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass
        else:
            self.model = base_model

        self.model.to(self.device)

        # keep last parsed rows as fallback if needed
        self._last_rows = None
        self._last_split_info = None

        # deterministic
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def load_and_prepare_data(self, m2_file: str) -> Tuple[Dataset, Dataset]:
        parsed = M2Parser.parse_m2_file(m2_file)
        rows = []
        for inst in parsed:
            src = inst.get('source', '').strip()
            tgt = M2Parser.apply_corrections(src, inst.get('corrections', []))
            rows.append({"source": src, "target": tgt})

        if len(rows) == 0:
            raise ValueError("No sentence pairs parsed from M2 file.")

        # deterministic shuffle & split
        rnd = random.Random(self.config.seed)
        rnd.shuffle(rows)
        n = len(rows)
        n_val = max(1, int(n * self.config.val_ratio))
        n_train = n - n_val
        train_rows = rows[:n_train]
        val_rows = rows[n_train:]

        # store for fallback
        self._last_rows = rows
        self._last_split_info = (n_train, n_val)

        # prepare tokenized tensors (return HF Dataset with torch tensors)
        def preprocess(data_rows):
            inputs = [r['source'] for r in data_rows]
            targets = [r['target'] for r in data_rows]
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # BART uses same tokenizer for labels
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels
            # convert to dict of lists for Dataset.from_dict
            dict_out = {k: v.tolist() for k, v in model_inputs.items()}
            return dict_out

        train_tokenized = preprocess(train_rows)
        val_tokenized = preprocess(val_rows)

        train_ds = Dataset.from_dict(train_tokenized).with_format("torch")
        val_ds = Dataset.from_dict(val_tokenized).with_format("torch")
        print(f"Prepared train ({len(train_ds)}) and val ({len(val_ds)}) datasets (n_train={n_train}, n_val={n_val})")
        return train_ds, val_ds

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Manual DataLoader training + generation eval per epoch (BLEU & exact match)."""
        print("Starting training")
        # fallback: try to extract raw val texts (if caller passed tokenized datasets)
        val_sources, val_targets = self._extract_val_texts(val_dataset)
        if len(val_sources) == 0:
            raise ValueError("Validation sources empty. Make sure val_dataset contains 'source' and 'target' or call load_and_prepare_data.")

        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.eval_batch_size)

        # optimizer on trainable params only (keeps LoRA adapters trainable and base weights frozen if applicable)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.learning_rate)

        best_bleu = -1.0
        best_dir = os.path.join(self.config.output_dir, "best_checkpoint")
        os.makedirs(self.config.output_dir, exist_ok=True)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{self.config.num_epochs}")
            for batch in pbar:
                # batch items are lists/tensors depending on Dataset format -> ensure tensors on device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": f"{(total_loss / (pbar.n + 1)):.4f}"})

            avg_loss = total_loss / max(1, len(train_loader))
            print(f"Epoch {epoch+1} finished. Avg train loss: {avg_loss:.4f}")

            # generation-based evaluation on val_sources
            print("Generating on validation set...")
            preds = self.batch_correct(val_sources)

            try:
                bleu_metric = BLEU()
                bleu = float(bleu_metric.corpus_score(preds, [val_targets]).score)
            except Exception:
                bleu = 0.0
            exact = float(np.mean([p.strip() == t.strip() for p, t in zip(preds, val_targets)]))
            print(f"Epoch {epoch+1} eval -> BLEU: {bleu:.4f}, exact_match: {exact:.4f}")

            # Save epoch checkpoint
            epoch_ckpt = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch+1}")
            print(f"Saving epoch checkpoint to {epoch_ckpt}")
            self._save_model_and_tokenizer(epoch_ckpt)

            # Save best
            if bleu > best_bleu:
                print(f"New best BLEU {bleu:.4f} (prev {best_bleu:.4f}) -> saving best checkpoint to {best_dir}")
                best_bleu = bleu
                self._save_model_and_tokenizer(best_dir)

        print(f"Training complete. Best BLEU: {best_bleu:.4f}. Best model at: {best_dir}")

    def _extract_val_texts(self, val_dataset: Dataset) -> Tuple[List[str], List[str]]:
        # If dataset still contains original 'source' and 'target' columns, use them
        if hasattr(val_dataset, "column_names") and 'source' in val_dataset.column_names and 'target' in val_dataset.column_names:
            val_sources = [s for s in val_dataset["source"]]
            val_targets = [t for t in val_dataset["target"]]
            print(f"Using validation texts directly from Dataset (n={len(val_sources)})")
            return val_sources, val_targets

        # fallback to self._last_rows using split info
        if self._last_rows is not None and self._last_split_info is not None:
            n_train, n_val = self._last_split_info
            rows = self._last_rows
            if len(rows) >= n_train + n_val:
                val_rows = rows[n_train:n_train + n_val]
                val_sources = [r["source"] for r in val_rows]
                val_targets = [r["target"] for r in val_rows]
                print(f"Falling back to last parsed rows for validation texts (n={len(val_sources)})")
                return val_sources, val_targets

        return [], []

    def batch_correct(self, sentences: List[str]) -> List[str]:
        """Generate corrected sentences using model.generate()."""
        self.model.eval()
        results: List[str] = []
        batch_size = self.config.batch_correct_batch_size
        tokenizer = self.tokenizer
        for i in tqdm(range(0, len(sentences), batch_size), desc="Batch generate"):
            batch = sentences[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=self.config.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask", None),
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            results.extend([s.strip() for s in decoded])
        return results

    def evaluate(self, source_file: str, reference_file: str) -> Dict[str, float]:
        with open(source_file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f if line.strip()]
        with open(reference_file, 'r', encoding='utf-8') as f:
            refs = [line.strip() for line in f if line.strip()]

        predictions = []
        for i in tqdm(range(0, len(sources), self.config.eval_batch_size), desc="Eval generate"):
            batch = sources[i:i + self.config.eval_batch_size]
            predictions.extend(self.batch_correct(batch))

        exact_matches = sum([p == r for p, r in zip(predictions, refs)])
        exact_match_accuracy = exact_matches / len(refs) if len(refs) > 0 else 0.0
        try:
            bleu_metric = BLEU()
            bleu_score = float(bleu_metric.corpus_score(predictions, [refs]).score)
        except Exception:
            bleu_score = 0.0
        return {"Exact Match Accuracy": exact_match_accuracy, "Bleu Score": bleu_score}

    def make_pred(self, source_csv: str, output_file: str):
        import pandas as pd
        df = pd.read_csv(source_csv)
        if 'source' not in df.columns:
            raise ValueError("Input CSV must contain a 'source' column")
        sources = df['source'].tolist()
        predictions = []
        for i in tqdm(range(0, len(sources), self.config.eval_batch_size), desc="Generating submission"):
            batch = sources[i:i + self.config.eval_batch_size]
            predictions.extend(self.batch_correct(batch))
        df['prediction'] = predictions
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    def _save_model_and_tokenizer(self, path: str):
        os.makedirs(path, exist_ok=True)
        # If model is a PeftModel -> save_pretrained will store adapters + base model info
        try:
            if PEFT_AVAILABLE and isinstance(self.model, PeftModel):
                self.model.save_pretrained(path)
            else:
                # regular HF save
                self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            print(f"Warning: saving model/tokenizer failed: {e}")

    @classmethod
    def load(cls, path: str, config: Optional[GECConfig] = None):
        if config is None:
            config = GECConfig()
        instance = cls(config)

        # load tokenizer from path (fallback to model_name)
        try:
            instance.tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=config.cache_dir, use_fast=True)
        except Exception:
            instance.tokenizer = AutoTokenizer.from_pretrained(instance.model_name, cache_dir=config.cache_dir, use_fast=True)

        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token

        # if PEFT saved adapter exists, load with PeftModel
        try:
            if PEFT_AVAILABLE:
                # PeftConfig will provide base model path
                peft_conf = PeftConfig.from_pretrained(path)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_conf.base_model_name_or_path, cache_dir=config.cache_dir)
                instance.model = PeftModel.from_pretrained(base_model, path)
                instance.model.to(config.device)
                try:
                    instance.model.print_trainable_parameters()
                except Exception:
                    pass
                print(f"Loaded LoRA-adapted BART model from {path}")
                return instance
        except Exception:
            # if PEFT load fails, try to load standard model
            pass

        # fallback: load normal HF weights from path
        try:
            instance.model = AutoModelForSeq2SeqLM.from_pretrained(path, cache_dir=config.cache_dir)
            instance.model.to(config.device)
            print(f"Loaded full BART model from {path}")
            return instance
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEC using BART + optional LoRA (PEFT)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--m2_file", type=str, help="Path to M2 file for training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--source_file", type=str, help="Path to source sentences")
    parser.add_argument("--reference_file", type=str, help="Path to reference corrections")
    parser.add_argument("--correct", action="store_true", help="Correct sentences (input lines -> generation)")
    parser.add_argument("--input_file", type=str, help="Path to input sentences")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument("--model_path", type=str, default="./gec_bart_outputs", help="Path to save/load model")
    parser.add_argument("--test_m2_file", type=str, help="Path to M2 file for evaluation")
    parser.add_argument("--submission_file", type=str, help="path to input csv for predictions (must contain 'source' col)")
    parser.add_argument("--predictions", type=str, help="path to save submission csv (with 'prediction' col)")
    args = parser.parse_args()

    cfg = GECConfig()
    cfg.output_dir = args.model_path

    if args.train and args.m2_file:
        corrector = GECorrector(cfg)
        train_ds, val_ds = corrector.load_and_prepare_data(args.m2_file)
        corrector.train(train_ds, val_ds)
        corrector._save_model_and_tokenizer(args.model_path)
    else:
        # load model if not training
        corrector = GECorrector.load(args.model_path, cfg)

    if args.evaluate and args.source_file and args.reference_file:
        results = corrector.evaluate(args.source_file, args.reference_file)
        print("Evaluation results:")
        print(results)

    if args.submission_file and args.predictions:
        corrector.make_pred(args.submission_file, args.predictions)

    if args.correct and args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        corrected_sentences = []
        for i in tqdm(range(0, len(sentences), cfg.batch_correct_batch_size), desc="Generating predictions"):
            batch = sentences[i:i + cfg.batch_correct_batch_size]
            corrected_sentences.extend(corrector.batch_correct(batch))
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as out_f:
                for s in corrected_sentences:
                    out_f.write(s + "\n")
            print(f"Corrected sentences saved to {args.output_file}")
        else:
            for original, corrected in zip(sentences, corrected_sentences):
                print(f"Original: {original}")
                print("Corrected: ")
                print(corrected)
                print("-" * 50)

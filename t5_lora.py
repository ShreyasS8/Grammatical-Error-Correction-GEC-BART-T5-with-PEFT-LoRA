##### Imports #####
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import transformers
from tqdm import tqdm 
from datasets import Dataset
import random
import pandas as pd
from sacrebleu.metrics import BLEU
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration

#############  config class for hyperparameters #############

@dataclass
class GECConfig:
    """Configuration for the GEC model."""
    output_dir: str = "./gec_model_outputs"
    cache_dir: str = "./cache"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    entry_number: str = "2024AIZ8309"

    # Training Hyperparameters
    train_batch_size: int = 8
    eval_batch_size: int = 8
    max_length: int = 128
    num_beams: int = 5
    learning_rate: float = 5e-4
    num_epochs: int = 5
    seed: int = 42

class M2Parser:
    """Parser for M2 formatted GEC data."""

    @staticmethod
    def parse_m2_file(filename: str) -> List[Dict]:
        """
        Parse an M2 file into a list of sentence dictionaries.

        Returns a list of dicts: {'source': ..., 'corrections': [...]}
        """
        data = []
        current_sentence = {}
        source_sentence = None
        corrections = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line.startswith('S '):
                    # if previous sentence had corrections, add
                    if source_sentence is not None and corrections:
                        current_sentence = {
                            'source': source_sentence,
                            'corrections': corrections
                        }
                        data.append(current_sentence)
                    source_sentence = line[2:]
                    corrections = []

                elif line.startswith('A '):
                    if "noop" in line:
                        continue
                    parts = line[2:].split("|||")
                    if len(parts) >= 3:
                        span = parts[0].strip()
                        span_parts = span.split()
                        if len(span_parts) >= 2:
                            try:
                                start_idx = int(span_parts[0])
                                end_idx = int(span_parts[1])
                            except ValueError:
                                start_idx = 0
                                end_idx = 0
                        else:
                            start_idx = 0
                            end_idx = 0
                        error_type = parts[1].strip()
                        correction = parts[2].strip()
                        corrections.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'error_type': error_type,
                            'correction': correction
                        })

        if source_sentence is not None and corrections:
            current_sentence = {
                'source': source_sentence,
                'corrections': corrections
            }
            data.append(current_sentence)

        return data

    @staticmethod
    def apply_corrections(source: str, corrections: List[Dict]) -> str:
        """
        Apply corrections to a source sentence (basic token index replacement).
        """
        tokens = source.split()
        # sort descending to apply replacements without shifting earlier indices
        sorted_corrections = sorted(corrections, key=lambda x: (x['start_idx'], x['end_idx']), reverse=True)

        for correction in sorted_corrections:
            start_idx = correction['start_idx']
            end_idx = correction['end_idx']
            corrected_text = correction['correction']

            if start_idx < len(tokens):
                # clamp end_idx
                end_idx_clamped = min(end_idx, len(tokens))
                del tokens[start_idx:end_idx_clamped]

                if corrected_text.strip():
                    corrected_tokens = corrected_text.split()
                    for i, token in enumerate(corrected_tokens):
                        tokens.insert(start_idx + i, token)

        corrected_sentence = ' '.join(tokens)
        return corrected_sentence


class GECorrector:
    """GEC system using the T5 model with LoRA."""

    def __init__(self, config: GECConfig):
        """
        Initialize the GEC system.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)

        # Seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        # Load base T5 model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        # Wrap the model with LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

    def load_and_prepare_data(self, m2_file: str, split_ratio: float = 0.9, shuffle: bool = True):
        """
        Load M2 file, apply corrections to create target texts, and produce tokenized train/val datasets.

        Returns:
            train_dataset (HuggingFace Dataset), val_dataset, train_text_pairs (list of tuples), val_text_pairs
        """
        print('Loading and preparing M2 file...')

        parsed_data = M2Parser.parse_m2_file(m2_file)

        data = []
        for item in parsed_data:
            source = item['source']
            target = M2Parser.apply_corrections(source, item['corrections'])
            data.append({'source': source, 'target': target})

        if shuffle:
            random.shuffle(data)

        split_index = int(split_ratio * len(data))
        train_pairs = data[:split_index]
        val_pairs = data[split_index:]

        # Tokenize and return datasets and raw text lists
        def preprocess_pairs(pairs):
            inputs = [x['source'] for x in pairs]
            targets = [x['target'] for x in pairs]
            # Return lists (not tensors) so we can construct Dataset.from_dict directly
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.config.max_length,
                    padding='max_length',
                    truncation=True
                )['input_ids']

            pad_id = self.tokenizer.pad_token_id
            # Replace pad token id with -100 for loss calculation
            labels = [
                [(tok if tok != pad_id else -100) for tok in label]
                for label in labels
            ]

            out = {
                'input_ids': model_inputs['input_ids'],
                'attention_mask': model_inputs['attention_mask'],
                'labels': labels
            }
            return out, inputs, targets

        tokenized_train, train_inputs, train_targets = preprocess_pairs(train_pairs)
        tokenized_val, val_inputs, val_targets = preprocess_pairs(val_pairs)

        train_dataset = Dataset.from_dict(tokenized_train).with_format('torch')
        val_dataset = Dataset.from_dict(tokenized_val).with_format('torch')

        print(f'Prepared train ({len(train_dataset)}) and val ({len(val_dataset)}) datasets.')
        return train_dataset, val_dataset, (train_inputs, train_targets), (val_inputs, val_targets)

    def train(self, train_dataset: Dataset, val_dataset: Dataset,
              train_texts: Tuple[List[str], List[str]],
              val_texts: Tuple[List[str], List[str]]):
        """
        Train the model with LoRA-enabled T5 and evaluate after each epoch.
        """
        print("Starting LoRA training...")

        batch_size = self.config.train_batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.eval_batch_size)

        # Only LoRA params will be optimized (parameters with requires_grad=True)
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.learning_rate)

        num_epochs = self.config.num_epochs
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                epoch_loss += loss.item()
                num_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': f'{(epoch_loss/num_batches):.4f}'})

            avg_train_loss = epoch_loss / max(1, num_batches)
            print(f"\nEpoch {epoch+1} training loss: {avg_train_loss:.4f}")

            # Run validation & print metrics
            val_inputs, val_targets = val_texts
            val_metrics = self.evaluate_texts(val_inputs, val_targets)
            print(f"Validation after epoch {epoch+1}: Exact Match Accuracy: {val_metrics['Exact Match Accuracy']:.4f}, BLEU: {val_metrics['Bleu Score']:.4f}")

        print("LoRA training completed.")

    def batch_correct(self, sentences: List[str]) -> List[str]:
        """
        Correct grammatical errors in a batch of sentences.
        """
        self.model.eval()
        self.tokenizer.model_max_length = self.config.max_length

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.config.max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                early_stopping=True
            )
        corrected_sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return corrected_sentences

    def evaluate_texts(self, sources: List[str], references: List[str]) -> Dict[str,float]:
        """
        Evaluate model on lists of source and reference strings (not tokenized datasets).
        Returns Exact Match Accuracy and BLEU.
        """
        if len(sources) == 0:
            return {'Exact Match Accuracy': 0.0, 'Bleu Score': 0.0}

        self.model.eval()
        predictions = []
        batch_size = self.config.eval_batch_size
        for i in tqdm(range(0, len(sources), batch_size), desc="Validating"):
            batch = sources[i:i + batch_size]
            preds = self.batch_correct(batch)
            predictions.extend(preds)

        # Ensure lengths match
        if len(predictions) != len(references):
            # pad predictions to avoid crash (shouldn't normally happen)
            predictions = predictions[:len(references)]
            while len(predictions) < len(references):
                predictions.append("")

        # Exact match
        exact_matches = sum([1 for p, r in zip(predictions, references) if p.strip() == r.strip()])
        exact_match_accuracy = exact_matches / len(references) if len(references) > 0 else 0.0

        # BLEU
        bleu = BLEU()
        try:
            bleu_score = bleu.corpus_score(predictions, [references]).score
        except Exception as e:
            print("BLEU computation failed:", e)
            bleu_score = 0.0

        return {
            'Exact Match Accuracy': exact_match_accuracy,
            'Bleu Score': bleu_score
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save adapter (PEFT) and tokenizer & config
        try:
            # PeftModel.save_pretrained will save adapter files
            self.model.save_pretrained(path)
        except Exception:
            # fallback to base_model save
            if hasattr(self.model, 'base_model'):
                self.model.base_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f'Model and tokenizer saved to {path}')

    @classmethod
    def load(cls, path: str, config: Optional[GECConfig] = None):
        """
        Load the model and tokenizer from a path (expects a PEFT adapter saved there).
        """
        if config is None:
            config = GECConfig()
        instance = cls(config)

        # Load tokenizer
        instance.tokenizer = T5Tokenizer.from_pretrained(path)

        # Load PEFT config from path
        peft_config = PeftConfig.from_pretrained(path)

        base_model = T5ForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float32
        )

        instance.model = PeftModel.from_pretrained(
            base_model,
            path,
            adapter_name="default",
            adapter_kwargs={"use_safetensors": True}
        )

        # Make sure correct adapter is active
        try:
            instance.model.set_adapter("default")
        except Exception:
            pass

        instance.model.to(config.device)

        # Print trainable parameters (if available)
        try:
            instance.model.print_trainable_parameters()
        except Exception:
            pass

        print(f"LoRA-adapted model loaded from {path}")
        return instance

    def evaluate(self, source_file: str, reference_file: str) -> Dict[str,float]:
        """
        Evaluate using files of lines (source_file and reference_file).
        """
        with open(source_file, 'r', encoding='utf-8') as src_f:
            sources = [line.strip() for line in src_f if line.strip()]

        with open(reference_file, 'r', encoding='utf-8') as ref_f:
            references = [line.strip() for line in ref_f if line.strip()]

        return self.evaluate_texts(sources, references)

    def make_pred(self, source_file: str, output_file: str):
        """
        Make predictions for CSV input with a 'source' column and save predictions as a new CSV.
        """
        df = pd.read_csv(source_file)
        if 'source' not in df.columns:
            raise ValueError("Input CSV must contain a 'source' column.")
        sources = df['source'].tolist()
        predictions = []
        batch_size = self.config.eval_batch_size
        for i in tqdm(range(0, len(sources), batch_size), desc="Generating predictions"):
            batch = sources[i:i + batch_size]
            preds = self.batch_correct(batch)
            predictions.extend(preds)

        df['prediction'] = predictions
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEC using T5")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--m2_file", type=str, help="Path to M2 file for training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--source_file", type=str, help="Path to source sentences")
    parser.add_argument("--reference_file", type=str, help="Path to reference corrections")
    parser.add_argument("--correct", action="store_true", help="Correct sentences")
    parser.add_argument("--input_file", type=str, help="Path to input sentences")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument("--model_path", type=str, default="./gec_model_outputs", help="Path to save/load model")
    parser.add_argument("--test_m2_file", type=str, help="Path to M2 file for evaluation")
    parser.add_argument("--submission_file", type=str, help="path to submission.csv file")
    parser.add_argument("--predictions", type=str, help="path to save submission.csv file")
    args = parser.parse_args()

    config = GECConfig(output_dir=args.model_path)

    if args.train and args.m2_file:
        corrector = GECorrector(config)
        train_dataset, val_dataset, train_texts, val_texts = corrector.load_and_prepare_data(args.m2_file)
        corrector.train(train_dataset, val_dataset, train_texts, val_texts)
        corrector.save(args.model_path)
    else:
        # If not training, attempt to load a saved LoRA-adapted model
        if os.path.isdir(args.model_path):
            corrector = GECorrector.load(args.model_path, config)
        else:
            # create default instance (no adapter loaded)
            corrector = GECorrector(config)

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
        for i in tqdm(range(0, len(sentences), config.eval_batch_size), desc="Generating predictions"):
            batch = sentences[i:i + config.eval_batch_size]
            preds = corrector.batch_correct(batch)
            corrected_sentences.extend(preds)

        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for sentence in corrected_sentences:
                    f.write(f"{sentence}\n")
            print("Corrected sentences saved ")
        else:
            for original, corrected in zip(sentences, corrected_sentences):
                print(f"Original: {original}")
                print("Corrected:")
                print(corrected)
                print("-" * 50)

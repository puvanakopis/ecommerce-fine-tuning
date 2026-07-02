"""
evaluate_model.py
==================
Computes BLEU, ROUGE, and perplexity for the fine-tuned model against a
held-out test set, and builds ground-truth vs. prediction comparison
tables. Used by notebook 06.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Computes standard NLG metrics for the ecommerce support model."""

    def __init__(self):
        self._rouge = None
        self._bleu = None

    # ------------------------------------------------------------------
    def _load_metrics(self):
        import evaluate

        if self._rouge is None:
            self._rouge = evaluate.load("rouge")
        if self._bleu is None:
            self._bleu = evaluate.load("sacrebleu")

    # ------------------------------------------------------------------
    def compute_rouge(self, predictions: List[str], references: List[str]) -> dict:
        """Compute ROUGE-1/2/L F-measure scores.

        Args:
            predictions: Model-generated responses.
            references: Ground-truth responses.

        Returns:
            Dict of ROUGE scores.
        """
        self._load_metrics()
        return self._rouge.compute(predictions=predictions, references=references)

    # ------------------------------------------------------------------
    def compute_bleu(self, predictions: List[str], references: List[str]) -> dict:
        """Compute corpus-level BLEU (via sacreBLEU) score.

        Args:
            predictions: Model-generated responses.
            references: Ground-truth responses.

        Returns:
            Dict with a `score` key (0-100 BLEU).
        """
        self._load_metrics()
        refs_formatted = [[r] for r in references]
        result = self._bleu.compute(predictions=predictions, references=refs_formatted)
        return result

    # ------------------------------------------------------------------
    def compute_perplexity(self, model, tokenizer, texts: List[str], device: str = "cuda") -> float:
        """Compute average perplexity of `texts` under `model`.

        Args:
            model: A causal LM (HF or Unsloth-wrapped).
            tokenizer: Matching tokenizer.
            texts: List of full text strings (instruction+response) to score.
            device: Device to run scoring on.

        Returns:
            Mean perplexity across all texts.
        """
        import math

        import torch

        model.eval()
        losses = []
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
                out = model(**enc, labels=enc["input_ids"])
                losses.append(out.loss.item())
        mean_loss = sum(losses) / max(len(losses), 1)
        return math.exp(mean_loss)

    # ------------------------------------------------------------------
    def build_comparison_table(
        self,
        instructions: List[str],
        ground_truths: List[str],
        predictions: List[str],
    ) -> pd.DataFrame:
        """Build a side-by-side comparison DataFrame for qualitative review.

        Returns:
            DataFrame with columns: instruction, ground_truth, prediction.
        """
        return pd.DataFrame(
            {
                "instruction": instructions,
                "ground_truth": ground_truths,
                "prediction": predictions,
            }
        )

    # ------------------------------------------------------------------
    def full_report(
        self,
        instructions: List[str],
        ground_truths: List[str],
        predictions: List[str],
        model=None,
        tokenizer=None,
        full_texts_for_ppl: Optional[List[str]] = None,
        device: str = "cuda",
    ) -> dict:
        """Run the complete evaluation suite and return a metrics dict plus
        the comparison table.

        Returns:
            {
              "rouge": {...}, "bleu": {...}, "perplexity": float | None,
              "comparison_table": pd.DataFrame
            }
        """
        logger.info(f"Evaluating on {len(predictions)} samples...")
        rouge_scores = self.compute_rouge(predictions, ground_truths)
        bleu_scores = self.compute_bleu(predictions, ground_truths)

        ppl = None
        if model is not None and tokenizer is not None and full_texts_for_ppl:
            ppl = self.compute_perplexity(model, tokenizer, full_texts_for_ppl, device=device)

        table = self.build_comparison_table(instructions, ground_truths, predictions)

        logger.info(f"ROUGE: {rouge_scores}")
        logger.info(f"BLEU: {bleu_scores.get('score')}")
        if ppl is not None:
            logger.info(f"Perplexity: {ppl:.3f}")

        return {
            "rouge": rouge_scores,
            "bleu": bleu_scores,
            "perplexity": ppl,
            "comparison_table": table,
        }

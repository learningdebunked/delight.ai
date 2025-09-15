"""Evaluation metrics for cultural adaptation and emotion detection."""

from typing import Dict, List, Any, Union, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import spacy

class MetricCalculator:
    """Class to compute various evaluation metrics."""
    
    def __init__(self, lang: str = 'en'):
        """Initialize with language for NLP tasks."""
        self.lang = lang
        self.nlp = spacy.load('en_core_web_sm' if lang == 'en' else 'xx_ent_wiki_sm')
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_classification_metrics(
        self,
        predictions: Union[List[int], np.ndarray],
        references: Union[List[int], np.ndarray],
        labels: List[str] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(references, predictions)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average=average, zero_division=0
        )
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
        
        # Per-class metrics if labels are provided
        if labels is not None:
            report = classification_report(
                references, predictions, 
                target_names=labels, 
                output_dict=True,
                zero_division=0
            )
            metrics['per_class'] = report
            
        return metrics
    
    def compute_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores for text generation tasks."""
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            score = self.rouge.score(ref, pred)
            for k in scores:
                scores[k].append(score[k].fmeasure)
        
        return {k: np.mean(v) for k, v in scores.items()}
    
    def compute_bert_score(
        self,
        predictions: List[str],
        references: List[str],
        lang: str = 'en'
    ) -> Dict[str, float]:
        """Compute BERTScore for semantic similarity."""
        P, R, F1 = bert_score(predictions, references, lang=lang)
        return {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item()
        }
    
    def compute_cultural_metrics(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        cultural_dimensions: List[str]
    ) -> Dict[str, float]:
        """Compute metrics specific to cultural adaptation."""
        metrics = {}
        
        for dim in cultural_dimensions:
            pred_vals = [p.get(dim, 0) for p in predictions]
            ref_vals = [r.get(dim, 0) for r in references]
            
            # Mean absolute error for each dimension
            mae = np.mean(np.abs(np.array(pred_vals) - np.array(ref_vals)))
            metrics[f'{dim}_mae'] = mae
            
            # Direction accuracy (whether prediction is on the same side of neutral)
            neutral = 0.5  # assuming 0-1 scale with 0.5 as neutral
            pred_sides = np.array(pred_vals) > neutral
            ref_sides = np.array(ref_vals) > neutral
            metrics[f'{dim}_direction_acc'] = np.mean(pred_sides == ref_sides)
        
        # Overall metrics
        all_preds = np.array([p[d] for p in predictions for d in cultural_dimensions])
        all_refs = np.array([r[d] for r in references for d in cultural_dimensions])
        
        metrics.update({
            'overall_mae': np.mean(np.abs(all_preds - all_refs)),
            'overall_direction_acc': np.mean((all_preds > 0.5) == (all_refs > 0.5))
        })
        
        return metrics
    
    def compute_emotion_metrics(
        self,
        predictions: List[Dict[str, float]],
        references: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute metrics for emotion detection."""
        emotions = list(references[0].keys()) if references else []
        metrics = {}
        
        for emo in emotions:
            pred_vals = [p.get(emo, 0) for p in predictions]
            ref_vals = [r.get(emo, 0) for r in references]
            
            # Mean squared error for each emotion
            mse = np.mean((np.array(pred_vals) - np.array(ref_vals)) ** 2)
            metrics[f'{emo}_mse'] = mse
            
            # Pearson correlation
            if len(set(pred_vals)) > 1 and len(set(ref_vals)) > 1:
                corr = np.corrcoef(pred_vals, ref_vals)[0, 1]
                metrics[f'{emo}_corr'] = corr
        
        return metrics

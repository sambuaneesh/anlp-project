"""
Evaluation metrics and utilities for GraphFC.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class FactCheckingMetrics:
    """
    Metrics calculator for fact-checking evaluation.
    """
    
    def __init__(self):
        self.predictions = []
        self.true_labels = []
        self.confidences = []
    
    def add_prediction(self, prediction: str, true_label: str, confidence: float = 1.0):
        """
        Add a prediction to the metrics calculator.
        
        Args:
            prediction: Predicted label ('True' or 'False')
            true_label: Ground truth label ('True' or 'False')
            confidence: Confidence score for the prediction
        """
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.confidences.append(confidence)
    
    def add_batch_predictions(self, 
                            predictions: List[str], 
                            true_labels: List[str], 
                            confidences: List[float] = None):
        """
        Add multiple predictions at once.
        
        Args:
            predictions: List of predicted labels
            true_labels: List of ground truth labels
            confidences: List of confidence scores
        """
        if confidences is None:
            confidences = [1.0] * len(predictions)
        
        if len(predictions) != len(true_labels) or len(predictions) != len(confidences):
            raise ValueError("All input lists must have the same length")
        
        self.predictions.extend(predictions)
        self.true_labels.extend(true_labels)
        self.confidences.extend(confidences)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Returns:
            Dictionary containing various metrics
        """
        if not self.predictions:
            return {}
        
        # Convert string labels to binary
        y_true = [1 if label == 'True' else 0 for label in self.true_labels]
        y_pred = [1 if pred == 'True' else 0 for pred in self.predictions]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Confidence-based metrics
        avg_confidence = np.mean(self.confidences)
        confidence_correct = np.mean([conf for i, conf in enumerate(self.confidences) 
                                    if y_true[i] == y_pred[i]])
        confidence_incorrect = np.mean([conf for i, conf in enumerate(self.confidences) 
                                      if y_true[i] != y_pred[i]])
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'avg_confidence': avg_confidence,
            'confidence_correct': confidence_correct if not np.isnan(confidence_correct) else 0.0,
            'confidence_incorrect': confidence_incorrect if not np.isnan(confidence_incorrect) else 0.0,
            'total_examples': len(self.predictions)
        }
        
        # Per-class metrics
        class_names = ['False', 'True']
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                metrics[f'{class_name.lower()}_precision'] = precision[i]
                metrics[f'{class_name.lower()}_recall'] = recall[i]
                metrics[f'{class_name.lower()}_f1'] = f1[i]
                metrics[f'{class_name.lower()}_support'] = support[i]
        
        # Confusion matrix elements
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            })
        
        return metrics
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Get detailed results including per-example information.
        
        Returns:
            Dictionary with detailed results
        """
        metrics = self.calculate_metrics()
        
        # Per-example results
        per_example_results = []
        for i, (pred, true, conf) in enumerate(zip(self.predictions, self.true_labels, self.confidences)):
            per_example_results.append({
                'index': i,
                'prediction': pred,
                'true_label': true,
                'confidence': conf,
                'correct': pred == true
            })
        
        return {
            'metrics': metrics,
            'per_example_results': per_example_results,
            'summary': {
                'total_examples': len(self.predictions),
                'correct_predictions': sum(1 for r in per_example_results if r['correct']),
                'incorrect_predictions': sum(1 for r in per_example_results if not r['correct'])
            }
        }
    
    def reset(self):
        """Reset the metrics calculator."""
        self.predictions = []
        self.true_labels = []
        self.confidences = []


def evaluate_model_on_dataset(model, dataset_examples, batch_size: int = 32) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset of examples.
    
    Args:
        model: The fact-checking model to evaluate
        dataset_examples: List of FactCheckingExample objects
        batch_size: Batch size for processing
        
    Returns:
        Dictionary containing evaluation results
    """
    metrics = FactCheckingMetrics()
    
    # Process examples in batches
    for i in range(0, len(dataset_examples), batch_size):
        batch = dataset_examples[i:i + batch_size]
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(dataset_examples) + batch_size - 1)//batch_size}")
        
        # Prepare batch inputs
        batch_claims_evidence = [(ex.claim, ex.evidence) for ex in batch]
        
        # Get model predictions
        try:
            results = model.batch_fact_check(batch_claims_evidence)
            
            # Extract predictions and confidences
            for j, result in enumerate(results):
                example = batch[j]
                metrics.add_prediction(
                    prediction=result.label,
                    true_label=example.label,
                    confidence=result.confidence
                )
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add default predictions for failed examples
            for example in batch:
                metrics.add_prediction(
                    prediction="False",
                    true_label=example.label,
                    confidence=0.0
                )
    
    return metrics.get_detailed_results()


def compare_models(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare evaluation results from multiple models.
    
    Args:
        model_results: Dictionary mapping model names to their evaluation results
        
    Returns:
        Dictionary containing comparison results
    """
    comparison = {
        'models': list(model_results.keys()),
        'metrics_comparison': {},
        'best_model_per_metric': {}
    }
    
    # Extract metrics for comparison
    all_metrics = set()
    for model_name, results in model_results.items():
        if 'metrics' in results:
            all_metrics.update(results['metrics'].keys())
    
    # Compare each metric
    for metric in all_metrics:
        metric_values = {}
        for model_name, results in model_results.items():
            if 'metrics' in results and metric in results['metrics']:
                metric_values[model_name] = results['metrics'][metric]
        
        if metric_values:
            comparison['metrics_comparison'][metric] = metric_values
            # Find best model for this metric (higher is better for most metrics)
            best_model = max(metric_values.items(), key=lambda x: x[1])
            comparison['best_model_per_metric'][metric] = {
                'model': best_model[0],
                'value': best_model[1]
            }
    
    return comparison


def print_evaluation_report(results: Dict[str, Any], model_name: str = "Model"):
    """
    Print a formatted evaluation report.
    
    Args:
        results: Evaluation results from evaluate_model_on_dataset
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Report for {model_name}")
    print(f"{'='*60}")
    
    if 'metrics' not in results:
        print("No metrics available in results")
        return
    
    metrics = results['metrics']
    
    # Main metrics
    print(f"Overall Performance:")
    print(f"  Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"  Macro F1:     {metrics.get('macro_f1', 0):.4f}")
    print(f"  Micro F1:     {metrics.get('micro_f1', 0):.4f}")
    print(f"  Total Examples: {metrics.get('total_examples', 0)}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    for class_name in ['true', 'false']:
        if f'{class_name}_f1' in metrics:
            print(f"  {class_name.capitalize()}_F1:      {metrics[f'{class_name}_f1']:.4f}")
            print(f"  {class_name.capitalize()}_Precision: {metrics[f'{class_name}_precision']:.4f}")
            print(f"  {class_name.capitalize()}_Recall:    {metrics[f'{class_name}_recall']:.4f}")
    
    # Confidence metrics
    print(f"\nConfidence Analysis:")
    print(f"  Average Confidence:       {metrics.get('avg_confidence', 0):.4f}")
    print(f"  Confidence (Correct):     {metrics.get('confidence_correct', 0):.4f}")
    print(f"  Confidence (Incorrect):   {metrics.get('confidence_incorrect', 0):.4f}")
    
    # Confusion matrix
    if all(key in metrics for key in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']):
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
    
    print(f"{'='*60}\n")
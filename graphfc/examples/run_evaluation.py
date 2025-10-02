"""
Comprehensive evaluation script for GraphFC and baseline models.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.graphfc import GraphFC
from models.baselines import create_baseline_model
from agents.llm_client import LLMClient
from datasets.loaders import load_dataset
from evaluation.metrics import evaluate_model_on_dataset, print_evaluation_report, compare_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_models(model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Set up all models for evaluation.
    
    Args:
        model_configs: Dictionary of model configurations
        
    Returns:
        Dictionary of initialized models
    """
    models = {}
    
    for model_name, config in model_configs.items():
        logger.info(f"Setting up model: {model_name}")
        
        # Create LLM client
        llm_client = LLMClient(
            model=config.get("llm_model", "gpt-3.5-turbo-0125"),
            api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 2048)
        )
        
        # Create model
        model_type = config.get("type", "graphfc")
        if model_type == "graphfc":
            models[model_name] = GraphFC(
                llm_client=llm_client,
                k_shot_examples=config.get("k_shot_examples", 10),
                early_stop=config.get("early_stop", True)
            )
        elif model_type in ["direct", "decomposition"]:
            models[model_name] = create_baseline_model(
                baseline_type=model_type,
                llm_client=llm_client,
                **config.get("baseline_kwargs", {})
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return models


def run_evaluation(args):
    """
    Run the evaluation experiment.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting GraphFC evaluation")
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset from {args.data_path}")
    try:
        examples = load_dataset(args.dataset, args.data_path)
        logger.info(f"Loaded {len(examples)} examples")
        
        # Limit examples if specified
        if args.max_examples and args.max_examples < len(examples):
            examples = examples[:args.max_examples]
            logger.info(f"Limited to {args.max_examples} examples for faster evaluation")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Define model configurations
    model_configs = {
        "GraphFC": {
            "type": "graphfc",
            "llm_model": args.model,
            "k_shot_examples": args.k_shot_examples,
            "early_stop": args.early_stop
        }
    }
    
    # Add baselines if requested
    if args.include_baselines:
        model_configs.update({
            "Direct": {
                "type": "direct",
                "llm_model": args.model
            },
            "Decomposition": {
                "type": "decomposition", 
                "llm_model": args.model,
                "baseline_kwargs": {"k_shot_examples": args.k_shot_examples}
            }
        })
    
    # Set up models
    try:
        models = setup_models(model_configs)
    except Exception as e:
        logger.error(f"Failed to setup models: {e}")
        return
    
    # Run evaluation for each model
    all_results = {}
    
    for model_name, model in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Run evaluation
            results = evaluate_model_on_dataset(
                model=model,
                dataset_examples=examples,
                batch_size=args.batch_size
            )
            
            all_results[model_name] = results
            
            # Print results
            print_evaluation_report(results, model_name)
            
            # Save individual results
            if args.output_dir:
                output_path = Path(args.output_dir) / f"{model_name}_{args.dataset}_results.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Saved {model_name} results to {output_path}")
        
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare models if multiple models were evaluated
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("Model Comparison")
        logger.info(f"{'='*60}")
        
        comparison = compare_models(all_results)
        
        # Print comparison
        print("\nModel Comparison:")
        print("-" * 40)
        
        key_metrics = ['accuracy', 'macro_f1', 'avg_confidence']
        for metric in key_metrics:
            if metric in comparison['metrics_comparison']:
                print(f"\n{metric.upper()}:")
                for model_name, value in comparison['metrics_comparison'][metric].items():
                    print(f"  {model_name:15}: {value:.4f}")
                
                if metric in comparison['best_model_per_metric']:
                    best = comparison['best_model_per_metric'][metric]
                    print(f"  Best: {best['model']} ({best['value']:.4f})")
        
        # Save comparison results
        if args.output_dir:
            comparison_path = Path(args.output_dir) / f"model_comparison_{args.dataset}.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            logger.info(f"Saved comparison results to {comparison_path}")
    
    # Save summary
    if args.output_dir:
        summary = {
            "dataset": args.dataset,
            "total_examples": len(examples),
            "models_evaluated": list(all_results.keys()),
            "experiment_config": {
                "model": args.model,
                "k_shot_examples": args.k_shot_examples,
                "batch_size": args.batch_size,
                "early_stop": args.early_stop,
                "include_baselines": args.include_baselines
            },
            "results_summary": {
                model_name: {
                    "accuracy": results.get("metrics", {}).get("accuracy", 0),
                    "macro_f1": results.get("metrics", {}).get("macro_f1", 0),
                    "total_examples": results.get("metrics", {}).get("total_examples", 0)
                }
                for model_name, results in all_results.items()
            }
        }
        
        summary_path = Path(args.output_dir) / f"evaluation_summary_{args.dataset}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved evaluation summary to {summary_path}")
    
    logger.info("Evaluation completed!")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate GraphFC and baseline models")
    
    # Dataset arguments
    parser.add_argument("--dataset", choices=["hover", "feverous", "scifact"], 
                       required=True, help="Dataset to evaluate on")
    parser.add_argument("--data_path", required=True, help="Path to dataset file")
    parser.add_argument("--max_examples", type=int, help="Maximum number of examples to evaluate")
    
    # Model arguments
    parser.add_argument("--model", default="gpt-3.5-turbo-0125", 
                       help="LLM model to use")
    parser.add_argument("--k_shot_examples", type=int, default=10,
                       help="Number of in-context examples for graph construction")
    parser.add_argument("--early_stop", action="store_true", default=True,
                       help="Enable early stopping in GraphFC")
    parser.add_argument("--include_baselines", action="store_true",
                       help="Include baseline models in evaluation")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--output_dir", help="Directory to save results")
    
    # Logging arguments
    parser.add_argument("--log_level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Run evaluation
    run_evaluation(args)


if __name__ == "__main__":
    main()
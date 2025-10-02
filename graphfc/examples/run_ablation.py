"""
Ablation study script for GraphFC components.
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
from agents.llm_client import LLMClient
from datasets.loaders import load_dataset
from evaluation.metrics import evaluate_model_on_dataset, print_evaluation_report, compare_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphFCAblationVariant:
    """
    GraphFC variant for ablation studies.
    Allows disabling specific components to study their impact.
    """
    
    def __init__(self, base_graphfc: GraphFC, **ablation_config):
        self.base_graphfc = base_graphfc
        self.ablation_config = ablation_config
    
    def fact_check(self, claim: str, evidence: List[str]):
        """
        Perform fact-checking with ablation modifications.
        This is a simplified implementation for demonstration.
        """
        # For full ablation implementation, you would modify the GraphFC workflow
        # based on ablation_config settings
        return self.base_graphfc.fact_check(claim, evidence)
    
    def batch_fact_check(self, claims_and_evidence):
        """Batch fact-checking with ablations."""
        return self.base_graphfc.batch_fact_check(claims_and_evidence)


def create_ablation_variants(base_llm_client: LLMClient) -> Dict[str, Any]:
    """
    Create different GraphFC variants for ablation study.
    
    Args:
        base_llm_client: Base LLM client to use
        
    Returns:
        Dictionary of model variants
    """
    variants = {}
    
    # Full GraphFC
    variants["GraphFC_Full"] = GraphFC(
        llm_client=base_llm_client,
        k_shot_examples=10,
        early_stop=True
    )
    
    # Variant without early stopping
    variants["GraphFC_NoEarlyStop"] = GraphFC(
        llm_client=base_llm_client,
        k_shot_examples=10,
        early_stop=False
    )
    
    # Variant with fewer examples
    variants["GraphFC_FewShot"] = GraphFC(
        llm_client=base_llm_client,
        k_shot_examples=5,
        early_stop=True
    )
    
    # Note: For true ablation study, you would need to modify the GraphFC class
    # to allow disabling specific components like evidence graph construction,
    # graph-guided planning, etc. This would require additional implementation.
    
    return variants


def run_ablation_study(args):
    """
    Run ablation study on GraphFC components.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting GraphFC ablation study")
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset from {args.data_path}")
    try:
        examples = load_dataset(args.dataset, args.data_path)
        logger.info(f"Loaded {len(examples)} examples")
        
        # Limit examples for faster ablation study
        if args.max_examples and args.max_examples < len(examples):
            examples = examples[:args.max_examples]
            logger.info(f"Limited to {args.max_examples} examples for ablation study")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create LLM client
    llm_client = LLMClient(
        model=args.model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0
    )
    
    # Create ablation variants
    model_variants = create_ablation_variants(llm_client)
    
    # Run evaluation for each variant
    all_results = {}
    
    for variant_name, model in model_variants.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {variant_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Run evaluation
            results = evaluate_model_on_dataset(
                model=model,
                dataset_examples=examples,
                batch_size=args.batch_size
            )
            
            all_results[variant_name] = results
            
            # Print results
            print_evaluation_report(results, variant_name)
            
            # Save individual results
            if args.output_dir:
                output_path = Path(args.output_dir) / f"{variant_name}_{args.dataset}_ablation.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Saved {variant_name} results to {output_path}")
        
        except Exception as e:
            logger.error(f"Error evaluating {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare variants
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("Ablation Study Results")
        logger.info(f"{'='*60}")
        
        comparison = compare_models(all_results)
        
        # Print ablation analysis
        print("\nAblation Study Analysis:")
        print("-" * 50)
        
        key_metrics = ['accuracy', 'macro_f1', 'avg_confidence']
        for metric in key_metrics:
            if metric in comparison['metrics_comparison']:
                print(f"\n{metric.upper()}:")
                values = comparison['metrics_comparison'][metric]
                
                # Sort by performance
                sorted_variants = sorted(values.items(), key=lambda x: x[1], reverse=True)
                
                for i, (variant_name, value) in enumerate(sorted_variants):
                    rank = f"#{i+1}"
                    print(f"  {rank:4} {variant_name:20}: {value:.4f}")
                
                # Calculate performance differences
                if len(sorted_variants) > 1:
                    best_value = sorted_variants[0][1]
                    print(f"\n  Performance differences from best:")
                    for variant_name, value in sorted_variants[1:]:
                        diff = best_value - value
                        print(f"    {variant_name:20}: -{diff:.4f}")
        
        # Save ablation results
        if args.output_dir:
            ablation_path = Path(args.output_dir) / f"ablation_study_{args.dataset}.json"
            with open(ablation_path, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            logger.info(f"Saved ablation study results to {ablation_path}")
    
    logger.info("Ablation study completed!")


def main():
    """Main function to run the ablation study."""
    parser = argparse.ArgumentParser(description="Run GraphFC ablation study")
    
    # Dataset arguments
    parser.add_argument("--dataset", choices=["hover", "feverous", "scifact"], 
                       required=True, help="Dataset to evaluate on")
    parser.add_argument("--data_path", required=True, help="Path to dataset file")
    parser.add_argument("--max_examples", type=int, default=100,
                       help="Maximum number of examples for ablation study")
    
    # Model arguments
    parser.add_argument("--model", default="gpt-3.5-turbo-0125", 
                       help="LLM model to use")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--output_dir", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Run ablation study
    run_ablation_study(args)


if __name__ == "__main__":
    main()
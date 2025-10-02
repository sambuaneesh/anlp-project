#!/usr/bin/env python3
"""
Test script to evaluate GraphFC framework using Gemini API.
This script loads environment variables, creates test data, and runs comprehensive evaluations.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main evaluation function."""
    logger.info("Starting GraphFC evaluation with Gemini API")
    
    # Check environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return
    
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-lite")
    logger.info(f"Using model: {model_name}")
    
    try:
        # Import required modules
        from agents.llm_client import LLMClient
        from models.graphfc import GraphFC
        from models.baselines import create_baseline_model
        from evaluation.metrics import evaluate_model_on_dataset, compare_models, print_evaluation_report
        
        # Load test dataset
        test_data_path = project_root / "data" / "test_dataset.json"
        with open(test_data_path, 'r') as f:
            test_examples = json.load(f)
        
        logger.info(f"Loaded {len(test_examples)} test examples")
        
        # Initialize LLM client with Gemini
        llm_client = LLMClient(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=2048
        )
        
        logger.info("Initialized Gemini LLM client")
        
        # Initialize GraphFC framework
        graphfc = GraphFC(
            llm_client=llm_client,
            k_shot_examples=3,  # Smaller for testing
            early_stop=False,   # Get complete analysis
            random_seed=42
        )
        
        logger.info("Initialized GraphFC framework")
        
        # Initialize baseline models
        direct_baseline = create_baseline_model("direct", llm_client)
        decomp_baseline = create_baseline_model("decomposition", llm_client)
        
        logger.info("Initialized baseline models")
        
        # Convert test data to expected format
        formatted_examples = []
        for example in test_examples:
            formatted_example = {
                'id': example['id'],
                'claim': example['claim'],
                'evidence': example['evidence'],
                'label': example['label'],
                'annotation_type': example.get('annotation_type', 'fact_verification'),
                'hop_count': example.get('hop_count', 1)
            }
            formatted_examples.append(formatted_example)
        
        # Evaluate models
        models = {
            "GraphFC": graphfc,
            "Direct": direct_baseline,
            "Decomposition": decomp_baseline
        }
        
        all_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            try:
                results = evaluate_model_on_dataset(
                    model=model,
                    dataset_examples=formatted_examples,
                    batch_size=1  # Process one at a time for detailed logging
                )
                all_results[model_name] = results
                
                # Print individual model results
                print(f"\n{'='*60}")
                print(f"Results for {model_name}")
                print(f"{'='*60}")
                print_evaluation_report(results, model_name)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        # Compare models if we have results
        if len(all_results) > 1:
            print(f"\n{'='*60}")
            print("MODEL COMPARISON")
            print(f"{'='*60}")
            
            comparison = compare_models(all_results)
            
            print("\nMetrics Comparison:")
            for metric, values in comparison['metrics_comparison'].items():
                print(f"\n{metric.upper()}:")
                for model_name, value in values.items():
                    print(f"  {model_name:15}: {value:.4f}")
            
            # Save detailed results
            results_file = project_root / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"Detailed results saved to {results_file}")
        
        logger.info("Evaluation completed successfully!")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed in the virtual environment")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
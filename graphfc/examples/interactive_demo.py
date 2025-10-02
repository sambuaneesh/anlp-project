"""
Interactive demo script for GraphFC.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.graphfc import GraphFC
from models.baselines import create_baseline_model
from agents.llm_client import LLMClient


def print_separator(char="=", length=60):
    """Print a separator line."""
    print(char * length)


def print_graph_summary(graph, graph_type="Graph"):
    """Print a summary of a graph."""
    print(f"\n{graph_type} ({len(graph.triplets)} triplets):")
    for i, triplet in enumerate(graph.triplets, 1):
        unknown_marker = ""
        if triplet.has_unknown_entities():
            unknown_count = triplet.unknown_entity_count()
            unknown_marker = f" [{unknown_count} unknown]"
        print(f"  {i}. {triplet}{unknown_marker}")


def interactive_demo():
    """Run an interactive demo of GraphFC."""
    print_separator("=")
    print("GraphFC Interactive Demo")
    print_separator("=")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable")
        return
    
    # Initialize models
    print("Initializing models...")
    llm_client = LLMClient(model="gpt-3.5-turbo-0125", api_key=api_key)
    
    models = {
        "GraphFC": GraphFC(llm_client, k_shot_examples=5),  # Fewer examples for demo
        "Direct": create_baseline_model("direct", llm_client),
        "Decomposition": create_baseline_model("decomposition", llm_client, k_shot_examples=5)
    }
    
    print("Models initialized successfully!")
    
    # Predefined examples
    examples = [
        {
            "claim": "The founder of the school was the daughter of Christopher.",
            "evidence": [
                "St Hugh's College was founded by Elizabeth Wordsworth in 1886.",
                "Elizabeth Wordsworth was the daughter of Christopher Wordsworth.",
                "Christopher Wordsworth was the Bishop of Lincoln.",
                "Kathleen was the sister of Christopher Wordsworth.",
                "Kathleen was the principal of Somerville College."
            ],
            "description": "Example from the GraphFC paper"
        },
        {
            "claim": "The capital of France is London.",
            "evidence": [
                "Paris is the capital and most populous city of France.",
                "London is the capital of England and the United Kingdom.",
                "France is a country in Western Europe."
            ],
            "description": "Simple false claim"
        },
        {
            "claim": "Einstein won the Nobel Prize in Physics in 1921.",
            "evidence": [
                "Albert Einstein won the Nobel Prize in Physics in 1921.",
                "The Nobel Prize in Physics 1921 was awarded to Albert Einstein.",
                "Einstein was awarded for his services to Theoretical Physics."
            ],
            "description": "Simple true claim"
        }
    ]
    
    while True:
        print_separator("-")
        print("\nOptions:")
        print("1. Use predefined example")
        print("2. Enter custom claim and evidence")
        print("3. Quit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == "3":
            print("Goodbye!")
            break
        
        elif choice == "1":
            # Show predefined examples
            print("\nPredefined Examples:")
            for i, example in enumerate(examples, 1):
                print(f"{i}. {example['description']}")
            
            try:
                example_choice = int(input(f"\nSelect example (1-{len(examples)}): ")) - 1
                if 0 <= example_choice < len(examples):
                    selected_example = examples[example_choice]
                    claim = selected_example["claim"]
                    evidence = selected_example["evidence"]
                else:
                    print("Invalid choice!")
                    continue
            except ValueError:
                print("Invalid input!")
                continue
        
        elif choice == "2":
            # Get custom input
            print("\nEnter your custom claim and evidence:")
            claim = input("Claim: ").strip()
            if not claim:
                print("Claim cannot be empty!")
                continue
            
            evidence = []
            print("Enter evidence passages (press Enter twice to finish):")
            while True:
                line = input(f"Evidence {len(evidence)+1}: ").strip()
                if not line:
                    break
                evidence.append(line)
            
            if not evidence:
                print("At least one evidence passage is required!")
                continue
        
        else:
            print("Invalid choice!")
            continue
        
        # Display the selected claim and evidence
        print_separator("-")
        print(f"Claim: {claim}")
        print(f"Evidence:")
        for i, ev in enumerate(evidence, 1):
            print(f"  {i}. {ev}")
        
        # Ask which models to run
        print(f"\nAvailable models: {', '.join(models.keys())}")
        model_choice = input("Which models to run? (comma-separated, or 'all'): ").strip()
        
        if model_choice.lower() == "all":
            selected_models = list(models.keys())
        else:
            selected_models = [m.strip() for m in model_choice.split(",")]
            selected_models = [m for m in selected_models if m in models]
        
        if not selected_models:
            print("No valid models selected!")
            continue
        
        # Run fact-checking
        print_separator("=")
        print("FACT-CHECKING RESULTS")
        print_separator("=")
        
        for model_name in selected_models:
            model = models[model_name]
            
            print(f"\n{model_name} Results:")
            print_separator("-")
            
            try:
                result = model.fact_check(claim, evidence)
                
                print(f"Result: {result.label}")
                print(f"Confidence: {result.confidence:.3f}")
                
                # Show additional details for GraphFC
                if model_name == "GraphFC":
                    print_graph_summary(result.claim_graph, "Claim Graph")
                    print_graph_summary(result.evidence_graph, "Evidence Graph")
                    
                    print(f"\nVerification Steps:")
                    for i, step in enumerate(result.verification_steps, 1):
                        print(f"  {i}. {step['step']}")
                        if step['step'] == 'graph_guided_checking':
                            for j, triplet_result in enumerate(step['triplet_results'], 1):
                                status = "✓" if triplet_result['result'] else "✗"
                                print(f"     {j}. {status} {triplet_result['triplet']} ({triplet_result['type']})")
                    
                    stats = result.statistics
                    print(f"\nStatistics:")
                    print(f"  Total triplets: {stats['total_triplets']}")
                    print(f"  Verified: {stats['verified_triplets']}")
                    print(f"  Failed: {stats['failed_triplets']}")
                
                # Show details for Decomposition baseline
                elif model_name == "Decomposition":
                    decomp_step = next((s for s in result.verification_steps if s['step'] == 'claim_decomposition'), None)
                    if decomp_step:
                        print(f"\nSub-claims:")
                        for i, sub_claim in enumerate(decomp_step['sub_claims'], 1):
                            result_info = decomp_step['sub_claim_results'][i-1]
                            status = "✓" if result_info['result'] else "✗"
                            print(f"  {i}. {status} {sub_claim}")
            
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main function to run the demo."""
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
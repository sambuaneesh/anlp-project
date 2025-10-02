"""
Basic example demonstrating GraphFC usage.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.graphfc import GraphFC
from agents.llm_client import LLMClient


def main():
    """
    Demonstrate basic GraphFC usage with a simple example.
    """
    print("GraphFC Basic Example")
    print("=" * 50)
    
    # Initialize the LLM client
    # Note: You need to set your OpenAI API key in the environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable")
        return
    
    llm_client = LLMClient(model="gpt-3.5-turbo-0125", api_key=api_key)
    
    # Initialize GraphFC
    graphfc = GraphFC(llm_client, k_shot_examples=5)  # Using fewer examples for demo
    
    # Example claim and evidence from the paper
    claim = "The founder of the school was the daughter of Christopher."
    evidence = [
        "St Hugh's College was founded by Elizabeth Wordsworth in 1886.",
        "Elizabeth Wordsworth was the daughter of Christopher Wordsworth.",
        "Christopher Wordsworth was the Bishop of Lincoln.",
        "Kathleen was the sister of Christopher Wordsworth.",
        "Kathleen was the principal of Somerville College."
    ]
    
    print(f"Claim: {claim}")
    print(f"Evidence: {evidence}")
    print("\nStarting fact-checking process...")
    print("-" * 50)
    
    try:
        # Perform fact-checking
        result = graphfc.fact_check(claim, evidence)
        
        # Display results
        print(f"\nFact-checking Result: {result.label}")
        print(f"Confidence: {result.confidence:.2f}")
        
        print(f"\nClaim Graph ({len(result.claim_graph.triplets)} triplets):")
        for i, triplet in enumerate(result.claim_graph.triplets, 1):
            print(f"  {i}. {triplet}")
        
        print(f"\nEvidence Graph ({len(result.evidence_graph.triplets)} triplets):")
        for i, triplet in enumerate(result.evidence_graph.triplets, 1):
            print(f"  {i}. {triplet}")
        
        print(f"\nVerification Steps:")
        for i, step in enumerate(result.verification_steps, 1):
            print(f"  {i}. {step['step']}")
            if step['step'] == 'graph_guided_checking':
                for j, triplet_result in enumerate(step['triplet_results'], 1):
                    status = "✓" if triplet_result['result'] else "✗"
                    print(f"     {j}. {status} {triplet_result['triplet']} ({triplet_result['type']})")
        
        print(f"\nStatistics:")
        stats = result.statistics
        print(f"  Total triplets: {stats['total_triplets']}")
        print(f"  Verified: {stats['verified_triplets']}")
        print(f"  Failed: {stats['failed_triplets']}")
        print(f"  Completions: {stats['completion_operations']}")
        print(f"  Matches: {stats['match_operations']}")
        
    except Exception as e:
        print(f"Error during fact-checking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
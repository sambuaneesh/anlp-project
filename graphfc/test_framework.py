#!/usr/bin/env python3
"""
Simple test to verify GraphFC framework works with Gemini API.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables from .env file
load_dotenv()

def test_basic_functionality():
    """Test basic GraphFC functionality."""
    print("Testing GraphFC with Gemini...")
    
    # Verify environment variables are loaded
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment. Please check your .env file.")
        return False
    
    try:
        # Import required modules
        from agents.llm_client import LLMClient
        print("✓ LLM Client imported")
        
        from models.graph import Entity, Triplet, ClaimGraph, EvidenceGraph
        print("✓ Graph models imported")
        
        from agents.graph_construction import GraphConstructionAgent
        print("✓ Graph construction agent imported")
        
        # Test LLM client
        llm_client = LLMClient(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-lite"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0
        )
        print("✓ LLM client initialized")
        
        # Test graph construction
        graph_agent = GraphConstructionAgent(llm_client, k_shot_examples=2)
        print("✓ Graph construction agent initialized")
        
        # Test with a simple claim
        test_claim = "Barack Obama was the first African American president of the United States."
        print(f"\\nTesting claim: {test_claim}")
        
        claim_graph = graph_agent.construct_claim_graph(test_claim)
        print(f"✓ Claim graph created with {len(claim_graph.triplets)} triplets")
        
        for i, triplet in enumerate(claim_graph.triplets):
            print(f"  Triplet {i+1}: {triplet}")
        
        # Test evidence graph
        test_evidence = [
            "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
            "Barack Obama is of African American heritage."
        ]
        
        evidence_graph = graph_agent.construct_evidence_graph(
            evidence_texts=test_evidence,
            known_entities=list(claim_graph.get_known_entities())
        )
        print(f"✓ Evidence graph created with {len(evidence_graph.triplets)} triplets")
        
        for i, triplet in enumerate(evidence_graph.triplets):
            print(f"  Evidence Triplet {i+1}: {triplet}")
        
        print("\\n🎉 Basic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_framework():
    """Test the full GraphFC framework."""
    print("\\n" + "="*60)
    print("Testing Full GraphFC Framework")
    print("="*60)
    
    try:
        from agents.llm_client import LLMClient
        from models.graphfc import GraphFC
        
        # Initialize framework
        llm_client = LLMClient(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-lite"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0
        )
        
        graphfc = GraphFC(
            llm_client=llm_client,
            k_shot_examples=2,
            early_stop=False,
            random_seed=42
        )
        print("✓ GraphFC framework initialized")
        
        # Test with sample data
        claim = "Barack Obama was the first African American president of the United States."
        evidence = [
            "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
            "Barack Obama is of African American heritage.",
            "No previous U.S. president was of African American descent before Obama."
        ]
        
        print(f"\\nTesting fact-checking...")
        print(f"Claim: {claim}")
        print(f"Evidence: {evidence}")
        
        result = graphfc.fact_check(claim, evidence)
        
        print(f"\\n✓ Fact-checking completed!")
        print(f"Result: {result.label}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Claim graph triplets: {len(result.claim_graph.triplets)}")
        print(f"Evidence graph triplets: {len(result.evidence_graph.triplets)}")
        
        # Print verification steps
        print(f"\\nVerification steps:")
        for i, step in enumerate(result.verification_steps):
            print(f"  Step {i+1}: {step['step']}")
        
        print("\\n🎉 Full framework test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Full framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting GraphFC Tests...")
    
    # Test 1: Basic functionality
    basic_test = test_basic_functionality()
    
    if basic_test:
        # Test 2: Full framework
        full_test = test_full_framework()
        
        if full_test:
            print("\\n🌟 All tests PASSED! GraphFC is ready for evaluation.")
        else:
            print("\\n⚠️  Basic tests passed but full framework test failed.")
    else:
        print("\\n❌ Basic tests failed. Check your setup.")
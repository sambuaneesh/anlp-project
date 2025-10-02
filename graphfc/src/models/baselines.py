"""
Baseline fact-checking methods for comparison with GraphFC.
"""

import logging
from typing import List, Dict, Any, Tuple
from src.agents.llm_client import LLMClient
from src.models.graphfc import FactCheckingResult
from src.models.graph import ClaimGraph, EvidenceGraph

logger = logging.getLogger(__name__)


class DirectBaseline:
    """
    Direct prompting baseline for fact-checking.
    Uses zero-shot prompting to directly verify claims against evidence.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        logger.info("Initialized Direct baseline")
    
    def fact_check(self, claim: str, evidence: List[str]) -> FactCheckingResult:
        """
        Perform direct fact-checking using zero-shot prompting.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence texts
            
        Returns:
            FactCheckingResult object
        """
        # Combine evidence into a single text
        evidence_text = "\n".join(f"{i+1}. {text}" for i, text in enumerate(evidence))
        
        # Create prompt
        prompt = f"""Evidence:
{evidence_text}

Based on the above information, is it true that "{claim}"? 

Answer with either "True" or "False" followed by a brief explanation.

Answer:"""
        
        # Get response
        response = self.llm_client.generate(prompt)
        
        # Parse response
        label = "False"  # Default to False
        confidence = 0.5  # Default confidence
        
        response_lower = response.lower().strip()
        if response_lower.startswith('true'):
            label = "True"
            confidence = 0.8
        elif response_lower.startswith('false'):
            label = "False"
            confidence = 0.8
        
        # Create result object
        return FactCheckingResult(
            label=label,
            confidence=confidence,
            claim_graph=ClaimGraph(claim, []),
            evidence_graph=EvidenceGraph(evidence, []),
            verification_steps=[{
                "step": "direct_prompting",
                "response": response,
                "method": "zero_shot"
            }],
            statistics={
                "method": "direct",
                "total_triplets": 0,
                "verified_triplets": 1 if label == "True" else 0,
                "failed_triplets": 0 if label == "True" else 1
            }
        )
    
    def batch_fact_check(self, claims_and_evidence: List[Tuple[str, List[str]]]) -> List[FactCheckingResult]:
        """Batch fact-checking for multiple claims."""
        results = []
        for claim, evidence in claims_and_evidence:
            try:
                result = self.fact_check(claim, evidence)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in direct baseline: {e}")
                results.append(FactCheckingResult(
                    label="False",
                    confidence=0.0,
                    claim_graph=ClaimGraph(claim, []),
                    evidence_graph=EvidenceGraph(evidence, []),
                    verification_steps=[],
                    statistics={"error": str(e)}
                ))
        return results


class DecompositionBaseline:
    """
    Decomposition baseline that breaks claims into sub-claims for verification.
    """
    
    def __init__(self, llm_client: LLMClient, k_shot_examples: int = 10):
        self.llm_client = llm_client
        self.k_shot_examples = k_shot_examples
        logger.info("Initialized Decomposition baseline")
    
    def fact_check(self, claim: str, evidence: List[str]) -> FactCheckingResult:
        """
        Perform fact-checking using claim decomposition approach.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence texts
            
        Returns:
            FactCheckingResult object
        """
        # Step 1: Decompose claim into sub-claims
        sub_claims = self._decompose_claim(claim)
        
        # Step 2: Verify each sub-claim
        sub_claim_results = []
        overall_result = True
        
        evidence_text = "\n".join(f"{i+1}. {text}" for i, text in enumerate(evidence))
        
        for i, sub_claim in enumerate(sub_claims):
            logger.info(f"Verifying sub-claim {i+1}/{len(sub_claims)}: {sub_claim}")
            
            # Verify individual sub-claim
            verification_prompt = f"""Evidence:
{evidence_text}

Based on the above evidence, is the following statement true or false?

Statement: "{sub_claim}"

Answer with either "True" or "False" followed by a brief explanation.

Answer:"""
            
            response = self.llm_client.generate(verification_prompt)
            
            # Parse response
            sub_result = self._parse_verification_response(response)
            sub_claim_results.append({
                "sub_claim": sub_claim,
                "result": sub_result,
                "response": response
            })
            
            if not sub_result:
                overall_result = False
                break  # Early stopping
        
        # Calculate confidence based on sub-claim results
        verified_count = sum(1 for r in sub_claim_results if r["result"])
        confidence = verified_count / len(sub_claims) if sub_claims else 0.0
        
        label = "True" if overall_result else "False"
        
        return FactCheckingResult(
            label=label,
            confidence=confidence,
            claim_graph=ClaimGraph(claim, []),
            evidence_graph=EvidenceGraph(evidence, []),
            verification_steps=[{
                "step": "claim_decomposition",
                "sub_claims": [r["sub_claim"] for r in sub_claim_results],
                "sub_claim_results": sub_claim_results,
                "method": "textual_decomposition"
            }],
            statistics={
                "method": "decomposition",
                "total_sub_claims": len(sub_claims),
                "verified_sub_claims": verified_count,
                "failed_sub_claims": len(sub_claims) - verified_count
            }
        )
    
    def _decompose_claim(self, claim: str) -> List[str]:
        """
        Decompose a claim into atomic sub-claims.
        
        Args:
            claim: The claim to decompose
            
        Returns:
            List of sub-claims
        """
        decomposition_prompt = f"""## Task Description:
Break down the given claim into multiple sub-claims that:
- Cannot be further divided into simpler meaningful statements
- Has a clear truth value (true or false)
- Contains a single subject and predicate
- Does not contain logical connectives (and, or, if-then, etc.)

## Input Format:
A complex claim or statement.

## Output Format:
Each atomic proposition should be presented on a new line, numbered.

## Examples:

Input: "John founded Microsoft and was born in Seattle."
Output:
1. John founded Microsoft.
2. John was born in Seattle.

Input: "The university that was established in 1636 is located in Cambridge."
Output:
1. A university was established in 1636.
2. The university is located in Cambridge.

## Real Data:
Input: "{claim}"
Output:"""
        
        response = self.llm_client.generate(decomposition_prompt)
        
        # Parse response to extract sub-claims
        sub_claims = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and extract the claim
                cleaned_line = line
                if '. ' in line:
                    cleaned_line = line.split('. ', 1)[1]
                elif '- ' in line:
                    cleaned_line = line.split('- ', 1)[1]
                
                if cleaned_line:
                    sub_claims.append(cleaned_line.strip())
        
        # If no sub-claims extracted, use the original claim
        if not sub_claims:
            sub_claims = [claim]
        
        logger.info(f"Decomposed claim into {len(sub_claims)} sub-claims")
        return sub_claims
    
    def _parse_verification_response(self, response: str) -> bool:
        """Parse verification response to extract True/False result."""
        response_lower = response.lower().strip()
        
        if response_lower.startswith('true'):
            return True
        elif response_lower.startswith('false'):
            return False
        
        # Look for true/false anywhere in the response
        if 'true' in response_lower and 'false' not in response_lower:
            return True
        elif 'false' in response_lower and 'true' not in response_lower:
            return False
        
        # Default to False if unclear
        return False
    
    def batch_fact_check(self, claims_and_evidence: List[Tuple[str, List[str]]]) -> List[FactCheckingResult]:
        """Batch fact-checking for multiple claims."""
        results = []
        for claim, evidence in claims_and_evidence:
            try:
                result = self.fact_check(claim, evidence)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in decomposition baseline: {e}")
                results.append(FactCheckingResult(
                    label="False",
                    confidence=0.0,
                    claim_graph=ClaimGraph(claim, []),
                    evidence_graph=EvidenceGraph(evidence, []),
                    verification_steps=[],
                    statistics={"error": str(e)}
                ))
        return results


def create_baseline_model(baseline_type: str, llm_client: LLMClient, **kwargs):
    """
    Factory function to create baseline models.
    
    Args:
        baseline_type: Type of baseline ('direct' or 'decomposition')
        llm_client: LLM client to use
        **kwargs: Additional arguments for the baseline
        
    Returns:
        Baseline model instance
    """
    if baseline_type.lower() == 'direct':
        return DirectBaseline(llm_client)
    elif baseline_type.lower() == 'decomposition':
        return DecompositionBaseline(llm_client, **kwargs)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}. Supported: 'direct', 'decomposition'")
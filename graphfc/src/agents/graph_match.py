"""
Graph match agent for verifying triplets with only known entities.
"""

import logging
from typing import List, Dict, Any, Optional
from src.models.graph import Triplet, EvidenceGraph, Entity
from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


class GraphMatchAgent:
    """
    Agent responsible for verifying triplets that contain only known entities
    by matching them against the evidence graph.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the graph match agent.
        
        Args:
            llm_client: LLM client for generating responses
        """
        self.llm_client = llm_client
    
    def verify_triplet(self, 
                      triplet: Triplet, 
                      evidence_graph: EvidenceGraph, 
                      evidence_texts: List[str]) -> bool:
        """
        Verify a triplet against the evidence graph and texts.
        
        Args:
            triplet: The triplet to verify (should contain only known entities)
            evidence_graph: The evidence graph to match against
            evidence_texts: Original evidence texts for additional context
            
        Returns:
            True if the triplet is supported by evidence, False otherwise
        """
        if triplet.has_unknown_entities():
            raise ValueError("Graph match agent can only verify triplets with known entities")
        
        logger.info(f"Verifying triplet: {triplet}")
        
        # Get relevant triplets from evidence graph
        relevant_triplets = self._get_relevant_evidence_triplets(triplet, evidence_graph)
        
        # Create prompt for verification
        prompt = self._create_verification_prompt(triplet, relevant_triplets, evidence_texts)
        
        # Get response from LLM
        response = self.llm_client.generate(prompt)
        
        # Parse response to get verification result
        result = self._parse_verification_response(response)
        
        logger.info(f"Verification result for {triplet}: {result}")
        return result
    
    def batch_verify_triplets(self, 
                            triplets: List[Triplet], 
                            evidence_graph: EvidenceGraph, 
                            evidence_texts: List[str]) -> List[bool]:
        """
        Verify multiple triplets in batch.
        
        Args:
            triplets: List of triplets to verify
            evidence_graph: The evidence graph to match against
            evidence_texts: Original evidence texts for additional context
            
        Returns:
            List of verification results corresponding to input triplets
        """
        results = []
        
        for triplet in triplets:
            try:
                result = self.verify_triplet(triplet, evidence_graph, evidence_texts)
                results.append(result)
            except Exception as e:
                logger.error(f"Error verifying triplet {triplet}: {e}")
                results.append(False)  # Default to False on error
        
        return results
    
    def _get_relevant_evidence_triplets(self, 
                                      target_triplet: Triplet, 
                                      evidence_graph: EvidenceGraph) -> List[Triplet]:
        """
        Get evidence triplets that are relevant for verifying the target triplet.
        
        Args:
            target_triplet: The triplet being verified
            evidence_graph: The evidence graph to search
            
        Returns:
            List of relevant evidence triplets
        """
        relevant_triplets = []
        
        # Find triplets that share entities with the target triplet
        for evidence_triplet in evidence_graph.triplets:
            if self._is_triplet_relevant(target_triplet, evidence_triplet):
                relevant_triplets.append(evidence_triplet)
        
        # If no direct matches, include triplets containing any of the entities
        if not relevant_triplets:
            for evidence_triplet in evidence_graph.triplets:
                if (self._entities_match(target_triplet.subject, evidence_triplet.subject) or
                    self._entities_match(target_triplet.subject, evidence_triplet.object) or
                    self._entities_match(target_triplet.object, evidence_triplet.subject) or
                    self._entities_match(target_triplet.object, evidence_triplet.object)):
                    relevant_triplets.append(evidence_triplet)
        
        return relevant_triplets
    
    def _is_triplet_relevant(self, target_triplet: Triplet, evidence_triplet: Triplet) -> bool:
        """
        Check if an evidence triplet is relevant for verifying the target triplet.
        
        Args:
            target_triplet: The triplet being verified
            evidence_triplet: The evidence triplet to check
            
        Returns:
            True if the evidence triplet is relevant
        """
        # Direct match: same subject, predicate, and object
        if (self._entities_match(target_triplet.subject, evidence_triplet.subject) and
            self._predicates_match(target_triplet.predicate, evidence_triplet.predicate) and
            self._entities_match(target_triplet.object, evidence_triplet.object)):
            return True
        
        # Entities match but different predicates (could be related)
        if (self._entities_match(target_triplet.subject, evidence_triplet.subject) and
            self._entities_match(target_triplet.object, evidence_triplet.object)):
            return True
        
        # Shared entities (could provide context)
        if (self._entities_match(target_triplet.subject, evidence_triplet.subject) or
            self._entities_match(target_triplet.subject, evidence_triplet.object) or
            self._entities_match(target_triplet.object, evidence_triplet.subject) or
            self._entities_match(target_triplet.object, evidence_triplet.object)):
            return True
        
        return False
    
    def _entities_match(self, entity1: Entity, entity2: Entity) -> bool:
        """
        Check if two entities match (with some flexibility for name variations).
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities match
        """
        # Exact match
        if entity1.name == entity2.name:
            return True
        
        # Case-insensitive match
        if entity1.name.lower() == entity2.name.lower():
            return True
        
        # Check if one name is contained in the other (for partial matches)
        name1_lower = entity1.name.lower()
        name2_lower = entity2.name.lower()
        
        if (name1_lower in name2_lower and len(name1_lower) > 3) or \
           (name2_lower in name1_lower and len(name2_lower) > 3):
            return True
        
        return False
    
    def _predicates_match(self, predicate1: str, predicate2: str) -> bool:
        """
        Check if two predicates match (with some flexibility for variations).
        
        Args:
            predicate1: First predicate
            predicate2: Second predicate
            
        Returns:
            True if predicates match
        """
        # Exact match
        if predicate1 == predicate2:
            return True
        
        # Case-insensitive match
        if predicate1.lower() == predicate2.lower():
            return True
        
        # Handle common variations
        pred1_norm = self._normalize_predicate(predicate1)
        pred2_norm = self._normalize_predicate(predicate2)
        
        return pred1_norm == pred2_norm
    
    def _normalize_predicate(self, predicate: str) -> str:
        """
        Normalize predicate for better matching.
        
        Args:
            predicate: The predicate to normalize
            
        Returns:
            Normalized predicate
        """
        # Convert to lowercase and replace underscores/hyphens with spaces
        normalized = predicate.lower().replace('_', ' ').replace('-', ' ')
        
        # Handle common synonyms
        synonyms = {
            'founded by': 'founder of',
            'created by': 'creator of',
            'written by': 'author of',
            'daughter of': 'child of',
            'son of': 'child of'
        }
        
        return synonyms.get(normalized, normalized)
    
    def _create_verification_prompt(self, 
                                  triplet: Triplet, 
                                  relevant_triplets: List[Triplet], 
                                  evidence_texts: List[str]) -> str:
        """
        Create prompt for triplet verification.
        
        Args:
            triplet: The triplet to verify
            relevant_triplets: Relevant evidence triplets
            evidence_texts: Original evidence texts
            
        Returns:
            Formatted prompt for the LLM
        """
        # Format evidence texts
        evidence_str = "\n".join(f"{i+1}. {text}" for i, text in enumerate(evidence_texts))
        
        # Format relevant triplets
        triplets_str = ""
        if relevant_triplets:
            triplets_str = "\n".join(f"- <{t.subject.name}, {t.predicate}, {t.object.name}>" 
                                   for t in relevant_triplets)
        else:
            triplets_str = "No directly relevant triplets found."
        
        prompt = f"""Evidence:
{evidence_str}

Relevant Evidence Triplets:
{triplets_str}

Using the provided evidence and relevant triplets, determine whether the given triplet is true or false.

Triplet to verify: <{triplet.subject.name}, {triplet.predicate}, {triplet.object.name}>

Instructions:
1. Check if the triplet is directly supported by the evidence texts
2. Check if the triplet matches or is consistent with the relevant evidence triplets
3. Consider semantic equivalence (e.g., "founded by" vs "founder of")
4. If there's any contradiction or lack of support, return false
5. Only return true if there's clear evidence supporting the triplet

Answer with either "true" or "false" followed by a brief explanation.

Answer:"""
        
        return prompt
    
    def _parse_verification_response(self, response: str) -> bool:
        """
        Parse the LLM response to extract the verification result.
        
        Args:
            response: The LLM response
            
        Returns:
            True if the triplet is verified, False otherwise
        """
        response_lower = response.lower().strip()
        
        # Look for explicit true/false at the beginning
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
        logger.warning(f"Unclear verification response: {response}")
        return False
    
    def get_verification_explanation(self, 
                                   triplet: Triplet, 
                                   evidence_graph: EvidenceGraph, 
                                   evidence_texts: List[str]) -> Dict[str, Any]:
        """
        Get detailed explanation for triplet verification.
        
        Args:
            triplet: The triplet to explain
            evidence_graph: The evidence graph
            evidence_texts: Original evidence texts
            
        Returns:
            Dictionary containing verification explanation
        """
        relevant_triplets = self._get_relevant_evidence_triplets(triplet, evidence_graph)
        result = self.verify_triplet(triplet, evidence_graph, evidence_texts)
        
        return {
            "triplet": str(triplet),
            "verification_result": result,
            "relevant_evidence_triplets": [str(t) for t in relevant_triplets],
            "evidence_texts": evidence_texts,
            "matching_strategy": "entity_and_predicate_matching"
        }
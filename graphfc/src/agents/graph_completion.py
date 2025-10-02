"""
Graph completion agent for resolving unknown entities in triplets.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from src.models.graph import Triplet, EvidenceGraph, Entity, EntityType
from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


class GraphCompletionAgent:
    """
    Agent responsible for completing triplets that contain unknown entities
    by finding the appropriate known entities from the evidence.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the graph completion agent.
        
        Args:
            llm_client: LLM client for generating responses
        """
        self.llm_client = llm_client
    
    def complete_triplet(self, 
                        triplet: Triplet, 
                        evidence_graph: EvidenceGraph, 
                        evidence_texts: List[str]) -> Tuple[Optional[Entity], bool]:
        """
        Complete a triplet by resolving its unknown entity.
        
        Args:
            triplet: The triplet with exactly one unknown entity
            evidence_graph: The evidence graph to search for completions
            evidence_texts: Original evidence texts for additional context
            
        Returns:
            Tuple of (resolved_entity, success_flag)
        """
        unknown_count = triplet.unknown_entity_count()
        if unknown_count != 1:
            raise ValueError(f"Graph completion agent can only handle triplets with exactly 1 unknown entity, got {unknown_count}")
        
        # Identify the unknown entity and known entity
        if triplet.subject.is_unknown:
            unknown_entity = triplet.subject
            known_entity = triplet.object
            position = "subject"
        else:
            unknown_entity = triplet.object
            known_entity = triplet.subject
            position = "object"
        
        logger.info(f"Completing unknown entity {unknown_entity.name} in position {position} for triplet: {triplet}")
        
        # Get relevant triplets from evidence graph
        relevant_triplets = self._get_relevant_evidence_triplets(triplet, evidence_graph)
        
        # Create prompt for completion
        prompt = self._create_completion_prompt(triplet, relevant_triplets, evidence_texts, position)
        
        # Get response from LLM
        response = self.llm_client.generate(prompt)
        
        # Parse response to get the resolved entity
        resolved_entity = self._parse_completion_response(response, evidence_graph)
        
        success = resolved_entity is not None
        
        if success:
            logger.info(f"Successfully resolved {unknown_entity.name} to {resolved_entity.name}")
        else:
            logger.warning(f"Failed to resolve unknown entity {unknown_entity.name}")
        
        return resolved_entity, success
    
    def batch_complete_triplets(self, 
                              triplets: List[Triplet], 
                              evidence_graph: EvidenceGraph, 
                              evidence_texts: List[str]) -> List[Tuple[Optional[Entity], bool]]:
        """
        Complete multiple triplets in batch.
        
        Args:
            triplets: List of triplets to complete
            evidence_graph: The evidence graph to search
            evidence_texts: Original evidence texts
            
        Returns:
            List of (resolved_entity, success_flag) tuples
        """
        results = []
        
        for triplet in triplets:
            try:
                resolved_entity, success = self.complete_triplet(triplet, evidence_graph, evidence_texts)
                results.append((resolved_entity, success))
            except Exception as e:
                logger.error(f"Error completing triplet {triplet}: {e}")
                results.append((None, False))
        
        return results
    
    def _get_relevant_evidence_triplets(self, 
                                      target_triplet: Triplet, 
                                      evidence_graph: EvidenceGraph) -> List[Triplet]:
        """
        Get evidence triplets that could help resolve the unknown entity.
        
        Args:
            target_triplet: The triplet with unknown entity
            evidence_graph: The evidence graph to search
            
        Returns:
            List of relevant evidence triplets
        """
        relevant_triplets = []
        
        # Get the known entity from the target triplet
        known_entity = target_triplet.subject if not target_triplet.subject.is_unknown else target_triplet.object
        
        # Find evidence triplets that involve the known entity
        for evidence_triplet in evidence_graph.triplets:
            if self._entities_match(known_entity, evidence_triplet.subject) or \
               self._entities_match(known_entity, evidence_triplet.object):
                relevant_triplets.append(evidence_triplet)
        
        # Also look for triplets with similar predicates
        for evidence_triplet in evidence_graph.triplets:
            if self._predicates_match(target_triplet.predicate, evidence_triplet.predicate):
                relevant_triplets.append(evidence_triplet)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_triplets = []
        for triplet in relevant_triplets:
            triplet_key = (triplet.subject.name, triplet.predicate, triplet.object.name)
            if triplet_key not in seen:
                seen.add(triplet_key)
                unique_triplets.append(triplet)
        
        return unique_triplets
    
    def _entities_match(self, entity1: Entity, entity2: Entity) -> bool:
        """
        Check if two entities match (with flexibility for name variations).
        
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
        
        # Check if one name is contained in the other
        name1_lower = entity1.name.lower()
        name2_lower = entity2.name.lower()
        
        if (name1_lower in name2_lower and len(name1_lower) > 3) or \
           (name2_lower in name1_lower and len(name2_lower) > 3):
            return True
        
        return False
    
    def _predicates_match(self, predicate1: str, predicate2: str) -> bool:
        """
        Check if two predicates match (with flexibility for variations).
        
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
    
    def _create_completion_prompt(self, 
                                triplet: Triplet, 
                                relevant_triplets: List[Triplet], 
                                evidence_texts: List[str],
                                unknown_position: str) -> str:
        """
        Create prompt for entity completion.
        
        Args:
            triplet: The triplet with unknown entity
            relevant_triplets: Relevant evidence triplets
            evidence_texts: Original evidence texts
            unknown_position: Position of unknown entity ("subject" or "object")
            
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
        
        # Get the unknown entity and known entity info
        if unknown_position == "subject":
            unknown_entity = triplet.subject
            known_entity = triplet.object
            triplet_str = f"<{unknown_entity.name}, {triplet.predicate}, {known_entity.name}>"
        else:
            unknown_entity = triplet.object
            known_entity = triplet.subject
            triplet_str = f"<{known_entity.name}, {triplet.predicate}, {unknown_entity.name}>"
        
        prompt = f"""Evidence:
{evidence_str}

Relevant Evidence Triplets:
{triplets_str}

Complete the unknown entity in the triplet based on the above evidence. The unknown entity is marked as "{unknown_entity.name}".

Triplet to complete: {triplet_str}

Instructions:
1. Look through the evidence texts and relevant triplets to find what entity should replace "{unknown_entity.name}"
2. The entity should be a specific, concrete entity mentioned in the evidence
3. Make sure the completed triplet makes sense and is supported by the evidence
4. If you cannot find a suitable entity, output "none"
5. Only output the entity name, nothing else

Example:
If the triplet is <x_1, founded, Harvard University> and the evidence says "Harvard University was founded by John Harvard", then output: John Harvard

Answer:"""
        
        return prompt
    
    def _parse_completion_response(self, response: str, evidence_graph: EvidenceGraph) -> Optional[Entity]:
        """
        Parse the LLM response to extract the resolved entity.
        
        Args:
            response: The LLM response
            evidence_graph: Evidence graph to find entity types
            
        Returns:
            Resolved Entity object or None if not found
        """
        response = response.strip()
        
        # Check for explicit "none" response
        if response.lower() in ["none", "null", "n/a", "not found", ""]:
            return None
        
        # Clean up the response (remove quotes, extra whitespace, etc.)
        entity_name = response.strip('\'"').strip()
        
        # Skip if empty after cleaning
        if not entity_name:
            return None
        
        # Try to find the entity in the evidence graph to get its type
        for evidence_triplet in evidence_graph.triplets:
            if self._entities_match(Entity(entity_name, EntityType.CONCEPT), evidence_triplet.subject):
                return Entity(entity_name, evidence_triplet.subject.entity_type, is_unknown=False)
            if self._entities_match(Entity(entity_name, EntityType.CONCEPT), evidence_triplet.object):
                return Entity(entity_name, evidence_triplet.object.entity_type, is_unknown=False)
        
        # If not found in evidence graph, create with default type
        entity_type = self._infer_entity_type(entity_name)
        return Entity(entity_name, entity_type, is_unknown=False)
    
    def _infer_entity_type(self, entity_name: str) -> EntityType:
        """
        Infer entity type based on the entity name.
        
        Args:
            entity_name: The name of the entity
            
        Returns:
            Inferred EntityType
        """
        name_lower = entity_name.lower()
        
        # Simple heuristics for entity type inference
        if any(keyword in name_lower for keyword in ["university", "college", "school", "institute", "company", "corporation"]):
            return EntityType.ORGANIZATION
        elif any(keyword in name_lower for keyword in ["city", "town", "country", "state", "province", "street", "avenue"]):
            return EntityType.LOCATION
        elif re.match(r'^\d{4}$', entity_name) or any(keyword in name_lower for keyword in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
            return EntityType.TIME
        elif entity_name.isdigit():
            return EntityType.NUMBER
        elif entity_name[0].isupper() and ' ' in entity_name:  # Likely a person name
            return EntityType.PERSON
        else:
            return EntityType.CONCEPT
    
    def get_completion_explanation(self, 
                                 triplet: Triplet, 
                                 evidence_graph: EvidenceGraph, 
                                 evidence_texts: List[str]) -> Dict[str, Any]:
        """
        Get detailed explanation for triplet completion.
        
        Args:
            triplet: The triplet to explain
            evidence_graph: The evidence graph
            evidence_texts: Original evidence texts
            
        Returns:
            Dictionary containing completion explanation
        """
        resolved_entity, success = self.complete_triplet(triplet, evidence_graph, evidence_texts)
        relevant_triplets = self._get_relevant_evidence_triplets(triplet, evidence_graph)
        
        unknown_entity = triplet.subject if triplet.subject.is_unknown else triplet.object
        
        return {
            "original_triplet": str(triplet),
            "unknown_entity": unknown_entity.name,
            "resolved_entity": resolved_entity.name if resolved_entity else None,
            "completion_success": success,
            "relevant_evidence_triplets": [str(t) for t in relevant_triplets],
            "evidence_texts": evidence_texts,
            "completion_strategy": "evidence_based_entity_resolution"
        }
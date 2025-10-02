"""
Graph-guided planning module for optimal triplet verification order.
"""

from typing import List, Tuple
from src.models.graph import Graph, Triplet, ClaimGraph
import random


class GraphGuidedPlanner:
    """
    Handles graph-guided planning by sorting triplets based on unknown entity count.
    
    The planning strategy prioritizes triplets based on the following order:
    1. Triplets with two known entities (priority 0) - can be verified directly
    2. Triplets with one unknown entity (priority 1) - require completion
    3. Triplets with two unknown entities (priority 2) - require multiple completions
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the planner.
        
        Args:
            random_seed: Random seed for consistent ordering within same priority levels
        """
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def plan_verification_order(self, claim_graph: ClaimGraph) -> List[Triplet]:
        """
        Plan the optimal verification order for triplets in the claim graph.
        
        Args:
            claim_graph: The claim graph containing triplets to be ordered
            
        Returns:
            List of triplets sorted by verification priority
        """
        triplets = claim_graph.triplets.copy()
        
        # Calculate priority for each triplet
        triplet_priorities = []
        for triplet in triplets:
            priority = self._calculate_priority(triplet)
            triplet_priorities.append((triplet, priority))
        
        # Sort by priority (ascending), with random ordering within same priority
        triplet_priorities.sort(key=lambda x: (x[1], random.random()))
        
        # Extract sorted triplets
        sorted_triplets = [triplet for triplet, _ in triplet_priorities]
        
        return sorted_triplets
    
    def _calculate_priority(self, triplet: Triplet) -> int:
        """
        Calculate verification priority for a triplet based on unknown entity count.
        
        Args:
            triplet: The triplet to calculate priority for
            
        Returns:
            Priority value (0=highest, 2=lowest)
        """
        unknown_count = triplet.unknown_entity_count()
        
        if unknown_count == 0:
            return 0  # Two known entities - verify directly
        elif unknown_count == 1:
            return 1  # One unknown entity - complete then verify
        else:
            return 2  # Two unknown entities - complete multiple times
    
    def get_planning_statistics(self, claim_graph: ClaimGraph) -> dict:
        """
        Get statistics about the planning for a claim graph.
        
        Args:
            claim_graph: The claim graph to analyze
            
        Returns:
            Dictionary containing planning statistics
        """
        triplets = claim_graph.triplets
        
        priority_0_count = 0  # Two known entities
        priority_1_count = 0  # One unknown entity
        priority_2_count = 0  # Two unknown entities
        
        for triplet in triplets:
            priority = self._calculate_priority(triplet)
            if priority == 0:
                priority_0_count += 1
            elif priority == 1:
                priority_1_count += 1
            else:
                priority_2_count += 1
        
        total_triplets = len(triplets)
        
        return {
            "total_triplets": total_triplets,
            "priority_0_count": priority_0_count,
            "priority_1_count": priority_1_count,
            "priority_2_count": priority_2_count,
            "priority_0_percentage": priority_0_count / total_triplets * 100 if total_triplets > 0 else 0,
            "priority_1_percentage": priority_1_count / total_triplets * 100 if total_triplets > 0 else 0,
            "priority_2_percentage": priority_2_count / total_triplets * 100 if total_triplets > 0 else 0,
            "unknown_entities": len(claim_graph.get_unknown_entities()),
            "known_entities": len(claim_graph.get_known_entities())
        }
    
    def explain_planning_decision(self, triplet: Triplet) -> str:
        """
        Provide explanation for why a triplet has certain priority.
        
        Args:
            triplet: The triplet to explain
            
        Returns:
            Human-readable explanation of the planning decision
        """
        priority = self._calculate_priority(triplet)
        unknown_count = triplet.unknown_entity_count()
        
        if priority == 0:
            return f"Priority 0: Both entities are known - can verify directly against evidence"
        elif priority == 1:
            unknown_entity = triplet.subject if triplet.subject.is_unknown else triplet.object
            known_entity = triplet.object if triplet.subject.is_unknown else triplet.subject
            return (f"Priority 1: One unknown entity ({unknown_entity.name}) and one known entity "
                   f"({known_entity.name}) - complete unknown entity first, then verify")
        else:
            return (f"Priority 2: Both entities are unknown ({triplet.subject.name}, {triplet.object.name}) "
                   f"- requires multiple completion steps before verification")
    
    def validate_planning_order(self, sorted_triplets: List[Triplet]) -> bool:
        """
        Validate that the planning order follows the correct priority rules.
        
        Args:
            sorted_triplets: List of triplets in planned order
            
        Returns:
            True if the order is valid, False otherwise
        """
        if not sorted_triplets:
            return True
        
        previous_priority = -1
        
        for triplet in sorted_triplets:
            current_priority = self._calculate_priority(triplet)
            
            # Priority should be non-decreasing
            if current_priority < previous_priority:
                return False
            
            previous_priority = current_priority
        
        return True


def create_verification_plan(claim_graph: ClaimGraph, random_seed: int = 42) -> Tuple[List[Triplet], dict]:
    """
    Convenience function to create a verification plan for a claim graph.
    
    Args:
        claim_graph: The claim graph to plan verification for
        random_seed: Random seed for consistent ordering
        
    Returns:
        Tuple of (sorted_triplets, planning_statistics)
    """
    planner = GraphGuidedPlanner(random_seed)
    sorted_triplets = planner.plan_verification_order(claim_graph)
    statistics = planner.get_planning_statistics(claim_graph)
    
    return sorted_triplets, statistics
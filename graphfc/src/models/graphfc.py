"""
Main GraphFC framework that orchestrates all components for end-to-end fact-checking.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from src.models.graph import ClaimGraph, EvidenceGraph, Triplet, Entity
from src.models.planning import GraphGuidedPlanner
from src.agents.llm_client import LLMClient
from src.agents.graph_construction import GraphConstructionAgent
from src.agents.graph_match import GraphMatchAgent
from src.agents.graph_completion import GraphCompletionAgent

logger = logging.getLogger(__name__)


@dataclass
class FactCheckingResult:
    """Result of fact-checking operation."""
    label: str  # "True" or "False"
    confidence: float
    claim_graph: ClaimGraph
    evidence_graph: EvidenceGraph
    verification_steps: List[Dict[str, Any]]
    statistics: Dict[str, Any]


class GraphFC:
    """
    Main GraphFC framework for graph-based fact-checking.
    
    The framework consists of three main components:
    1. Graph Construction: Convert claims and evidence to graph structures
    2. Graph-Guided Planning: Determine optimal verification order
    3. Graph-Guided Checking: Verify triplets using match and completion agents
    """
    
    def __init__(self, 
                 llm_client: LLMClient,
                 k_shot_examples: int = 10,
                 early_stop: bool = True,
                 random_seed: int = 42):
        """
        Initialize the GraphFC framework.
        
        Args:
            llm_client: LLM client for all agents
            k_shot_examples: Number of in-context examples for graph construction
            early_stop: Whether to stop early when a triplet fails verification
            random_seed: Random seed for consistent planning
        """
        self.llm_client = llm_client
        self.early_stop = early_stop
        
        # Initialize agents
        self.graph_construction_agent = GraphConstructionAgent(llm_client, k_shot_examples)
        self.graph_match_agent = GraphMatchAgent(llm_client)
        self.graph_completion_agent = GraphCompletionAgent(llm_client)
        self.planner = GraphGuidedPlanner(random_seed)
        
        logger.info("Initialized GraphFC framework")
    
    def fact_check(self, 
                   claim: str, 
                   evidence: List[str]) -> FactCheckingResult:
        """
        Perform end-to-end fact-checking on a claim with evidence.
        
        Args:
            claim: The natural language claim to verify
            evidence: List of evidence passages
            
        Returns:
            FactCheckingResult containing the verification result and details
        """
        logger.info(f"Starting fact-checking for claim: {claim}")
        
        verification_steps = []
        
        # Step 1: Graph Construction
        logger.info("Step 1: Graph Construction")
        claim_graph = self.graph_construction_agent.construct_claim_graph(claim)
        
        known_entities = list(claim_graph.get_known_entities())
        evidence_graph = self.graph_construction_agent.construct_evidence_graph(evidence, known_entities)
        
        construction_stats = self.graph_construction_agent.validate_graph_construction(
            claim_graph, evidence_graph
        )
        
        verification_steps.append({
            "step": "graph_construction",
            "claim_triplets": len(claim_graph.triplets),
            "evidence_triplets": len(evidence_graph.triplets),
            "unknown_entities": len(claim_graph.get_unknown_entities()),
            "statistics": construction_stats
        })
        
        # Step 2: Graph-Guided Planning
        logger.info("Step 2: Graph-Guided Planning")
        sorted_triplets = self.planner.plan_verification_order(claim_graph)
        planning_stats = self.planner.get_planning_statistics(claim_graph)
        
        verification_steps.append({
            "step": "graph_guided_planning",
            "triplet_order": [str(t) for t in sorted_triplets],
            "planning_statistics": planning_stats
        })
        
        # Step 3: Graph-Guided Checking
        logger.info("Step 3: Graph-Guided Checking")
        overall_result = True
        triplet_results = []
        
        for i, triplet in enumerate(sorted_triplets):
            logger.info(f"Verifying triplet {i+1}/{len(sorted_triplets)}: {triplet}")
            
            if triplet.unknown_entity_count() == 0:
                # Graph Match: Verify triplet with only known entities
                result = self._verify_known_triplet(triplet, evidence_graph, evidence)
                triplet_results.append({
                    "triplet": str(triplet),
                    "type": "graph_match",
                    "result": result,
                    "unknown_entities_before": 0,
                    "unknown_entities_after": 0
                })
                
                if not result:
                    overall_result = False
                    if self.early_stop:
                        logger.info("Early stopping due to failed verification")
                        break
            
            elif triplet.unknown_entity_count() == 1:
                # Graph Completion: Resolve unknown entity then verify
                resolved_entity, completion_success = self._complete_unknown_triplet(
                    triplet, claim_graph, evidence_graph, evidence
                )
                
                if completion_success:
                    # Update claim graph with resolved entity
                    unknown_entity = triplet.subject if triplet.subject.is_unknown else triplet.object
                    claim_graph.replace_unknown_entity(unknown_entity, resolved_entity)
                    
                    # Update the current triplet
                    updated_triplet = triplet.replace_unknown_entity(unknown_entity, resolved_entity)
                    
                    # Verify the updated triplet
                    verification_result = self._verify_known_triplet(updated_triplet, evidence_graph, evidence)
                    
                    triplet_results.append({
                        "triplet": str(triplet),
                        "type": "graph_completion",
                        "result": verification_result,
                        "unknown_entities_before": 1,
                        "unknown_entities_after": 0,
                        "resolved_entity": resolved_entity.name,
                        "updated_triplet": str(updated_triplet)
                    })
                    
                    if not verification_result:
                        overall_result = False
                        if self.early_stop:
                            logger.info("Early stopping due to failed verification after completion")
                            break
                else:
                    # Completion failed
                    triplet_results.append({
                        "triplet": str(triplet),
                        "type": "graph_completion",
                        "result": False,
                        "unknown_entities_before": 1,
                        "unknown_entities_after": 1,
                        "resolved_entity": None,
                        "error": "Failed to resolve unknown entity"
                    })
                    
                    overall_result = False
                    if self.early_stop:
                        logger.info("Early stopping due to failed completion")
                        break
            
            else:
                # Triplet with 2 unknown entities - should be handled after other completions
                triplet_results.append({
                    "triplet": str(triplet),
                    "type": "deferred",
                    "result": False,
                    "unknown_entities_before": 2,
                    "unknown_entities_after": 2,
                    "error": "Multiple unknown entities - requires prior completions"
                })
                
                overall_result = False
                if self.early_stop:
                    logger.info("Early stopping due to unresolved multiple unknown entities")
                    break
        
        verification_steps.append({
            "step": "graph_guided_checking",
            "triplet_results": triplet_results,
            "overall_result": overall_result
        })
        
        # Calculate confidence based on verification results
        confidence = self._calculate_confidence(triplet_results)
        
        # Compile statistics
        final_stats = {
            "total_triplets": len(claim_graph.triplets),
            "verified_triplets": sum(1 for r in triplet_results if r["result"]),
            "failed_triplets": sum(1 for r in triplet_results if not r["result"]),
            "completion_operations": sum(1 for r in triplet_results if r["type"] == "graph_completion"),
            "match_operations": sum(1 for r in triplet_results if r["type"] == "graph_match"),
            "final_unknown_entities": len(claim_graph.get_unknown_entities()),
            "confidence": confidence
        }
        
        label = "True" if overall_result else "False"
        logger.info(f"Fact-checking completed. Result: {label} (confidence: {confidence:.2f})")
        
        return FactCheckingResult(
            label=label,
            confidence=confidence,
            claim_graph=claim_graph,
            evidence_graph=evidence_graph,
            verification_steps=verification_steps,
            statistics=final_stats
        )
    
    def _verify_known_triplet(self, 
                            triplet: Triplet, 
                            evidence_graph: EvidenceGraph, 
                            evidence_texts: List[str]) -> bool:
        """
        Verify a triplet with only known entities using the graph match agent.
        
        Args:
            triplet: Triplet with only known entities
            evidence_graph: Evidence graph to match against
            evidence_texts: Original evidence texts
            
        Returns:
            True if triplet is verified, False otherwise
        """
        try:
            return self.graph_match_agent.verify_triplet(triplet, evidence_graph, evidence_texts)
        except Exception as e:
            logger.error(f"Error verifying triplet {triplet}: {e}")
            return False
    
    def _complete_unknown_triplet(self, 
                                triplet: Triplet, 
                                claim_graph: ClaimGraph, 
                                evidence_graph: EvidenceGraph, 
                                evidence_texts: List[str]) -> Tuple[Optional[Entity], bool]:
        """
        Complete a triplet with unknown entities using the graph completion agent.
        
        Args:
            triplet: Triplet with exactly one unknown entity
            claim_graph: Current claim graph
            evidence_graph: Evidence graph to search
            evidence_texts: Original evidence texts
            
        Returns:
            Tuple of (resolved_entity, success_flag)
        """
        try:
            return self.graph_completion_agent.complete_triplet(triplet, evidence_graph, evidence_texts)
        except Exception as e:
            logger.error(f"Error completing triplet {triplet}: {e}")
            return None, False
    
    def _calculate_confidence(self, triplet_results: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on triplet verification results.
        
        Args:
            triplet_results: List of triplet verification results
            
        Returns:
            Confidence score between 0 and 1
        """
        if not triplet_results:
            return 0.0
        
        successful_verifications = sum(1 for r in triplet_results if r["result"])
        total_triplets = len(triplet_results)
        
        # Base confidence is the proportion of successful verifications
        base_confidence = successful_verifications / total_triplets
        
        # Adjust confidence based on completion operations
        completion_operations = sum(1 for r in triplet_results if r["type"] == "graph_completion")
        if completion_operations > 0:
            # Reduce confidence for each completion operation (introduces uncertainty)
            completion_penalty = 0.1 * completion_operations / total_triplets
            base_confidence = max(0.0, base_confidence - completion_penalty)
        
        return base_confidence
    
    def batch_fact_check(self, 
                        claims_and_evidence: List[Tuple[str, List[str]]]) -> List[FactCheckingResult]:
        """
        Perform fact-checking on multiple claims in batch.
        
        Args:
            claims_and_evidence: List of (claim, evidence_list) tuples
            
        Returns:
            List of FactCheckingResult objects
        """
        results = []
        
        for i, (claim, evidence) in enumerate(claims_and_evidence):
            logger.info(f"Processing claim {i+1}/{len(claims_and_evidence)}")
            try:
                result = self.fact_check(claim, evidence)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing claim {i+1}: {e}")
                # Create a failure result
                results.append(FactCheckingResult(
                    label="False",
                    confidence=0.0,
                    claim_graph=ClaimGraph(claim, []),
                    evidence_graph=EvidenceGraph(evidence, []),
                    verification_steps=[],
                    statistics={"error": str(e)}
                ))
        
        return results
    
    def get_framework_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the GraphFC framework configuration.
        
        Returns:
            Dictionary containing framework statistics
        """
        return {
            "llm_model": self.llm_client.get_model_info(),
            "k_shot_examples": self.graph_construction_agent.k_shot_examples,
            "early_stop": self.early_stop,
            "components": {
                "graph_construction_agent": type(self.graph_construction_agent).__name__,
                "graph_match_agent": type(self.graph_match_agent).__name__,
                "graph_completion_agent": type(self.graph_completion_agent).__name__,
                "planner": type(self.planner).__name__
            }
        }
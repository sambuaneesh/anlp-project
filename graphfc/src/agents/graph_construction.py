"""
Graph construction agent for converting natural language claims and evidence into graph structures.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from src.models.graph import Entity, Triplet, ClaimGraph, EvidenceGraph, EntityType
from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


class GraphConstructionAgent:
    """
    Agent responsible for constructing claim graphs and evidence graphs from natural language text.
    """
    
    def __init__(self, llm_client: LLMClient, k_shot_examples: int = 10):
        """
        Initialize the graph construction agent.
        
        Args:
            llm_client: LLM client for generating responses
            k_shot_examples: Number of in-context examples to use
        """
        self.llm_client = llm_client
        self.k_shot_examples = k_shot_examples
        
        # Define entity types
        self.entity_types = [
            "Person", "Location", "Organization", "Time", 
            "Number", "Concept", "Object", "Work", "Event", "Species"
        ]
        
        # Define constraint types
        self.constraint_types = ["temporal", "spatial", "condition", "context"]
    
    def construct_claim_graph(self, claim_text: str) -> ClaimGraph:
        """
        Construct a claim graph from natural language claim text.
        
        Args:
            claim_text: The natural language claim to convert
            
        Returns:
            ClaimGraph object containing the extracted triplets
        """
        logger.info(f"Constructing claim graph for: {claim_text}")
        
        # Generate prompt for claim graph construction
        prompt = self._create_claim_graph_prompt(claim_text)
        
        # Get response from LLM
        response = self.llm_client.generate(prompt)
        
        # Parse response to extract triplets
        triplets = self._parse_claim_graph_response(response, claim_text)
        
        # Create and return claim graph
        claim_graph = ClaimGraph(claim_text, triplets)
        
        logger.info(f"Constructed claim graph with {len(triplets)} triplets")
        return claim_graph
    
    def construct_evidence_graph(self, evidence_texts: List[str], known_entities: List[Entity]) -> EvidenceGraph:
        """
        Construct an evidence graph from evidence texts and known entities.
        
        Args:
            evidence_texts: List of evidence text passages
            known_entities: List of known entities from the claim graph
            
        Returns:
            EvidenceGraph object containing the extracted triplets
        """
        logger.info(f"Constructing evidence graph from {len(evidence_texts)} evidence passages")
        
        # Combine evidence texts
        combined_evidence = " ".join(evidence_texts)
        
        # Generate prompt for evidence graph construction
        prompt = self._create_evidence_graph_prompt(combined_evidence, known_entities)
        
        # Get response from LLM
        response = self.llm_client.generate(prompt)
        
        # Parse response to extract triplets
        triplets = self._parse_evidence_graph_response(response, evidence_texts)
        
        # Create and return evidence graph
        evidence_graph = EvidenceGraph(evidence_texts, triplets)
        
        logger.info(f"Constructed evidence graph with {len(triplets)} triplets")
        return evidence_graph
    
    def _create_claim_graph_prompt(self, claim_text: str) -> str:
        """Create prompt for claim graph construction."""
        prompt = f"""# Knowledge Graph Construction Specification

You are an expert in knowledge graph construction. Your task is to parse natural language claims into a formal claim graph representation by following these specifications:

## 1. Entity Types
- Person: Real individuals
- Location: Places, cities, regions
- Organization: Companies, institutions, groups 
- Time: Dates, years, periods
- Number: Numerical values
- Concept: Abstract ideas, categories
- Object: Physical items
- Work: Creative works (books, songs, etc.)
- Event: Occurrences, happenings
- Species: Biological organisms

## 2. Constraint Types
- temporal: Time-related constraints (year, date, period)
- spatial: Location-related constraints (in, at, from)
- condition: Qualifying conditions or attributes
- context: Broader situational context

## 3. Atomic Proposition Rules

### Definition
An atomic proposition must:
- Express a single, indivisible fact
- Cannot be broken down into simpler meaningful statements
- Must preserve all relevant context
- Must maintain temporal and spatial relationships

### Decomposition Guidelines
1. Structural Analysis:
   - Split complex sentences at conjunction words (and, but, or)
   - Separate conditional statements (if/then) into distinct propositions
   - Identify dependent clauses and their relationships
   - Preserve modifiers and qualifiers with their related concepts

2. Semantic Preservation:
   - Maintain causal relationships
   - Preserve temporal order
   - Keep spatial relationships intact
   - Retain contextual qualifiers

## 4. Fuzzy Entity Identification Process

### Entity Reference Types
1. Direct Reference:
   - Uses proper name ("John", "Paris")
   - Specific numerical values
   - Well-defined concepts

2. Indirect Reference:
   - Uses descriptions ("the teacher", "that city")
   - Role-based references ("the founder", "the mother")
   - Attribute-based references ("the tall building")

3. Contextual Reference:
   - Requires information from other statements
   - Part of a collective reference
   - Implied entities

### Fuzzy Entity Decision Tree
1. Initial Check:
   - Is the entity referred to by proper name? → Not fuzzy
   - Is the entity a specific number or date? → Not fuzzy
   - Is the entity a well-defined concept? → Not fuzzy

2. Context Analysis:
   - Does the entity require contextual information? → Fuzzy
   - Is the entity part of a group or collection? → Fuzzy
   - Is the entity only described by role or attribute? → Fuzzy
   - Is the entity referenced through relationships? → Fuzzy

3. Coreference Resolution:
   - Track entities across multiple atomic propositions
   - Maintain consistent fuzzy entity IDs (x_1, x_2, etc.)
   - Document relationships between fuzzy entities

## 5. Output Format

Return a JSON object with the following structure:

```json
{{
  "entities": [
    {{
      "name": "entity_name",
      "type": "entity_type",
      "is_unknown": boolean
    }}
  ],
  "triplets": [
    {{
      "atomic_proposition": "the original atomic statement",
      "subject": "entity_name_or_id",
      "predicate": "relationship_type",
      "object": "entity_name_or_id",
      "constraint": {{
        "type": "constraint_type",
        "value": "constraint_value"
      }}
    }}
  ]
}}
```

## Example

Input: "Elizabeth founded St Hugh's College and was the daughter of Christopher."

Output:
```json
{{
  "entities": [
    {{"name": "Elizabeth", "type": "Person", "is_unknown": false}},
    {{"name": "St Hugh's College", "type": "Organization", "is_unknown": false}},
    {{"name": "Christopher", "type": "Person", "is_unknown": false}}
  ],
  "triplets": [
    {{
      "atomic_proposition": "Elizabeth founded St Hugh's College",
      "subject": "Elizabeth",
      "predicate": "founded",
      "object": "St Hugh's College"
    }},
    {{
      "atomic_proposition": "Elizabeth was the daughter of Christopher",
      "subject": "Elizabeth",
      "predicate": "daughter_of",
      "object": "Christopher"
    }}
  ]
}}
```

Now, please parse the following claim into the formal representation:

Claim: "{claim_text}"

Output:"""
        
        return prompt
    
    def _create_evidence_graph_prompt(self, evidence_text: str, known_entities: List[Entity]) -> str:
        """Create prompt for evidence graph construction."""
        
        # Format entity set
        entity_set_str = ""
        for entity in known_entities:
            entity_set_str += f"- {entity.name} ({entity.entity_type.value})\n"
        
        prompt = f"""## Task:  
Extract semantic triples from the given evidence and ensure every extracted triple is context-independent.  

## Input:  
- Evidence: {evidence_text}
- Entities: {entity_set_str}

## Output Format:  
Return a JSON object with the following structure:

```json
{{
  "triplets": [
    {{
      "subject": "entity_name",
      "predicate": "relationship",
      "object": "entity_name"
    }}
  ]
}}
```

## Extraction Guidelines:  
1. Ensure that at least one entity (subject or object) in each triple is from the provided Entity Set. 
2. Identify all relationships from the Evidence Text that meet this requirement.  
3. Every triple should be context-independent. Use full forms or expanded phrases for relational references where necessary.
4. Focus on factual relationships that can be verified.

## Example

Evidence: "St Hugh's College was founded by Elizabeth Wordsworth in 1886. Elizabeth was the daughter of Christopher Wordsworth."
Entity Set:
- Elizabeth (Person)
- St Hugh's College (Organization)
- Christopher (Person)

Output:
```json
{{
  "triplets": [
    {{
      "subject": "St Hugh's College",
      "predicate": "founded_by",
      "object": "Elizabeth Wordsworth"
    }},
    {{
      "subject": "St Hugh's College", 
      "predicate": "founded_in",
      "object": "1886"
    }},
    {{
      "subject": "Elizabeth Wordsworth",
      "predicate": "daughter_of", 
      "object": "Christopher Wordsworth"
    }}
  ]
}}
```

Now extract triples from the given evidence:

Evidence: {evidence_text}

Entity Set: 
{entity_set_str}

Output:"""
        
        return prompt
    
    def _parse_claim_graph_response(self, response: str, claim_text: str) -> List[Triplet]:
        """Parse LLM response to extract triplets for claim graph."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_str = response.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract entities
            entities_dict = {}
            for entity_data in data.get("entities", []):
                entity = Entity(
                    name=entity_data["name"],
                    entity_type=EntityType(entity_data["type"]),
                    is_unknown=entity_data.get("is_unknown", False)
                )
                entities_dict[entity.name] = entity
            
            # Extract triplets
            triplets = []
            for triplet_data in data.get("triplets", []):
                subject_name = triplet_data["subject"]
                object_name = triplet_data["object"]
                
                # Get or create entities
                subject = entities_dict.get(subject_name, Entity(subject_name, EntityType.CONCEPT, is_unknown=subject_name.startswith('x_')))
                object_entity = entities_dict.get(object_name, Entity(object_name, EntityType.CONCEPT, is_unknown=object_name.startswith('x_')))
                
                triplet = Triplet(
                    subject=subject,
                    predicate=triplet_data["predicate"],
                    object=object_entity,
                    atomic_proposition=triplet_data.get("atomic_proposition"),
                    constraint=triplet_data.get("constraint")
                )
                triplets.append(triplet)
            
            return triplets
            
        except Exception as e:
            logger.error(f"Error parsing claim graph response: {e}")
            logger.error(f"Response: {response}")
            return []
    
    def _parse_evidence_graph_response(self, response: str, evidence_texts: List[str]) -> List[Triplet]:
        """Parse LLM response to extract triplets for evidence graph."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_str = response.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract triplets
            triplets = []
            for triplet_data in data.get("triplets", []):
                subject = Entity(triplet_data["subject"], EntityType.CONCEPT)
                object_entity = Entity(triplet_data["object"], EntityType.CONCEPT)
                
                triplet = Triplet(
                    subject=subject,
                    predicate=triplet_data["predicate"],
                    object=object_entity
                )
                triplets.append(triplet)
            
            return triplets
            
        except Exception as e:
            logger.error(f"Error parsing evidence graph response: {e}")
            logger.error(f"Response: {response}")
            return []
    
    def validate_graph_construction(self, claim_graph: ClaimGraph, evidence_graph: EvidenceGraph) -> Dict[str, Any]:
        """
        Validate the constructed graphs and return statistics.
        
        Args:
            claim_graph: The constructed claim graph
            evidence_graph: The constructed evidence graph
            
        Returns:
            Dictionary containing validation statistics
        """
        return {
            "claim_graph": {
                "total_triplets": len(claim_graph.triplets),
                "total_entities": len(claim_graph.entities),
                "unknown_entities": len(claim_graph.get_unknown_entities()),
                "known_entities": len(claim_graph.get_known_entities())
            },
            "evidence_graph": {
                "total_triplets": len(evidence_graph.triplets),
                "total_entities": len(evidence_graph.entities),
                "unknown_entities": len(evidence_graph.get_unknown_entities()),
                "known_entities": len(evidence_graph.get_known_entities())
            }
        }
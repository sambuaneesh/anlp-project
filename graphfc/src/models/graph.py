"""
Core data structures for GraphFC: Entity, Triplet, and Graph classes.
"""

from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class EntityType(Enum):
    """Enumeration of entity types used in GraphFC."""
    PERSON = "Person"
    LOCATION = "Location"
    ORGANIZATION = "Organization"
    TIME = "Time"
    NUMBER = "Number"
    CONCEPT = "Concept"
    OBJECT = "Object"
    WORK = "Work"
    EVENT = "Event"
    SPECIES = "Species"


@dataclass
class Entity:
    """Represents an entity in the graph."""
    name: str
    entity_type: EntityType
    is_unknown: bool = False
    description: Optional[str] = None
    
    def __hash__(self):
        return hash((self.name, self.entity_type.value, self.is_unknown))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.name == other.name and 
                self.entity_type == other.entity_type and 
                self.is_unknown == other.is_unknown)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "is_unknown": self.is_unknown,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary representation."""
        return cls(
            name=data["name"],
            entity_type=EntityType(data["type"]),
            is_unknown=data.get("is_unknown", False),
            description=data.get("description")
        )


@dataclass
class Triplet:
    """Represents a triplet (subject, predicate, object) in the graph."""
    subject: Union[Entity, str]
    predicate: str
    object: Union[Entity, str]
    atomic_proposition: Optional[str] = None
    constraint: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Convert string entities to Entity objects if needed."""
        if isinstance(self.subject, str):
            self.subject = Entity(self.subject, EntityType.CONCEPT, is_unknown=self.subject.startswith('x_'))
        if isinstance(self.object, str):
            self.object = Entity(self.object, EntityType.CONCEPT, is_unknown=self.object.startswith('x_'))
    
    def has_unknown_entities(self) -> bool:
        """Check if triplet contains unknown entities."""
        return self.subject.is_unknown or self.object.is_unknown
    
    def unknown_entity_count(self) -> int:
        """Count the number of unknown entities in the triplet."""
        count = 0
        if self.subject.is_unknown:
            count += 1
        if self.object.is_unknown:
            count += 1
        return count
    
    def get_priority(self) -> int:
        """Get verification priority based on unknown entity count."""
        unknown_count = self.unknown_entity_count()
        if unknown_count == 0:
            return 0  # Highest priority - can be verified directly
        elif unknown_count == 1:
            return 1  # Medium priority - requires completion
        else:
            return 2  # Lowest priority - requires multiple completions
    
    def replace_unknown_entity(self, unknown_entity: Entity, known_entity: Entity) -> "Triplet":
        """Replace an unknown entity with a known entity."""
        new_subject = known_entity if self.subject == unknown_entity else self.subject
        new_object = known_entity if self.object == unknown_entity else self.object
        
        return Triplet(
            subject=new_subject,
            predicate=self.predicate,
            object=new_object,
            atomic_proposition=self.atomic_proposition,
            constraint=self.constraint
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triplet to dictionary representation."""
        return {
            "subject": self.subject.to_dict(),
            "predicate": self.predicate,
            "object": self.object.to_dict(),
            "atomic_proposition": self.atomic_proposition,
            "constraint": self.constraint
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Triplet":
        """Create triplet from dictionary representation."""
        return cls(
            subject=Entity.from_dict(data["subject"]),
            predicate=data["predicate"],
            object=Entity.from_dict(data["object"]),
            atomic_proposition=data.get("atomic_proposition"),
            constraint=data.get("constraint")
        )
    
    def __str__(self) -> str:
        """String representation of the triplet."""
        return f"<{self.subject.name}, {self.predicate}, {self.object.name}>"
    
    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))
    
    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        return (self.subject == other.subject and 
                self.predicate == other.predicate and 
                self.object == other.object)


class Graph:
    """Represents a graph structure with entities and triplets."""
    
    def __init__(self, triplets: Optional[List[Triplet]] = None):
        self.triplets: List[Triplet] = triplets or []
        self.entities: Set[Entity] = set()
        self._update_entities()
    
    def _update_entities(self):
        """Update the entity set based on current triplets."""
        self.entities.clear()
        for triplet in self.triplets:
            self.entities.add(triplet.subject)
            self.entities.add(triplet.object)
    
    def add_triplet(self, triplet: Triplet):
        """Add a triplet to the graph."""
        self.triplets.append(triplet)
        self.entities.add(triplet.subject)
        self.entities.add(triplet.object)
    
    def remove_triplet(self, triplet: Triplet):
        """Remove a triplet from the graph."""
        if triplet in self.triplets:
            self.triplets.remove(triplet)
            self._update_entities()
    
    def get_known_entities(self) -> Set[Entity]:
        """Get all known entities in the graph."""
        return {entity for entity in self.entities if not entity.is_unknown}
    
    def get_unknown_entities(self) -> Set[Entity]:
        """Get all unknown entities in the graph."""
        return {entity for entity in self.entities if entity.is_unknown}
    
    def get_triplets_with_entity(self, entity: Entity) -> List[Triplet]:
        """Get all triplets containing the specified entity."""
        return [t for t in self.triplets if t.subject == entity or t.object == entity]
    
    def get_triplets_for_verification(self, target_triplet: Triplet) -> List[Triplet]:
        """Get relevant triplets for verifying a target triplet."""
        relevant_triplets = []
        
        # Get triplets that share entities with the target triplet
        for triplet in self.triplets:
            if (triplet.subject == target_triplet.subject or 
                triplet.object == target_triplet.subject or
                triplet.subject == target_triplet.object or 
                triplet.object == target_triplet.object):
                relevant_triplets.append(triplet)
        
        return relevant_triplets
    
    def replace_unknown_entity(self, unknown_entity: Entity, known_entity: Entity):
        """Replace an unknown entity with a known entity throughout the graph."""
        updated_triplets = []
        
        for triplet in self.triplets:
            updated_triplet = triplet.replace_unknown_entity(unknown_entity, known_entity)
            updated_triplets.append(updated_triplet)
        
        self.triplets = updated_triplets
        self._update_entities()
    
    def get_sorted_triplets_by_priority(self) -> List[Triplet]:
        """Get triplets sorted by verification priority."""
        return sorted(self.triplets, key=lambda t: t.get_priority())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "triplets": [triplet.to_dict() for triplet in self.triplets],
            "entities": [entity.to_dict() for entity in self.entities]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Graph":
        """Create graph from dictionary representation."""
        triplets = [Triplet.from_dict(t_data) for t_data in data["triplets"]]
        return cls(triplets)
    
    def save_to_file(self, filepath: str):
        """Save graph to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "Graph":
        """Load graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __len__(self) -> int:
        """Return the number of triplets in the graph."""
        return len(self.triplets)
    
    def __str__(self) -> str:
        """String representation of the graph."""
        triplet_strs = [str(triplet) for triplet in self.triplets]
        return f"Graph with {len(self.triplets)} triplets:\n" + "\n".join(triplet_strs)
    
    def __iter__(self):
        """Make graph iterable over triplets."""
        return iter(self.triplets)


class ClaimGraph(Graph):
    """Specialized graph for representing claims."""
    
    def __init__(self, claim_text: str, triplets: Optional[List[Triplet]] = None):
        super().__init__(triplets)
        self.claim_text = claim_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claim graph to dictionary representation."""
        data = super().to_dict()
        data["claim_text"] = self.claim_text
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimGraph":
        """Create claim graph from dictionary representation."""
        triplets = [Triplet.from_dict(t_data) for t_data in data["triplets"]]
        return cls(data["claim_text"], triplets)


class EvidenceGraph(Graph):
    """Specialized graph for representing evidence."""
    
    def __init__(self, evidence_texts: List[str], triplets: Optional[List[Triplet]] = None):
        super().__init__(triplets)
        self.evidence_texts = evidence_texts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence graph to dictionary representation."""
        data = super().to_dict()
        data["evidence_texts"] = self.evidence_texts
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceGraph":
        """Create evidence graph from dictionary representation."""
        triplets = [Triplet.from_dict(t_data) for t_data in data["triplets"]]
        return cls(data["evidence_texts"], triplets)
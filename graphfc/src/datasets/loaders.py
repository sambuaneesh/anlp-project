"""
Dataset loaders for HOVER, FEVEROUS, and SciFact datasets.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FactCheckingExample:
    """Represents a single fact-checking example."""
    id: str
    claim: str
    evidence: List[str]
    label: str
    metadata: Dict[str, Any] = None


class BaseDatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    def load_examples(self) -> List[FactCheckingExample]:
        """Load examples from the dataset."""
        raise NotImplementedError
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        raise NotImplementedError


class HOVERDatasetLoader(BaseDatasetLoader):
    """Loader for HOVER dataset."""
    
    def load_examples(self) -> List[FactCheckingExample]:
        """Load HOVER examples."""
        examples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Extract claim and evidence
                claim = data['claim']
                
                # HOVER evidence is in 'supporting_facts' format
                evidence_texts = []
                if 'supporting_facts' in data:
                    for fact in data['supporting_facts']:
                        if isinstance(fact, list) and len(fact) >= 2:
                            # fact[0] is title, fact[1] is sentence
                            evidence_texts.append(fact[1])
                        elif isinstance(fact, str):
                            evidence_texts.append(fact)
                
                # If no supporting facts, use all evidence
                if not evidence_texts and 'evidence' in data:
                    evidence_texts = data['evidence']
                
                # Map HOVER labels to standard format
                label_mapping = {
                    'SUPPORTS': 'True',
                    'REFUTES': 'False',
                    'NOT_ENOUGH_INFO': 'False'
                }
                label = label_mapping.get(data.get('label', 'NOT_ENOUGH_INFO'), 'False')
                
                # Extract hop information if available
                metadata = {
                    'hops': data.get('num_hops', 2),
                    'dataset': 'hover',
                    'original_label': data.get('label')
                }
                
                example = FactCheckingExample(
                    id=data.get('id', str(len(examples))),
                    claim=claim,
                    evidence=evidence_texts,
                    label=label,
                    metadata=metadata
                )
                examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from HOVER dataset")
        return examples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get HOVER dataset information."""
        examples = self.load_examples()
        
        hop_distribution = {}
        for example in examples:
            hops = example.metadata.get('hops', 'unknown')
            hop_distribution[hops] = hop_distribution.get(hops, 0) + 1
        
        return {
            'name': 'HOVER',
            'total_examples': len(examples),
            'hop_distribution': hop_distribution,
            'label_distribution': self._get_label_distribution(examples)
        }
    
    def _get_label_distribution(self, examples: List[FactCheckingExample]) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        distribution = {}
        for example in examples:
            distribution[example.label] = distribution.get(example.label, 0) + 1
        return distribution


class FEVEROUSDatasetLoader(BaseDatasetLoader):
    """Loader for FEVEROUS dataset."""
    
    def load_examples(self) -> List[FactCheckingExample]:
        """Load FEVEROUS examples."""
        examples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                claim = data['claim']
                
                # Extract evidence texts
                evidence_texts = []
                if 'evidence' in data:
                    for evidence_group in data['evidence']:
                        for evidence_item in evidence_group:
                            if 'content' in evidence_item:
                                evidence_texts.append(evidence_item['content'])
                            elif isinstance(evidence_item, str):
                                evidence_texts.append(evidence_item)
                
                # Map FEVEROUS labels
                label_mapping = {
                    'SUPPORTS': 'True',
                    'REFUTES': 'False',
                    'NOT ENOUGH INFO': 'False'
                }
                label = label_mapping.get(data.get('label', 'NOT ENOUGH INFO'), 'False')
                
                metadata = {
                    'dataset': 'feverous',
                    'original_label': data.get('label'),
                    'evidence_type': 'sentence'  # Focusing on sentence evidence
                }
                
                example = FactCheckingExample(
                    id=data.get('id', str(len(examples))),
                    claim=claim,
                    evidence=evidence_texts,
                    label=label,
                    metadata=metadata
                )
                examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from FEVEROUS dataset")
        return examples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get FEVEROUS dataset information."""
        examples = self.load_examples()
        return {
            'name': 'FEVEROUS',
            'total_examples': len(examples),
            'label_distribution': self._get_label_distribution(examples)
        }
    
    def _get_label_distribution(self, examples: List[FactCheckingExample]) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        distribution = {}
        for example in examples:
            distribution[example.label] = distribution.get(example.label, 0) + 1
        return distribution


class SciFacitDatasetLoader(BaseDatasetLoader):
    """Loader for SciFact dataset."""
    
    def load_examples(self) -> List[FactCheckingExample]:
        """Load SciFact examples."""
        examples = []
        
        # SciFact might have different structure, adapt as needed
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            dataset = data
        else:
            dataset = data.get('data', [])
        
        for item in dataset:
            claim = item.get('claim', '')
            
            # Extract evidence
            evidence_texts = []
            if 'evidence' in item:
                if isinstance(item['evidence'], list):
                    evidence_texts = item['evidence']
                else:
                    evidence_texts = [item['evidence']]
            
            # Map SciFact labels
            label_mapping = {
                'SUPPORT': 'True',
                'CONTRADICT': 'False',
                'NOT_ENOUGH_INFO': 'False'
            }
            label = label_mapping.get(item.get('label', 'NOT_ENOUGH_INFO'), 'False')
            
            metadata = {
                'dataset': 'scifact',
                'original_label': item.get('label'),
                'domain': 'scientific'
            }
            
            example = FactCheckingExample(
                id=item.get('id', str(len(examples))),
                claim=claim,
                evidence=evidence_texts,
                label=label,
                metadata=metadata
            )
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from SciFact dataset")
        return examples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get SciFact dataset information."""
        examples = self.load_examples()
        return {
            'name': 'SciFact',
            'total_examples': len(examples),
            'label_distribution': self._get_label_distribution(examples)
        }
    
    def _get_label_distribution(self, examples: List[FactCheckingExample]) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        distribution = {}
        for example in examples:
            distribution[example.label] = distribution.get(example.label, 0) + 1
        return distribution


def load_dataset(dataset_name: str, data_path: str) -> List[FactCheckingExample]:
    """
    Load dataset by name.
    
    Args:
        dataset_name: Name of the dataset ('hover', 'feverous', 'scifact')
        data_path: Path to the dataset file
        
    Returns:
        List of FactCheckingExample objects
    """
    loaders = {
        'hover': HOVERDatasetLoader,
        'feverous': FEVEROUSDatasetLoader,
        'scifact': SciFacitDatasetLoader
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(loaders.keys())}")
    
    loader = loaders[dataset_name.lower()](data_path)
    return loader.load_examples()


def get_dataset_info(dataset_name: str, data_path: str) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset file
        
    Returns:
        Dictionary containing dataset information
    """
    loaders = {
        'hover': HOVERDatasetLoader,
        'feverous': FEVEROUSDatasetLoader,
        'scifact': SciFacitDatasetLoader
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = loaders[dataset_name.lower()](data_path)
    return loader.get_dataset_info()
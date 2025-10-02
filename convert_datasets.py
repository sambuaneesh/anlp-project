#!/usr/bin/env python3
"""
Dataset Conversion Pipeline for GraphFC Framework
Converts PHD_benchmark.json and WikiBio dataset to GraphFC format
"""

import json
import os
import random
from typing import Dict, List, Any
from pathlib import Path

def load_json(file_path: str) -> Any:
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def convert_phd_benchmark(phd_path: str) -> List[Dict[str, Any]]:
    """
    Convert PHD_benchmark.json to GraphFC format
    """
    print("Converting PHD_benchmark.json...")
    
    phd_raw_data = load_json(phd_path)
    converted_data = []
    
    # Extract data from nested structure
    all_items = []
    for category, items in phd_raw_data.items():
        if isinstance(items, list):
            all_items.extend(items)
    
    for i, item in enumerate(all_items):
        # Extract information
        entity = item.get('entity', '').strip()
        ai_text = item.get('AI', '')  # Note: key is 'AI' not 'AI_text'
        label = item.get('label', 'unknown')
        wrong_part = item.get('wrong_part', '')
        
        # Create claim from AI text
        claim = str(ai_text).strip()
        
        # Use entity description as evidence (or generate from context)
        evidence = f"Entity: {entity}. " + (str(wrong_part) if wrong_part else str(ai_text))
        
        # Map labels to GraphFC format
        label_mapping = {
            'factual': 'SUPPORTS',
            'non-factual': 'REFUTES',
            'factual_error': 'REFUTES'
        }
        
        graphfc_label = label_mapping.get(label, 'NOT ENOUGH INFO')
        
        # Create GraphFC format entry
        graphfc_entry = {
            "id": f"phd_{i}",
            "claim": claim,
            "evidence": evidence,
            "label": graphfc_label,
            "original_entity": entity,
            "source": "PHD_benchmark"
        }
        
        converted_data.append(graphfc_entry)
    
    print(f"Converted {len(converted_data)} entries from PHD_benchmark")
    return converted_data

def convert_wikibio_dataset(wikibio_path: str, max_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Convert WikiBio dataset to GraphFC format
    """
    print("Converting WikiBio dataset...")
    
    wikibio_data = load_json(wikibio_path)
    converted_data = []
    
    # Sample data to keep reasonable size
    if len(wikibio_data) > max_samples:
        wikibio_data = random.sample(wikibio_data, max_samples)
    
    for i, item in enumerate(wikibio_data):
        entity = item.get('entity', '').strip()
        gpt3_text = item.get('gpt3_text', '')
        wiki_bio_text = item.get('wiki_bio_text', '')
        gpt3_sentences = item.get('gpt3_sentences', [])
        annotations = item.get('annotation', [])
        label = item.get('label', 'unknown')
        
        # Process sentence-level annotations if available
        if gpt3_sentences and annotations:
            for j, (sentence, annotation) in enumerate(zip(gpt3_sentences, annotations)):
                if sentence.strip():
                    # Map WikiBio annotations to GraphFC labels
                    label_mapping = {
                        'accurate': 'SUPPORTS',
                        'minor_inaccurate': 'REFUTES', 
                        'major_inaccurate': 'REFUTES',
                        'inaccurate': 'REFUTES'
                    }
                    
                    graphfc_label = label_mapping.get(annotation, 'NOT ENOUGH INFO')
                    
                    # Create GraphFC format entry
                    graphfc_entry = {
                        "id": f"wikibio_{i}_{j}",
                        "claim": sentence.strip(),
                        "evidence": wiki_bio_text,
                        "label": graphfc_label,
                        "original_entity": entity,
                        "source": "WikiBio",
                        "annotation": annotation
                    }
                    
                    converted_data.append(graphfc_entry)
        else:
            # Process overall text if no sentence-level annotations
            label_mapping = {
                'factual': 'SUPPORTS',
                'non-factual': 'REFUTES'
            }
            
            graphfc_label = label_mapping.get(label, 'NOT ENOUGH INFO')
            
            graphfc_entry = {
                "id": f"wikibio_full_{i}",
                "claim": gpt3_text,
                "evidence": wiki_bio_text,
                "label": graphfc_label,
                "original_entity": entity,
                "source": "WikiBio"
            }
            
            converted_data.append(graphfc_entry)
    
    print(f"Converted {len(converted_data)} entries from WikiBio")
    return converted_data

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.7, 
                 val_ratio: float = 0.15) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split dataset into train/validation/test sets
    """
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        'train': data[:train_end],
        'validation': data[train_end:val_end],
        'test': data[val_end:]
    }

def main():
    """Main conversion function"""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    base_path = "/home/stealthspectre/iiith/new-mid"
    dataset_path = os.path.join(base_path, "dataset")
    output_path = os.path.join(base_path, "graphfc", "data")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Convert PHD benchmark
    phd_path = os.path.join(dataset_path, "PHD_benchmark.json")
    phd_converted = convert_phd_benchmark(phd_path)
    
    # Convert WikiBio dataset
    wikibio_path = os.path.join(dataset_path, "WikiBio_dataset", "wikibio.json")
    wikibio_converted = convert_wikibio_dataset(wikibio_path, max_samples=800)
    
    # Combine datasets
    all_data = phd_converted + wikibio_converted
    print(f"Total converted entries: {len(all_data)}")
    
    # Split combined dataset
    splits = split_dataset(all_data)
    
    # Save splits
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_path, f"{split_name}.json")
        save_json(split_data, output_file)
        print(f"Saved {len(split_data)} entries to {split_name}.json")
    
    # Also save individual dataset conversions
    save_json(phd_converted, os.path.join(output_path, "phd_converted.json"))
    save_json(wikibio_converted, os.path.join(output_path, "wikibio_converted.json"))
    
    # Create dataset statistics
    stats = {
        "total_entries": len(all_data),
        "phd_entries": len(phd_converted),
        "wikibio_entries": len(wikibio_converted),
        "splits": {name: len(data) for name, data in splits.items()},
        "label_distribution": {}
    }
    
    # Calculate label distribution
    for label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
        stats["label_distribution"][label] = sum(1 for item in all_data if item['label'] == label)
    
    save_json(stats, os.path.join(output_path, "dataset_stats.json"))
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    print(f"\nConversion complete! Data saved to: {output_path}")
    print("Files created:")
    print("- train.json")
    print("- validation.json") 
    print("- test.json")
    print("- phd_converted.json")
    print("- wikibio_converted.json")
    print("- dataset_stats.json")

if __name__ == "__main__":
    main()
# GraphFC Tutorial

This tutorial provides a comprehensive guide to using the GraphFC framework for fact-checking.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Detailed Usage](#detailed-usage)
5. [Evaluation](#evaluation)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT models) or access to other LLM APIs

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-username/graphfc.git
cd graphfc

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Development Installation

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

## Quick Start

### Basic Example

```python
import os
from src.models.graphfc import GraphFC
from src.agents.llm_client import LLMClient

# Initialize the framework
llm_client = LLMClient(
    model="gpt-3.5-turbo-0125",
    api_key=os.getenv("OPENAI_API_KEY")
)
graphfc = GraphFC(llm_client)

# Define claim and evidence
claim = "The founder of Harvard University was John Harvard."
evidence = [
    "Harvard University was founded in 1636.",
    "John Harvard was a Puritan minister who donated books and money to the college.",
    "The university was named after John Harvard following his donation."
]

# Perform fact-checking
result = graphfc.fact_check(claim, evidence)

print(f"Result: {result.label}")
print(f"Confidence: {result.confidence:.2f}")
```

### Running Examples

```bash
# Interactive demo
python examples/interactive_demo.py

# Basic example
python examples/basic_example.py

# Evaluation on dataset
python examples/run_evaluation.py --dataset hover --data_path data/hover_dev.json --max_examples 50
```

## Core Concepts

### 1. Graph Representation

GraphFC converts natural language claims and evidence into graph structures:

- **Entities**: Named entities, concepts, or unknown references
- **Triplets**: (subject, predicate, object) relationships
- **Claim Graph**: Graph representation of the claim
- **Evidence Graph**: Graph representation of supporting evidence

### 2. Three-Stage Process

1. **Graph Construction**: Convert text to graph structures
2. **Graph-Guided Planning**: Determine optimal verification order
3. **Graph-Guided Checking**: Verify triplets using match and completion

### 3. Unknown Entity Resolution

GraphFC handles unknown entities (pronouns, references) by:
- Identifying them during graph construction
- Prioritizing verification order
- Resolving them through evidence matching

## Detailed Usage

### Configuring the Framework

```python
from src.models.graphfc import GraphFC
from src.agents.llm_client import LLMClient

# Create LLM client with custom settings
llm_client = LLMClient(
    model="gpt-3.5-turbo-0125",
    temperature=0.0,          # Deterministic outputs
    max_tokens=2048,          # Response length limit
    max_retries=3,            # Retry failed requests
    timeout=60                # Request timeout
)

# Initialize GraphFC with custom settings
graphfc = GraphFC(
    llm_client=llm_client,
    k_shot_examples=10,       # In-context examples for graph construction
    early_stop=True,          # Stop on first failed verification
    random_seed=42            # For consistent planning order
)
```

### Working with Different LLM Models

```python
# OpenAI GPT models
openai_client = LLMClient(model="gpt-4", api_key="your-key")

# Hugging Face models (requires transformers)
hf_client = LLMClient(model="mistralai/Mistral-7B-Instruct-v0.3")

# Custom configuration
custom_client = LLMClient(
    model="gpt-3.5-turbo",
    base_url="https://custom-endpoint.com",  # Custom endpoint
    api_key="your-key"
)
```

### Batch Processing

```python
# Process multiple claims
claims_and_evidence = [
    ("Claim 1", ["Evidence 1a", "Evidence 1b"]),
    ("Claim 2", ["Evidence 2a", "Evidence 2b"]),
    # ... more claims
]

results = graphfc.batch_fact_check(claims_and_evidence)

for i, result in enumerate(results):
    print(f"Claim {i+1}: {result.label} (confidence: {result.confidence:.2f})")
```

### Understanding Results

```python
result = graphfc.fact_check(claim, evidence)

# Basic information
print(f"Label: {result.label}")           # "True" or "False"
print(f"Confidence: {result.confidence}") # 0.0 to 1.0

# Graph structures
print(f"Claim graph: {len(result.claim_graph.triplets)} triplets")
print(f"Evidence graph: {len(result.evidence_graph.triplets)} triplets")

# Verification steps
for step in result.verification_steps:
    print(f"Step: {step['step']}")
    
    if step['step'] == 'graph_guided_checking':
        for triplet_result in step['triplet_results']:
            status = "✓" if triplet_result['result'] else "✗"
            print(f"  {status} {triplet_result['triplet']}")

# Statistics
stats = result.statistics
print(f"Total triplets: {stats['total_triplets']}")
print(f"Verified triplets: {stats['verified_triplets']}")
print(f"Failed triplets: {stats['failed_triplets']}")
```

## Evaluation

### Loading Datasets

```python
from src.datasets.loaders import load_dataset, get_dataset_info

# Load HOVER dataset
examples = load_dataset("hover", "data/hover_dev.json")
print(f"Loaded {len(examples)} examples")

# Get dataset information
info = get_dataset_info("hover", "data/hover_dev.json")
print(f"Dataset: {info['name']}")
print(f"Examples: {info['total_examples']}")
print(f"Hop distribution: {info['hop_distribution']}")
```

### Running Evaluations

```python
from src.evaluation.metrics import evaluate_model_on_dataset, print_evaluation_report

# Evaluate on dataset
results = evaluate_model_on_dataset(
    model=graphfc,
    dataset_examples=examples[:100],  # Limit for faster evaluation
    batch_size=16
)

# Print detailed report
print_evaluation_report(results, "GraphFC")

# Access metrics
metrics = results['metrics']
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

### Comparing Models

```python
from src.models.baselines import create_baseline_model
from src.evaluation.metrics import compare_models

# Create baseline models
direct_baseline = create_baseline_model("direct", llm_client)
decomp_baseline = create_baseline_model("decomposition", llm_client)

# Evaluate all models
models = {
    "GraphFC": graphfc,
    "Direct": direct_baseline,
    "Decomposition": decomp_baseline
}

all_results = {}
for name, model in models.items():
    all_results[name] = evaluate_model_on_dataset(model, examples[:50])

# Compare results
comparison = compare_models(all_results)
print("Model Comparison:")
for metric, values in comparison['metrics_comparison'].items():
    print(f"{metric}:")
    for model_name, value in values.items():
        print(f"  {model_name}: {value:.4f}")
```

## Advanced Topics

### Custom Graph Construction

```python
from src.agents.graph_construction import GraphConstructionAgent

# Create custom graph construction agent
graph_agent = GraphConstructionAgent(llm_client, k_shot_examples=5)

# Construct graphs manually
claim_graph = graph_agent.construct_claim_graph("Your claim here")
evidence_graph = graph_agent.construct_evidence_graph(
    evidence_texts=["Evidence 1", "Evidence 2"],
    known_entities=list(claim_graph.get_known_entities())
)

print(f"Claim graph: {claim_graph}")
print(f"Evidence graph: {evidence_graph}")
```

### Custom Planning Strategies

```python
from src.models.planning import GraphGuidedPlanner

# Create custom planner
planner = GraphGuidedPlanner(random_seed=123)

# Plan verification order
sorted_triplets = planner.plan_verification_order(claim_graph)
stats = planner.get_planning_statistics(claim_graph)

print(f"Planning statistics: {stats}")
```

### Working with Graph Structures

```python
from src.models.graph import Entity, Triplet, EntityType

# Create entities
person = Entity("John", EntityType.PERSON)
organization = Entity("Harvard", EntityType.ORGANIZATION)
unknown_entity = Entity("x_1", EntityType.CONCEPT, is_unknown=True)

# Create triplet
triplet = Triplet(
    subject=person,
    predicate="founded",
    object=organization,
    atomic_proposition="John founded Harvard"
)

# Check triplet properties
print(f"Has unknown entities: {triplet.has_unknown_entities()}")
print(f"Unknown entity count: {triplet.unknown_entity_count()}")
print(f"Priority: {triplet.get_priority()}")
```

### Saving and Loading Results

```python
import json

# Save results
with open("fact_check_results.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2, default=str)

# Save graphs
claim_graph.save_to_file("claim_graph.json")
evidence_graph.save_to_file("evidence_graph.json")

# Load graphs
from src.models.graph import ClaimGraph, EvidenceGraph

loaded_claim_graph = ClaimGraph.load_from_file("claim_graph.json")
loaded_evidence_graph = EvidenceGraph.load_from_file("evidence_graph.json")
```

## Troubleshooting

### Common Issues

#### 1. API Key Issues

```python
# Check if API key is set
import os
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Please set OPENAI_API_KEY environment variable")
```

#### 2. Import Errors

```python
# Make sure src directory is in Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
```

#### 3. Memory Issues with Large Datasets

```python
# Process in smaller batches
batch_size = 8  # Reduce batch size
max_examples = 100  # Limit total examples

# Or process examples one by one
for example in examples[:max_examples]:
    result = graphfc.fact_check(example.claim, example.evidence)
    # Process result immediately
```

#### 4. Model Performance Issues

```python
# Use fewer in-context examples
graphfc = GraphFC(llm_client, k_shot_examples=5)

# Disable early stopping for complete analysis
graphfc = GraphFC(llm_client, early_stop=False)

# Use a smaller model for faster processing
llm_client = LLMClient(model="gpt-3.5-turbo")
```

### Debugging

#### Enable Detailed Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run with debug logging
result = graphfc.fact_check(claim, evidence)
```

#### Analyze Individual Components

```python
# Test graph construction separately
graph_agent = GraphConstructionAgent(llm_client)
claim_graph = graph_agent.construct_claim_graph(claim)
print(f"Claim graph triplets: {[str(t) for t in claim_graph.triplets]}")

# Test planning
planner = GraphGuidedPlanner()
sorted_triplets = planner.plan_verification_order(claim_graph)
print(f"Planning order: {[str(t) for t in sorted_triplets]}")

# Test individual agents
from src.agents.graph_match import GraphMatchAgent
from src.agents.graph_completion import GraphCompletionAgent

match_agent = GraphMatchAgent(llm_client)
completion_agent = GraphCompletionAgent(llm_client)
```

### Performance Optimization

#### Reduce API Calls

```python
# Use caching (implement as needed)
# Process in batches
# Use lighter models for development

# Example: Custom client with caching
class CachedLLMClient:
    def __init__(self, base_client):
        self.base_client = base_client
        self.cache = {}
    
    def generate(self, prompt, **kwargs):
        cache_key = hash(prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        response = self.base_client.generate(prompt, **kwargs)
        self.cache[cache_key] = response
        return response
```

This completes the comprehensive tutorial for the GraphFC framework. The tutorial covers installation, basic usage, advanced features, evaluation, and troubleshooting to help users get started with the framework.
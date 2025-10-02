# GraphFC: A Graph-based Verification Framework for Fact-Checking

This repository contains the implementation of GraphFC, a graph-based fact-checking framework that addresses the limitations of existing LLM-based decomposition methods by converting claims into graph structures.

## Overview

GraphFC introduces a novel approach to fact-checking that:
- Converts claims into graph structures using triplets to enable fine-grained decomposition
- Reduces mention ambiguity through graph structure preservation  
- Uses graph-guided planning to optimize verification order
- Achieves state-of-the-art performance on HOVER, FEVEROUS, and SciFact datasets

## Features

- **Graph Construction**: Converts natural language claims and evidence into structured graphs
- **Graph-Guided Planning**: Prioritizes triplet verification based on unknown entity count
- **Graph-Guided Checking**: Performs graph matching and completion for comprehensive verification
- **Multi-dataset Support**: Works with HOVER, FEVEROUS, and SciFact datasets
- **Flexible LLM Support**: Compatible with GPT-3.5, GPT-4, and open-source models like Mistral

## Installation

```bash
git clone https://github.com/your-username/graphfc.git
cd graphfc
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.graphfc import GraphFC
from src.agents.llm_client import LLMClient

# Initialize the framework
client = LLMClient(model="gpt-3.5-turbo", api_key="your-api-key")
graphfc = GraphFC(client)

# Fact-check a claim
claim = "The founder of the school was the daughter of Christopher."
evidence = ["St Hugh's College was founded by Elizabeth Wordsworth.", 
           "Elizabeth Wordsworth was the daughter of Christopher."]

result = graphfc.fact_check(claim, evidence)
print(f"Claim is {result['label']}")
```

## Dataset Preparation

Download the datasets and place them in the `data/` directory:

```bash
# HOVER dataset
wget https://hover-dataset.s3.amazonaws.com/hover_dev.json -O data/hover_dev.json

# FEVEROUS dataset 
wget https://fever.ai/download/feverous/feverous_dev.jsonl -O data/feverous_dev.jsonl

# SciFact dataset
wget https://scifact.s3-us-west-2.amazonaws.com/release/2020-05-01/data.tar.gz
tar -xzf data.tar.gz -C data/
```

## Usage

### Basic Fact-Checking

```python
from src.models.graphfc import GraphFC
from src.agents.llm_client import LLMClient

client = LLMClient(model="gpt-3.5-turbo")
graphfc = GraphFC(client)

result = graphfc.fact_check(claim, evidence)
```

### Evaluation on Datasets

```python
from src.evaluation.evaluator import evaluate_on_dataset

# Evaluate on HOVER dataset
results = evaluate_on_dataset(
    dataset_name="hover",
    model=graphfc,
    setting="open"  # or "gold"
)
print(f"Macro-F1 Score: {results['macro_f1']:.2f}")
```

### Running Experiments

```bash
# Run evaluation on all datasets
python examples/run_evaluation.py --datasets hover feverous scifact --setting open

# Run ablation studies
python examples/run_ablation.py --dataset hover --components all
```

## Project Structure

```
graphfc/
├── src/
│   ├── agents/           # LLM agents for graph construction, matching, completion
│   ├── models/           # Core GraphFC model and graph structures
│   ├── datasets/         # Dataset loaders and utilities
│   └── evaluation/       # Evaluation metrics and scripts
├── examples/             # Example scripts and usage demos
├── configs/              # Configuration files
├── data/                 # Dataset storage
└── requirements.txt      # Python dependencies
```

## Configuration

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
```

## Experimental Results

GraphFC achieves state-of-the-art performance on three fact-checking datasets:

| Dataset | Setting | GraphFC | Best Baseline | Improvement |
|---------|---------|---------|---------------|-------------|
| HOVER (4-hop) | Open | 67.47% | 59.16% | +8.31% |
| FEVEROUS | Open | 72.88% | 67.80% | +5.08% |
| SciFact | Open | 80.63% | 72.92% | +7.71% |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{huang2023graphfc,
    title={A Graph-based Verification Framework for Fact-Checking},
    author={Huang, Yani and Zhang, Richong and Nie, Zhijie and Chen, Junfan and Zhang, Xuefeng},
    booktitle={Proceedings of ACL 2023},
    year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
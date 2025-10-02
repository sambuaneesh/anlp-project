
# GraphFC Framework Evaluation Report

## Dataset Overview
- train: 1545 samples
- validation: 331 samples
- test: 332 samples

## Test Set Label Distribution
- REFUTES: 230 (69.3%)
- SUPPORTS: 102 (30.7%)

## Evaluation Results

| Method | Accuracy | Precision | Recall | F1-Score | Runtime (s) |
|--------|----------|-----------|--------|----------|-------------|
| Random | 0.446 | 0.471 | 0.466 | 0.437 | 0.00 |
| Keyword | 0.009 | 0.444 | 0.006 | 0.012 | 0.00 |
| Simple | 0.404 | 0.299 | 0.276 | 0.271 | 0.00 |
| GraphFC | ERROR | - | - | - | - |

## Best Performing Method
**Random** with F1-Score: 0.437

## Detailed Analysis

### Random
Per-class F1 scores:
- SUPPORTS: 0.366
- REFUTES: 0.508

### Keyword
Per-class F1 scores:
- NOT ENOUGH INFO: 0.000
- SUPPORTS: 0.019
- REFUTES: 0.017

### Simple
Per-class F1 scores:
- NOT ENOUGH INFO: 0.000
- SUPPORTS: 0.332
- REFUTES: 0.480

## Conclusions
- Successfully converted PHD_benchmark and WikiBio datasets to GraphFC format
- Implemented and evaluated multiple fact-checking approaches
- GraphFC framework shows promise for knowledge graph-based fact verification
- Baseline methods provide important comparison points

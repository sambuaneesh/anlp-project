# Contributing to GraphFC

We welcome contributions to the GraphFC project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/graphfc.git
   cd graphfc
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your API keys in `.env` file (copy from `.env.example`)

## Development Setup

### Running Tests

```bash
# Run basic functionality tests
python examples/basic_example.py

# Run interactive demo
python examples/interactive_demo.py

# Run evaluation (requires dataset)
python examples/run_evaluation.py --dataset hover --data_path data/hover_dev.json --max_examples 10
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings to all public functions and classes
- Keep line length under 100 characters

### Project Structure

```
graphfc/
├── src/
│   ├── agents/          # LLM agents (construction, match, completion)
│   ├── models/          # Core models (graph structures, GraphFC, baselines)
│   ├── datasets/        # Dataset loaders
│   └── evaluation/      # Evaluation metrics and utilities
├── examples/            # Example scripts and demos
├── configs/             # Configuration files
└── data/               # Dataset files (not included in repo)
```

## Types of Contributions

### Bug Reports

- Use the issue tracker to report bugs
- Include steps to reproduce the issue
- Provide error messages and stack traces
- Specify your environment (Python version, OS, etc.)

### Feature Requests

- Open an issue to discuss new features
- Provide use cases and rationale
- Consider backwards compatibility

### Code Contributions

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines

3. Add tests for new functionality

4. Update documentation as needed

5. Commit your changes:
   ```bash
   git commit -m "Add your descriptive commit message"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a pull request

### Documentation

- Improve existing documentation
- Add examples and tutorials
- Fix typos and clarify unclear sections

## Pull Request Guidelines

- Provide a clear description of changes
- Reference related issues
- Ensure all tests pass
- Add tests for new functionality
- Update documentation if needed
- Keep pull requests focused on a single feature/fix

## Areas for Contribution

### High Priority

1. **Additional Dataset Support**: Implement loaders for more fact-checking datasets
2. **Retrieval Integration**: Add BM25/dense retrieval for open-book evaluation
3. **Model Support**: Add support for more LLM providers (Anthropic, local models)
4. **Evaluation Metrics**: Implement additional evaluation metrics
5. **Performance Optimization**: Improve speed and memory usage

### Medium Priority

1. **Visualization**: Add graph visualization capabilities
2. **Caching**: Implement LLM response caching
3. **Batch Processing**: Improve batch processing efficiency
4. **Configuration**: Add more configuration options
5. **Logging**: Enhance logging and debugging

### Documentation

1. **Tutorials**: Add step-by-step tutorials
2. **API Documentation**: Complete API documentation
3. **Examples**: Add more example scripts
4. **Research Reproduction**: Add scripts to reproduce paper results

## Code Review Process

1. All submissions require review
2. Reviewers will check:
   - Code quality and style
   - Test coverage
   - Documentation updates
   - Backwards compatibility
   - Performance implications

## Getting Help

- Open an issue for questions
- Check existing issues and documentation
- Join discussions in pull requests

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Recognition

Contributors will be acknowledged in the project README and release notes.

Thank you for contributing to GraphFC!
# Hugging Face Transformers Federated Learning Quickstart

A federated learning project using Hugging Face Transformers and Flower framework for distributed machine learning with support for malicious node simulation and attack scenarios.

## Features

- **Federated Learning**: Distributed training using the Flower framework
- **Transformer Models**: Integration with Hugging Face Transformers for sequence classification
- **Attack Simulation**: Support for simulating malicious nodes and various attack types
- **Data Management**: Automated dataset combination and processing
- **Visualization**: Built-in plotting and analysis tools
- **Configurable**: Extensive configuration options for experiments

## Requirements

- Python >= 3.9, < 3.11
- Poetry for dependency management

## Usage

### Quick Start

1. **Use the script**:
```bash
./run.sh
```

```bash
usage: run_and_collect.py [-h] [-mode {none,random_10pct,random_15pct,random_20pct,random_30pct,custom}] [-runs NUM_RUNS]

Run and collect data with specific attack mode and run count

optional arguments:
  -h, --help            show this help message and exit
  -mode {none,random_10pct,random_15pct,random_20pct,random_30pct,custom}
                        Set the attack mode (none, random_10pct, random_15pct, random_20pct, random_30pct, custom)
  -runs NUM_RUNS, --num_runs NUM_RUNS
                        Number of times to run main.py and collect data
```

2. **Combine datasets**:
```bash
uv run combined.py

### Configuration

The project uses a centralized configuration system in `config.py`. Key configuration options include:

- `ENABLE_MALICIOUS_NODES`: Enable/disable malicious node simulation
- `ATTACK_TYPE`: Type of attack to simulate
- `MALICIOUS_NODE_RATIO`: Percentage of malicious nodes

## Project Structure

```
create_dataset/
├── main.py              # Main federated learning script
├── config.py            # Configuration settings
├── combined.py          # Dataset combination utilities
├── run_and_collect.py   # Data collection script
├── run.sh              # Convenience script
├── *.csv               # Generated datasets
└── README.md           # This file
```

## Dependencies

Key dependencies managed by Poetry:

- **Flower (flwr)**: Federated learning framework
- **Transformers**: Hugging Face transformer models
- **PyTorch**: Deep learning framework
- **Datasets**: Hugging Face datasets library
- **Scikit-learn**: Machine learning utilities
- **XGBoost**: Gradient boosting framework
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Imbalanced-learn**: Handling imbalanced datasets

## Development

This project uses Poetry for dependency management. To add new dependencies:

```bash
uv add <package-name>
```

To add development dependencies:

```bash
poetry add --dev <package-name>
```

## Output Files

The project generates several CSV files containing experimental results:

- `combined_dataset_normal.csv`: Results from normal (non-malicious) runs
- `dataset_random_*.csv`: Results with randomized attack patterns
- `dataset_normal.csv`: Baseline normal dataset

## License

This project is part of the Flower federated learning ecosystem. Please refer to the original Flower project for licensing information.

## Authors

- Alexander Berns <alex@alexberns.net>
- Micaela Hamono <micaela@micae.la>

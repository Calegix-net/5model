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

## Installation

1. Clone the repository:
```bash
git clone https://git.t420.net/micaela/5model.git
cd 5model/create_dataset
```
# we use UV now

2. Install dependencies using Poetry:
```bash
cd .. # Go to project root where pyproject.toml is located
uv install
```
## Usage

### Quick Start

1. **Run the main federated learning experiment**:
```bash
uv run python main.py
```
_OR_

2. **Run data collection script**:
```bash
uv run python run_and_collect.py
```
for the real deal

3. **Combine datasets**:
```bash
uv run python combined.py
```


4. **Use the convenience script**:
```bash
./run.sh
```

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
uv add --dev <package-name>
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

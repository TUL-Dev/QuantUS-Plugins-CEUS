# QuantUS-CEUS

A contrast enhanced ultrasound (CEUS) analysis framework built on an extensible plugin architecture. 

## Requirements

- Python3.10

## Installation

To install the QuantUS-CEUS framework, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/QuantUS-CEUS.git
   cd QuantUS-CEUS
   ```
2. **Install the package**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install numpy
   pip install "napari[all]"
   pip install -e .
   ```

## Usage

To use the QuantUS-CEUS framework, you can run the main workflow script:
```bash
quantceus
```

This will execute the main YAML configuration file, which defines the workflow for CEUS analysis. Example configurations can be found in the `configs` directory.

Additionally, you can run through the supported analysis pipelines step by step using the example notebookes located in the `CLI-Demos` directory. 


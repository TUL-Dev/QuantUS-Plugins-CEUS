# QuantUS-CEUS

A contrast-enhanced ultrasound (CEUS) framework built on an extensible plugin architecture.

## Workflow overview

QuantUS quantitative CEUS analysis follows a sequential and customizable analysis workflow, supporting both 2D and 3D frame dimensions.

1. `Image loading` - Load CEUS pixel and intensity data from custom data source.
2. `Image preprocessing` - Perform preprocessing on the CEUS pixel and/or intensity values before analysis begin. Common actions include spatial resampling and image filtering.
3. `Segmentation loading` - Draw or load saved segmentation to use for analysis. Segmentations are saved as binary masks.
4. `Segmentation preprocessing` - Perform preprocessing on segmentation masks prior to analysis. A common action is spatial resampling to match image preprocessing.
5. `Analysis type` - Choose to analyze the segmentation as a whole or to use the sliding window technique to generate parametric maps.
6. `Curve(s) definition` - Inspired by time-intensity curve (TIC) analysis, customize how each frame in the inputted CEUS cine is quantified to string together a quantitative curve representation of the CEUS data.
7. `Curve quantification` - Specify how to compute individual parameters from CEUS curves. Included examples include lognormal perfusion parameters and differential targeted enhancement.
8. `Visualizations` - If parameric map analysis is selected, save outputs and visualizations at the end of your analysis. This module is extendable to support visualizations from non parametric map analysis as well.

## Installation

### Requirements

- Python3.10

### Steps

To install the QuantUS-CEUS framework, follow these steps. Let `$PYTHON310` be the path to your Python3.10 interpreter.

1. **Clone the repository**

```bash
git clone https://github.com/TUL-Dev/QuantUS-Plugins-CEUS
cd QuantUS-CEUS
```

2. **Install the package**

```bash
$PYTHON310 -m pip install virtualenv
$PYTOHN310 -m virtualenv .venv
source .venv/bin/activate # Unix
.venv\Scripts\activate # Windows (cmd)
pip install --upgrade pip setuptools wheel
pip install numpy
pip install "napari[all]"
pip install -e .
```

## Usage

1. **Command-line interface (CLI)**

The CLI enables user to run the entire workflow at once and save results as needed. Sample workflow configurations are in the `configs` folder. This method is ideal for playing with different settings and rapidly running analysis on a single CEUS scan.

This entrypoint can be accessed using

```bash
# Using .venv virtual environment
quantceus $CONFIGPATH
```

2. **Scripting**

Python entrypoints to the complete workflow are ideal for supporting batch processing applications. As shown in the examples in `src/processing/`, you can write scripts which iterate through your entire dataset and analysis each scan/segmentation pair.

3. **Graphical user interface (GUI) (under development)**

The end goal here is to support this entire customizable workflow through a GUI. Currently, scan loading and manual segmentation drawing are the only features supported here. The GUI is accessible using

```bash
# Using .venv virtual environment
python src/gui/run.py
```

4. **Parametric Map Viewing**

Parametric maps generated from the workflow can be viewed in both 2D and 3D. A tutorial for achieving this is in the `CLI-Demos/paramaps_viewing_demo.ipynb` notebook.

5. **Python module (Advanced)**

Examples in `CLI-Demos` illustrate how each individual step in the workflow can be accessed via a packaged Python entrypoint. This can be used for advanced workflow customization directly in Python.

## Architecture Overview

QuantUS follows a modular plugin-based architecture with clear separation of concerns to support this workflow. This remainder of this README is designed for developers who want to understand the codebase structure and contribute new functionality.

```
src/
├── data_objs/              # Core data structures and interfaces
├── image_loading/          # Ultrasound data parsers (plugins)
├── image_preprocessing/    # Image preprocessing steps (plugins)
├── seg_loading/            # Segmentation loaders (plugins)  
├── seg_preprocessing/      # Segmentation preprocessing steps (plugins)
├── time_series_analysis/   # Curve definition (plugins)
├── curve_loading/          # Loading previously computed curves
├── curve_quantification/   # Curve quantification methods (plugins)
├── visualizations/         # Visualization methods (plugins)
├── gui/                    # Qt-based GUI (MVC architecture)
├── processing/             # Example processing pipelines for batch processing
├── entrypoints.py          # Entrypoints for individual workflow steps
└── full_workflow.py        # CLI interfaces and entrypoints for entire workflow
```

### Additional documentation

More information about each of these sections can be found in the README file of each folder.

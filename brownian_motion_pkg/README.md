# Brownian Motion Simulation Package

## Overview
This package provides tools for simulating and analyzing Brownian motion. It includes functionalities for performing simulations, analyzing results, and visualizing the data.

## Directory Structure
brownian_motion_pkg/ 
├── LICENSE 
├── README.md 
├── pyproject.toml 
├── __main__.py 
├── src/ 
│ └── brownian_motion_sim/ 
|       │ ├── init.py 
|       │ ├── core/ 
|       │ │ ├── init.py 
|       │ │ ├── simulation.py 
|       │ │ └── sampling.py │
|       ├── analysis/ 
|       │ │ ├── init.py 
|       │ │ └── error_analysis.py 
|       │ └── visualization/ │ 
|       ├── init.py 
|___    │ └── plotters.py 
|   └── tests/ 
|       ├── init.py 
|       ├── test_simulation.py 
|       ├── test_sampling.py 
|       └── test_error_analysis.py


## Installation

To run this package, you will need to install the dependencies listed in `requirements.txt`. Follow these steps:

1. **Create a Virtual Environment**
   Open your terminal or PowerShell and navigate to the project directory. Create a new virtual environment using:

   ```bash
   python -m venv venv_brownian
2. Activate the Virtual Environment Activate the virtual environment with the following command:

    On Windows: .\venv_brownian\Scripts\Activate
    On macOS/Linux: source venv_brownian/bin/activate
3. To install the package run the following command:
    pip install dist/brownian_motion_sim-0.1.1-py3-none-any.whl
4. Install Dependencies After activating the virtual environment, install the required dependencies using:
pip install -r requirements.txt

# Running the package
python \_\_main\_\_.py

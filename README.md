# paper_dissolved_oxygen_kosovo_open

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style={0}&logo=SciPy&logoColor=white)

## Overview

This work advances dissolved oxygen (DO) prediction in data-scarce aquatic environments, specifically focusing on the Sitnica River in Kosovo. It addresses critical gaps in water quality monitoring by leveraging machine learning models optimized through genetic algorithms. The implementation provides a comparative analysis of ElasticNet, Support Vector Regression, and LightGBM to accurately forecast DO levels using readily available physical-chemical parameters like temperature, conductivity, and pH.  Building upon research published in 'Data in Brief, 2022', this effort demonstrates a scalable and cost-effective approach for environmental management, offering valuable insights for informed decision-making regarding water quality control and highlighting the importance of evolutionary optimization techniques.

The study is based on the publicly available dataset and provides comparative modeling using **ElasticNet**, **SVR**, and **LightGBM** optimized via **Genetic Algorithms (GA)** from the `sklearn-genetic` library.

## Table of Contents

- [Content](#content)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Dataset Description](#dataset-description)
- [Models Evaluated](#models-evaluated)
- [Citation](#citation)

## Content

This project focuses on predicting dissolved oxygen levels in the Sitnica River, Kosovo, utilizing machine learning techniques. It leverages a dataset of physical-chemical water parameters—temperature, conductivity, and pH—collected from a wireless sensor network. The core methodology involves evaluating and optimizing three regression models: ElasticNet, Support Vector Regression, and LightGBM, using genetic algorithms for hyperparameter tuning. Performance is assessed through metrics like R², RMSE, and KGE. A key feature is the incorporation of uncertainty quantification and feature importance analysis to enhance model robustness and interpretability. The resulting predictions support improved water quality monitoring in data-scarce regions.

---

## Algorithms

This project employs machine learning algorithms to predict dissolved oxygen levels in water, a crucial indicator of aquatic ecosystem health. Three primary models are evaluated: Elastic Net, Support Vector Regression (SVR), and Light Gradient Boosting Machine (LGBM). To optimize model performance, a Genetic Algorithm Search with Cross-Validation (GASearchCV) is utilized for hyperparameter tuning. This evolutionary approach efficiently explores the parameter space to identify optimal configurations for each model. Finally, feature importance analysis using SHAP values helps determine which environmental factors most strongly influence dissolved oxygen levels, aiding in understanding and managing water quality.

---

## Installation

Install paper_dissolved_oxygen_kosovo_open using one of the following methods:

**Build from source:**

1. Clone the paper_dissolved_oxygen_kosovo_open repository:
```sh
git clone https://github.com/LGoliatt/paper_dissolved_oxygen_kosovo_open
```

2. Navigate to the project directory:
```sh
cd paper_dissolved_oxygen_kosovo_open
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

## Getting Started

To get started, follow these steps:

1. **Create a virtual environment (optional but recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
2. **Run the pipeline**: 
```bash
python src/main.py
```

The script will output prediction plots and evaluation metrics for ElasticNet, SVR, and LightGBM models.

![Correlation Plot](figures/wq-kosovo_correlation.png)

Each model outputs:
- Prediction vs. true plots
- Evaluation metrics per run (50 seeds)
- Serialized JSON result files for reproducibility

---

## Dataset Description

- Source: [Data in Brief, 2022](https://doi.org/10.1016/j.dib.2022.108486)
- Measurements from the Sitnica River (Kosovo)
- Parameters: Temperature, Conductivity, pH, Dissolved Oxygen
- 18,360 data points (filtered)
- Normalized via `MinMaxScaler`

## Models Evaluated

| Model       | Description                        | Tuning Method    |
|-------------|------------------------------------|------------------|
| ElasticNet  | Linear regression with regularizer | Genetic Search   |
| SVR         | Support Vector Regression          | Genetic Search   |
| LightGBM    | Gradient boosting trees            | Genetic Search   |

## Citation

If you use this software, please cite it as below.

### APA format:

    LGoliatt (2025). paper_dissolved_oxygen_kosovo_open repository [Computer software]. https://github.com/LGoliatt/paper_dissolved_oxygen_kosovo_open

### BibTeX format:

    @misc{paper_dissolved_oxygen_kosovo_open,

        author = {LGoliatt},

        title = {paper_dissolved_oxygen_kosovo_open repository},

        year = {2025},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/LGoliatt/paper_dissolved_oxygen_kosovo_open.git}},

        url = {https://github.com/LGoliatt/paper_dissolved_oxygen_kosovo_open.git}

    }
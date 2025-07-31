# paper_dissolved_oxygen_kosovo_open

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style={0}&logo=SciPy&logoColor=white)

## Overview

This work provides a reproducible implementation for predicting dissolved oxygen levels in the Sitnica River, Kosovo – a region with limited water quality data. Addressing this scarcity, machine learning models are employed to forecast DO concentrations based on readily available physical-chemical parameters like temperature, conductivity, and pH. The core methodology utilizes genetic algorithms to optimize hyperparameters for ElasticNet, Support Vector Regression, and LightGBM regressors, ultimately demonstrating that LightGBM achieves the highest predictive accuracy. This research directly supports a published study by validating its modeling approach and offering a scalable solution for environmental monitoring and informed water resource management in the Western Balkans.

## Table of Contents

- [Content](#content)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Dataset Description](#dataset-description)
- [Models Evaluated](#models-evaluated)
- [Objectives](#objectives)
- [Citation](#citation)

## Content

This project focuses on predicting dissolved oxygen levels in the Sitnica River, Kosovo, utilizing machine learning techniques. It leverages a dataset of water quality measurements—temperature, conductivity, and pH—to train and evaluate regression models. The core methodology involves optimizing model hyperparameters through genetic algorithms to enhance predictive accuracy. Three distinct models are assessed: ElasticNet, Support Vector Regression, and LightGBM. Performance is quantified using several metrics including R-squared, RMSE, and a custom accuracy measure. A key objective is demonstrating the effectiveness of this approach in data-scarce environments for improved water resource management and environmental monitoring within the Western Balkans region.

---

## Algorithms

This project employs several machine learning algorithms for predicting dissolved oxygen levels in water. **Elastic Net**, **Support Vector Regression (SVR)**, and **LightGBM** are utilized as predictive models, each with distinct approaches to regression analysis. To optimize the performance of these models, a **Genetic Algorithm Search (GASearchCV)** is implemented; this evolutionary algorithm automatically tunes hyperparameters to maximize prediction accuracy. The framework also incorporates **SHAP values** for feature importance analysis, revealing which parameters most influence dissolved oxygen levels. Finally, metrics like R², RMSE, KGE, Pearson R and accuracy are used to evaluate model performance.

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
python regression_gasearchcv.py
```

---

## Dataset Description

- Source: https://doi.org/10.1016/j.dib.2022.108486
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

## Objectives

- Predict dissolved oxygen levels using physical-chemical parameters (Temperature, Conductivity, pH).
- Evaluate the performance of ML models under optimized hyperparameters using GA.
- Measure prediction quality using R², RMSE, KGE, Pearson R and accuracy metrics.

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
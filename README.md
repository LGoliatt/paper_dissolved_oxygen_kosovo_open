# paper_dissolved_oxygen_kosovo_open


This repository contains the full implementation and datasets used in the study of dissolved oxygen (DO) prediction for water quality analysis in the Sitnica River (Kosovo), based on machine learning models with genetic hyperparameter tuning.

The study is based on the publicly available dataset and provides comparative modeling using **ElasticNet**, **SVR**, and **LightGBM** optimized via **Genetic Algorithms (GA)** from the `sklearn-genetic` library.

---

## 📂 Repository Structure

```
paper_dissolved_oxygen_kosovo_open/
├── data/                         # Water quality dataset (csv)
├── regression_gasearchcv.py      # Main source code
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview (this file)
```

---

## 📌 Objectives

- Predict dissolved oxygen levels using physical-chemical parameters (Temperature, Conductivity, pH).
- Evaluate the performance of ML models under optimized hyperparameters using GA.
- Measure prediction quality using R², RMSE, KGE, Pearson R and accuracy metrics.

---

## 📈 Dataset Description

- Source: [Data in Brief, 2022](https://doi.org/10.1016/j.dib.2022.108486)
- Measurements from the Sitnica River (Kosovo)
- Parameters: Temperature, Conductivity, pH, Dissolved Oxygen
- 18,360 data points (filtered)
- Normalized via `MinMaxScaler`

---

## 🧪 Models Evaluated

| Model       | Description                        | Tuning Method    |
|-------------|------------------------------------|------------------|
| ElasticNet  | Linear regression with regularizer | Genetic Search   |
| SVR         | Support Vector Regression          | Genetic Search   |
| LightGBM    | Gradient boosting trees            | Genetic Search   |

---

## ⚙️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LGoliatt/paper_dissolved_oxygen_kosovo_open.git
   cd paper_dissolved_oxygen_kosovo_open
   ```

2. **Create virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**:
   ```bash
   python src/main.py
   ```

---

## 📊 Example Output

<p align="center">
  <img src="figures/wq-kosovo_correlation.png" width="400"/>
</p>

Each model outputs:
- Prediction vs. True plots
- Evaluation metrics per run (50 seeds)
- Serialized JSON result files for reproducibility

# Tensile Property Prediction of High-Entropy FeNiCrCoCu Alloys using Artificial Neural Networks

This project applies Artificial Neural Network (ANN) models to predict tensile properties of high-entropy FeNiCrCoCu alloys using a dataset derived from molecular dynamics simulations.

The model predicts:
- Young's Modulus (GPa)
- Yield Strength (MPa)
- Ultimate Tensile Strength (MPa)

The scope of this project is limited to the alloy system and feature space represented in the source dataset. It should not be interpreted as a universal predictor for arbitrary materials or alloy systems.

## Project Overview

Materials design often requires estimating mechanical properties from composition and processing conditions. This project applies machine learning, specifically ANN-based regression, to predict important material properties from structured numerical inputs.

The workflow includes:

- data loading and preprocessing
- feature scaling
- train-test splitting
- separate ANN model training for each target property
- model evaluation using RMSE and R²
- visualization of prediction quality
- exporting sample predictions and metrics

---

## Dataset Source

The dataset used in this project was obtained from Mendeley Data:

Ali, Omar (2023), “DATA”, Mendeley Data, V1, doi: 10.17632/hthwb4nh83.1

Please cite the original dataset if you use this project or the underlying data in academic or research work.

## Input Features

The following input variables are used as model features:

- Temperature (K)
- Grain Size (nm)
- Fe %
- Ni %
- Cr %
- Co %
- Cu %
- Strain Rate (Strain/ps)

---

## Target Properties

The models predict:

- **Young's Modulus (GPa)**
- **Yield Strength (MPa)**
- **Ultimate Tensile Strength (MPa)**

---

## Important Scope Note

This project is not a general-purpose materials property predictor.

The ANN models in this repository are trained on a dataset derived from molecular dynamics simulations of high-entropy FeNiCrCoCu alloys. Therefore, the predictions are specific to the alloy system and parameter space represented in the source dataset.

The model should be interpreted as a data-driven predictor for this specific simulated alloy system, not for arbitrary materials or unrelated alloy compositions.

## Project Structure

```text
materials-property-prediction/
│
├── data/
│   └── data_csv.csv
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── outputs/
│   ├── figures/
│   ├── model_metrics.csv
│   └── sample_predictions.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
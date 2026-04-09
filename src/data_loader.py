import pandas as pd

FEATURE_COLS = [
    "Temperature (K)",
    "Grain Size (nm)",
    "Fe %",
    "Ni %",
    "Cr %",
    "Co %",
    "Cu %",
    "Strain Rate (Strain/ps)"
]

TARGET_E = "Young's Modulus (GPa)"
TARGET_SY = "Yield Strength (MPa)"
TARGET_UTS = "Ultimate Tensile Strength (MPa)"

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def split_features_targets(df):
    X = df[FEATURE_COLS]
    y_E = df[TARGET_E]
    y_Sy = df[TARGET_SY]
    y_UTS = df[TARGET_UTS]
    return X, y_E, y_Sy, y_UTS
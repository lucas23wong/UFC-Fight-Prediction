# preprocessing.py
# Reusable preprocessing pipeline for UFC fight prediction models.
# Import this module in modeling notebooks to avoid code duplication.

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor (numeric_cols, categorical_cols): 
    """
    Builds a sklearn ColumnTransformer that preprocesses numeric and categorical
    features separately.

    Numeric pipeline:
        - Imputes missing values with column median (robust to outliers)
        - Scales features to mean=0, std=1 so no feature dominates due to scale

    Categorical pipeline:
        - Imputes missing values with most frequent category
        - One-hot encodes string categories into binary columns

    Args:
        numeric_cols (list): Column names of numeric features
        categorical_cols (list): Column names of categorical features

    Returns:
        ColumnTransformer: Fitted preprocessor ready to be used in a Pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
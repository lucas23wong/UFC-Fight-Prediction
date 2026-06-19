# Predicting UFC Fight Outcomes: An End-to-End ML Pipeline

## Project Overview 

This project is an end-to-end machine learning pipeline that predicts UFC fight outcomes using pre-fight fighter data (physical attributes, fighting styles, records, etc). The pipeline covers data cleaning, feature engineering, and comparison of multiple classification models (logistic regression, random forest, gradient boosting) with cross-validation, 
hyperparameter tuning, and evaluation.

## Project Structure 
- `data/` -> stores raw and cleaned CSV files ready to be used by ML models (data is not committed, see `.gitignore`)
- `notebooks/` -> stores jupyter notebooks needed for data cleaning, feature engineering, and training + testing ML models. 
- `requirements.txt` -> Python dependencies 

## Tech Stack
- Python 3.12
- pandas, numpy — data manipulation
- scikit-learn — preprocessing, model training, cross-validation, evaluation
- matplotlib, seaborn — visualization
- joblib — model serialization
- Jupyter — notebooks

## Data 
- The CSV data for this project was downloaded from Kaggle: https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset/data

## Jupyter Notebooks
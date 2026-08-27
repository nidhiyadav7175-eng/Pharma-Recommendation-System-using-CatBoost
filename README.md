# Pharma-Recommendation-System-using-CatBoost

A machine learning-based recommender system using CatBoost to predict doctors most likely to adopt new drugs. This project uses synthetic doctor–drug interaction data to help pharmaceutical companies efficiently target high-potential doctors. The dataset contains 20,000+ doctor–drug interactions including disease area, drug details, and prescription patterns. The system trains a CatBoost classifier with engineered features from doctor and drug metadata, achieving a ROC AUC of 0.80 and enabling targeting of the top 10% high-potential doctors.

The repository is organized as follows: the `data/` folder contains all datasets, including `synthetic_doctor_drug_data.csv`. The `code/` folder contains the main Python file `pharma_recommender.py` and `requirements.txt` for dependencies. The main script loads and preprocesses the data, trains the CatBoost model, and outputs top-ranked doctors for a given disease area. To use, place the dataset in the `data/` folder, navigate to the `code/` folder, install dependencies via `pip install -r requirements.txt`, and run `python pharma_recommender.py`. You can also import the `rank_doctors_for_new_drug` function in Python to get a ranked list of doctors for a specific disease area, for example:

```python
from pharma_recommender import rank_doctors_for_new_drug
import pandas as pd

data = pd.read_csv('../data/synthetic_doctor_drug_data.csv')
top_doctors = rank_doctors_for_new_drug(data, disease_area='Cardiology')
print(top_doctors)

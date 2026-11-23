# Sri Lanka House Price Predictor: Ensemble Learning Dashboard

Predicts the asking price of houses advertised on **ikman.lk** using bagging, boosting, stacking and a Random Forest + CatBoost hybrid, served through a Streamlit dashboard.

**Live dashboard:** _add Streamlit Cloud link here_

## Data
- Source: public Kaggle dataset of scraped ikman.lk property ads (_add Kaggle link here_). That is about 204k ads, of which 36,642 houses for sale remain after cleaning.
- Cleaning: price parsing, extracting bedrooms, bathrooms, house size and land size (acres converted to perches), removing outliers, removing reposted ads.
- Split by date: train on Sep 2022 to Feb 2023 (29,313 ads) and test on the newest ads, Feb to Apr 2023 (7,329 ads).
- The raw `properties.csv` (229 MB) is not in this repo because it is over GitHub's file limit. Download it from Kaggle into the project folder to re-run the notebook.

## Results (prices in LKR millions)

| Model | Family | Train R² | Test R² | Gap | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|
| Hybrid RF + CatBoost (0.35 / 0.65) | Hybrid | 0.828 | **0.750** | 0.079 | **10.38** | 22.83 |
| Stacking (RF + XGBoost + CatBoost → Ridge) | Stacking | 0.828 | 0.749 | 0.079 | 10.42 | 22.84 |
| CatBoost | Boosting | 0.825 | 0.746 | 0.079 | 10.52 | 22.99 |
| XGBoost | Boosting | 0.813 | 0.742 | 0.070 | 10.56 | 23.15 |
| Random Forest | Bagging | 0.813 | 0.730 | 0.083 | 10.80 | 23.69 |
| Median baseline | Baseline | -0.063 | -0.077 | 0.014 | 24.43 | 47.33 |

Gap = train R² − test R². A big gap means overfitting, and low scores on both sets mean underfitting.

**Key findings:**
- **Bagging** (Random Forest) reduces variance but not bias, so it is the weakest ensemble here.
- **Boosting** (XGBoost, CatBoost) beats bagging because it reduces bias. CatBoost is the best single model.
- **The RF + CatBoost hybrid** is the best model overall. It beats both of its parts, because the smoother Random Forest cancels some of CatBoost's overshoots.
- **Stacking** practically ties the hybrid. Its meta-model gives CatBoost most of the weight (0.78). The gain is small because the base models' errors are correlated.
- **No model clearly overfits.** Every ensemble has a train-test gap of only 0.07–0.08.

The full interpretation is in section 18 of the notebook.

## Project files
```
ML_assignment.ipynb     data cleaning, EDA, all models, comparison, interpretation
app.py                  Streamlit dashboard
models/                 stacking_model.pkl (RF + XGBoost + CatBoost), metadata.json
data/clean_houses.csv   cleaned listings used by the dashboard charts
requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The models were trained with Python 3.14 and scikit-learn 1.9.1. Use the same versions so the pickles load.

# InsurPrime: Can You Guess the Insurance Premium?
Crédit Agricole Assurances Data Challenge ([source](https://challengedata.ens.fr/participants/challenges/161/))

We present our solution to the challenge offered by Crédit Agricole Assurances, aiming to predict the pure fire premium for agricultural insurance contracts. The premium is decomposed into claim frequency, average claim severity, and exposure. 
We offer three modeling approaches to this challenge:
- **Baseline Generalized Linear Model (GLM)**
- **Histogram-based Gradient Boosting Regressor (HGBR)**
- **CatBoost regressor** 

We incorporate the year as an exposure offset in the frequency model. Model performance is evaluated using Root Mean Square Error (RMSE) on the predicted charge.

## Files Overview:
- **EDA.ipynb** : Exploratory Data Analysis and data preprocessing.
- **feature_engineering.ipynb** : Feature engineering pipeline and baseline GLM model.
- **data_preprocessing.py** and **feature_eng.py** : Data preprocessing the feature engineering functions, respectively. The functions can be imported from these files and used across models.
- **hgbr.ipynb** : Histogram-based Gradient Boosting Regressor implementation, including hyperparameter tuning.
- **catboost.ipynb** : Catboost implementation, including hyperparameter tuning.
- **InsurPrime.pdf** : Detailed report.

## Performance:

| Model    | GLM      | HGBR     | CatBoost 
|----------|----------|----------|----------
| **RMSE** |  5603.60 | 5598.75  | 5599.58
| **Time** |   1.6s   | 2m 21s   | 37m 30s

## Build the Project:
The dataset for this project can be found ([here](https://challengedata.ens.fr/participants/challenges/161/)). 

To run the project:
- Place the dataset in a folder named `Data/` at the root of the repository.
- Create a `Submissions/` folder to store generated submission files.
- Run the notebooks in the order: `EDA.ipynb` → `feature_engineering.ipynb` → `model notebooks`.



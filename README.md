# Term Deposit Predictor

Streamlit app for my Data Science mini project. It predicts whether a bank customer will subscribe to a term deposit, using a soft-voting ensemble of AdaBoost, Gradient Boosting and XGBoost trained on the UCI Bank Marketing dataset.

**Pages**
- **Home**: project summary and pipeline
- **Explore the data**: class balance, subscription rate by any feature, numeric distributions
- **Model performance**: metrics table, ROC curves, confusion matrix with an adjustable threshold, feature importance
- **Predict a customer**: enter a customer's details and get a live prediction from all four models

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud

1. Create a new GitHub repository (for example `term-deposit-predictor`) and upload `app.py`, `requirements.txt` and this README.
2. Also upload `bank-additional-full.csv` (about 5 MB). The app can download it from UCI itself, but keeping it in the repo makes startup reliable.
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, pick the repository, branch `main` and main file `app.py`, then **Deploy**.
4. The first load trains the models (about a minute). After that they stay cached.

Data: Moro, S., Cortez, P., & Rita, P. (2014). UCI Machine Learning Repository, Bank Marketing.

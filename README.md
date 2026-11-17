Credit Card Fraud Detection

This is a simple university project about machine learning. I use real data from Kaggle and compare two algorithms to see which one finds credit card fraud better.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

What I did:
- Random Forest: tested with cross-validation, 6 folds
- Logistic Regression: tested with a simple split, 80% train and 20% test
- Then I compared the results of both models

Project structure:
- main.py: run this file
- ui/dashboard.py: simple window to see the results
- models/random_forest.py: the Random Forest model
- models/logistic_regression.py: the Logistic Regression model
- utils/data_loader.py: loads the data
- utils/evaluation.py: calculates the metrics
- requirements.txt: the libraries to install
- README.md: this file

How I compare the models:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Training time
- Prediction time

How to run:
- Install the libraries: pip install -r requirements.txt
- Start the project: python main.py
- The data is downloaded automatically the first time, so it can take a little while

Results:
At the end, two CSV files are saved:
- random_forest_6fold_results.csv: results of each fold
- model_comparison.csv: final comparison between the two models
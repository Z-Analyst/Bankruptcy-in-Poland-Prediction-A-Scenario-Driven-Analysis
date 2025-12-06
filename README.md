#WQU Project
# Bankruptcy-in-Poland-Prediction-A-Scenario-Driven-Analysis
💡 Project Goal
Developed and deployed Ensemble Models (Random Forest and Gradient Boosting Trees) to predict corporate bankruptcy in Poland. The primary objective was to overcome severe class imbalance (Baseline Accuracy: 0.9519) by optimizing the model's output for two distinct, high-stakes business scenarios, showcasing the critical trade-off between Recall and Precision.

📊 Key Results: Scenario-Driven EvaluationThe final Random Forest model demonstrates the necessity of setting the prediction threshold based on the cost of error in the real world.

Scenario,Business Objective,Metric Prioritized,Final Score,Threshold
1. Regulatory Agency,"Minimize the cost of False Negatives (missed bankruptcies, costing €50,000).",Recall,0.76,0.50
2. Private Equity Firm,"Minimize the cost of False Positives (bad investment/purchase, costing €250 million).",Precision,0.76,0.80

Overall Achievement: Successfully identified the model and threshold that balance stakeholder risk profiles, moving beyond simple Accuracy metrics.

🛠️ Methodology and Techniques
1. Data Engineering & Preparation
Source: Financial data from the Emerging Markets Information Service (EMIS).

File Handling: Demonstrated proficiency in managing various data formats, including compressed files (gzip), JSON, and model serialization (pickle).

Key Features: Predictive power was driven by complex financial ratios like profit on operating activities / financial expenses and gross profit (in 3 years) / total assets. Features were preprocessed using SimpleImputer.

2. Addressing Imbalanced Data
Technique: Used Random Oversampling (RandomOverSampler from imblearn) on the training data to balance the class distribution, ensuring the model could learn the patterns of the minority (bankrupt) class.

Evaluation: Focused metrics on the minority class (Precision, Recall, F1-Score) using the Confusion Matrix to properly assess performance.

3. Model Building and Optimization
Models: Compared Random Forest Classifier and Gradient Boosting Trees for superior non-linear modeling capability.

Tuning: Employed GridSearchCV to optimize hyperparameters for the chosen ensemble model, specifically targeting an optimal balance between Precision and Recall.

📈 Most Insightful Visualization
The Confusion Matrix Display is the most vital plot, as it directly illustrates the impact of adjusting the prediction threshold to meet the demands of Scenario 1 (high Recall) versus Scenario 2 (high Precision).

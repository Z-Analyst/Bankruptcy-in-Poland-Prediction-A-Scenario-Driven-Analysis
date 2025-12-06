# Ensemble Model:Random Forest Model

import gzip
import json
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np

# --- 1. CONFIGURATION (Edit These Values) ---
DATA_PATH = "data/financial_data.json.gz"  # <--- CUSTOMIZE: Your data file path
TARGET_COL = "bankrupt"
FEATURES = [
    'profit on operating activities / financial expenses', 
    'gross profit (in 3 years) / total assets', 
    '(gross profit + depreciation) / sales',
    # <--- ADD ALL YOUR FINAL FEATURE NAMES HERE
]
# Thresholds based on your scenario analysis
RECALL_THRESHOLD = 0.50
PRECISION_THRESHOLD = 0.80

# --- 2. DATA LOADING ---
def load_data(data_path):
    """Loads data from compressed JSON file."""
    print(f"Loading data from {data_path}...")
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def split_and_resample(X, y):
    """Splits data and applies Random Oversampling to the training set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Applying Random Oversampling to the training data...")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    print(f"Original training shape: {Counter(y_train)}")
    print(f"Resampled training shape: {Counter(y_train_resampled)}")
    
    return X_train_resampled, X_test, y_train_resampled, y_test

# --- 3. MODELING & TUNING ---
def create_model_pipeline():
    """Defines the preprocessing and Random Forest classifier pipeline."""
    # Use make_pipeline for simplicity with no complex transformers
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'), # Handle missing financial data
        RandomForestClassifier(random_state=42, class_weight='balanced')
        # NOTE: Class weight can be used instead of resampling, but you used both!
    )
    return pipeline

def tune_and_fit_model(pipeline, X_train, y_train):
    """Placeholder for Hyperparameter Tuning (using a simple RF for demo)."""
    
    # <--- CUSTOMIZE: Use your specific Grid Search parameters here
    param_grid = {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_depth': [5, 10, 15] 
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='recall', # Optimize for the minority class early
        cv=3, 
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Fit the final model with best parameters
    final_model = grid_search.best_estimator_.fit(X_train, y_train)
    
    return final_model

# --- 4. SCENARIO-BASED EVALUATION ---
def evaluate_scenarios(model, X_test, y_test):
    """Evaluates performance using different prediction thresholds."""
    
    # Get probability scores for the positive class (Bankruptcy = 1)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results = {}
    
    # SCENARIO 1: Regulatory Agency (Optimize for RECALL)
    y_pred_r = (y_proba >= RECALL_THRESHOLD).astype(int)
    results['Recall'] = recall_score(y_test, y_pred_r)
    results['Recall_CM'] = confusion_matrix(y_test, y_pred_r)
    
    # SCENARIO 2: Private Equity Firm (Optimize for PRECISION)
    y_pred_p = (y_proba >= PRECISION_THRESHOLD).astype(int)
    results['Precision'] = precision_score(y_test, y_pred_p, zero_division=0)
    results['Precision_CM'] = confusion_matrix(y_test, y_pred_p)
    
    return results

def plot_confusion_matrix(cm, title, filename):
    """Plots and saves the Confusion Matrix."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Bankrupt', 'Bankrupt'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(f'assets/{filename}')
    plt.show()

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    data = load_data(DATA_PATH)
    
    # Use 1 for 'bankrupt' and 0 for 'not bankrupt'
    X = data[FEATURES]
    y = data[TARGET_COL]

    # Split and Oversample Training Data
    X_train_resampled, X_test, y_train_resampled, y_test = split_and_resample(X, y)
    
    # Train/Tune Model
    final_rf_model = tune_and_fit_model(create_model_pipeline(), X_train_resampled, y_train_resampled)

    # Evaluate Scenarios
    scenario_results = evaluate_scenarios(final_rf_model, X_test, y_test)
    
    print("\n--- Final Scenario Results ---")
    print(f"Baseline Accuracy: {1 - y.mean():.4f}") # Majority Class
    print(f"Scenario 1 (Recall optimized at {RECALL_THRESHOLD}): {scenario_results['Recall']:.4f}")
    print(f"Scenario 2 (Precision optimized at {PRECISION_THRESHOLD}): {scenario_results['Precision']:.4f}")

    # Visualize Results
    plot_confusion_matrix(
        scenario_results['Recall_CM'],
        f"Confusion Matrix (Scenario 1: Recall {scenario_results['Recall']:.2f} at T={RECALL_THRESHOLD})",
        "cm_scenario1_recall.png"
    )

    plot_confusion_matrix(
        scenario_results['Precision_CM'],
        f"Confusion Matrix (Scenario 2: Precision {scenario_results['Precision']:.2f} at T={PRECISION_THRESHOLD})",
        "cm_scenario2_precision.png"
    )

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Import your custom visualization functions
from visuals import plot_target_distribution, plot_importance

def preprocess_data(df):
    """
    Cleans features without dropping rows. 
    Crucial for preventing 'n_samples=0' errors.
    """
    # Fill missing values with median/constants
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    
    # Fix outlier error codes (96, 98) in delinquency columns
    cols_to_fix = [
        'NumberOfTime30-59DaysPastDueNotWorse', 
        'NumberOfTime60-89DaysPastDueNotWorse', 
        'NumberOfTimes90DaysLate'
    ]
    for col in cols_to_fix:
        # We cap these at the median or a reasonable max to remove noise
        df.loc[df[col] >= 90, col] = df[col].median()
        
    return df

# 1. Load Data
# Ensure cs-training.csv is in your project folder
train_df = pd.read_csv('data\cs-training.csv', index_col=0)

# 2. Handle Target NaNs 
# We only drop if the label itself is missing. 
# This preserves the dataset for the split.
train_df = train_df.dropna(subset=['SeriousDlqin2yrs'])

# 3. Handle Feature NaNs (Imputation)
train_df = preprocess_data(train_df)

# 4. Debug Check
print(f"Total samples available for training: {len(train_df)}")

# 5. Visual: Distribution (Run before training)
plot_target_distribution(train_df, 'SeriousDlqin2yrs')

# 6. Prepare for Split
X = train_df.drop('SeriousDlqin2yrs', axis=1)
y = train_df['SeriousDlqin2yrs']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Model Training
# scale_pos_weight=14 handles the ~1:14 ratio of defaults
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=14,
    eval_metric='auc',
    random_state=42
)

model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=50
)

# 8. Visual: Importance (Run after training)
plot_importance(model, X.columns)

# 9. Final Metric
val_preds = model.predict_proba(X_val)[:, 1]
print(f"\nFinal Validation AUC Score: {roc_auc_score(y_val, val_preds):.4f}")
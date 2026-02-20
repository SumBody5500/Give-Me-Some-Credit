import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb

def plot_target_distribution(df, target_col):
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title('Distribution of Credit Defaults')
    plt.xlabel('Defaulted (1) vs. Stayed Current (0)')
    plt.ylabel('Count')
    
    total = len(df[target_col])
    for p in plt.gca().patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        plt.gca().annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()), 
                           ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('target_imbalance.png')
    plt.show()

def plot_importance(model, features):
   
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    importances.plot(kind='barh', color='skyblue')
    plt.title('XGBoost Feature Importance')
    plt.xlabel('F-Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
# Give-Me-Some-Credit
Picking up on a classical financial stats/data-science model from kaggle titled "Give Me Some Credit" 
Credit Risk Prediction: Model Analysis
1. Why AUC over Accuracy?
In this dataset, only ~6.6% of individuals experienced financial distress (the "Positive" class).

_target_imbalance.png_

The Flaw of Accuracy: If a model predicted that nobody would default, it would achieve 93.4% accuracy while being completely useless for risk management.

The Power of AUC: The Area Under the ROC Curve (AUC) measures the model's ability to distinguish between classes. It evaluates how well the model ranks a "defaulter" higher than a "non-defaulter," regardless of the decision threshold. This is crucial in finance, where we need to rank borrowers by risk level.

2. Business Impact: False Negatives vs. False Positives
In credit scoring, errors are not created equal:

False Positive (Type I Error): Predicting a good borrower will default. The cost is the lost interest/opportunity cost of a potential customer.

False Negative (Type II Error): Predicting a high-risk borrower will stay current. The cost is the total loss of the principal loan amount.

Conclusion: Missing a default (False Negative) is significantly more expensive than turning away a good customer. I addressed this by using scale_pos_weight in XGBoost, which penalizes the model more heavily for missing a default.

3. Key Findings from Feature Importance
The model identified RevolvingUtilizationOfUnsecuredLines (the ratio of credit card balances to total credit limits) as the strongest predictor of default. This aligns with financial theory: as a borrower "maxes out" their available credit, their risk of financial distress increases exponentially. Other critical features included the number of times a borrower was 90+ days late, suggesting that historical delinquency is a strong predictor of future failure.

_feature_importance.png_

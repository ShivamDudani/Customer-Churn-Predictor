# Customer Churn Predictor

An end-to-end machine learning system that identifies telecom 
customers at risk of churning before they leave.

## Live Demo
[Launch App](https://huggingface.co/spaces/ShivamDudani/Customer-Churn-Predictor)

## Project Overview
Trained on 7,043 customers across 27 features, the model catches 
73.5% of actual churners — enabling proactive retention campaigns.

## Results
| Model               | AUC-ROC | Recall | Missed Churners |
|---------------------|---------|--------|-----------------|
| Logistic Regression | 0.8323  | 0.7353 | 99  ✅ champion |
| XGBoost Tuned       | 0.8327  | 0.6595 | 127             |
| XGBoost Default     | 0.8244  | 0.6542 | 129             |
| Random Forest       | 0.8206  | 0.6300 | 138             |

## Key Findings (SHAP Analysis)
- MonthlyCharges is the #1 churn driver
- Fiber optic customers churn more despite premium product
- Online Security + Tech Support act as retention anchors
- Month-to-month contracts are highest risk segment
- New customers (low tenure) churn far more than long-term ones

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, imbalanced-learn
- SHAP for explainability
- Streamlit for deployment
- Hugging Face Spaces for hosting

## Project Structure

# Boston Housing Price Prediction Using Machine Learning

This project applies multiple machine learning regression techniques—Decision Tree, Random Forest, and Ridge Regression (implemented from scratch)—to predict housing prices using the Boston Housing dataset. The aim is to evaluate and compare model performance using standard metrics and gain insights into the predictive importance of housing features.

## 📁 Contents

- `notebook.ipynb` – Main code implementation: data preprocessing, EDA, model training, and evaluation.
- `Supervised Learning and Model Evaluation report.docx` – Technical report covering supervised learning theory, model descriptions, and results.
- `ridge_regression.py` – Custom implementation of Ridge Regression from scratch (if modularized).
- `plots/` – Residual and prediction plots for visual evaluation (if applicable).

## 📊 Dataset

- Dataset used: [Boston Housing Dataset](http://lib.stat.cmu.edu/datasets/boston)
- Features: 13 predictors (e.g., CRIM, RM, LSTAT, etc.)
- Target: MEDV (Median value of owner-occupied homes in $1000s)

## 🔍 Models Used

1. **Decision Tree Regressor**  
   - Depth-limited to avoid overfitting  
   - Interpretable model with moderate accuracy

2. **Random Forest Regressor**  
   - Ensemble of decision trees  
   - Highest performance across all metrics

3. **Ridge Regression (from scratch)**  
   - Linear model with L2 regularization  
   - Serves as a baseline for model comparison

## 🧪 Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 🏁 Final Results

| Model             | MSE     | RMSE    | MAE     | R²     |
|------------------|---------|---------|---------|--------|
| Decision Tree     | 16.77   | 4.09    | 3.18    | 0.771  |
| Random Forest     | 7.91    | 2.81    | 2.04    | 0.892  |
| Ridge Regression  | 24.31   | 4.93    | 3.19    | 0.668  |

## 🛠️ Technologies Used

- Python (NumPy, Pandas, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook
- Git & GitHub

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/boston-housing-regression-ml.git

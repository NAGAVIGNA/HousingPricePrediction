# Housing Price Prediction Projects

## Overview
This repository contains machine learning projects for predicting housing prices using the **Boston**, **California**, and **Ames** Housing datasets. The projects implement and compare four regression models: Decision Tree, Random Forest, Gradient Boosting, and a custom Ridge Regression. The analyses include data preprocessing, exploratory data analysis (EDA), model training, evaluation, and feature importance analysis, with a focus on capturing non-linear relationships and temporal trends.

## Datasets
- **Boston Housing**: 506 records, 13 features (e.g., `RM`, `LSTAT`), target: `MEDV` (median house value, capped at $50,000).
- Source: Scrapped Data from (https://lib.stat.cmu.edu/datasets/boston)
- **California Housing**: 20,640 records, 8 numerical and 1 categorical feature (`ocean_proximity`), target: `median_house_value` (capped at $500,000).
- Source: Download via kagglehub ( https://www.kaggle.com/camnugent/california-housing-prices)
- **Ames Housing**: 2,930 records, 19 features selected from 80 based on correlation, target: `SalePrice`.
- Source: Download from OpenIntro (https://www.openintro.org/book/statdata/?data=ames)

## Methodology
1. **Preprocessing**:
   - Boston: Log transformation of skewed features (`CRIM`, `ZN`, `TAX`, `LSTAT`), dropped `CHAS` and `RAD`, standardized for Ridge Regression.
   - California: Median imputation for `total_bedrooms`, feature engineering (`rooms_per_household`, etc.), dropped redundant features, one-hot encoded `ocean_proximity`, RobustScaler.
   - Ames: Log transformation of `SalePrice`, median/mode imputation, one-hot encoding, feature selection (19 features), RobustScaler for Ridge Regression.
2. **EDA**: Histograms, scatterplots, correlation matrices, and temporal analysis (Ames only).
3. **Models**: Decision Tree (`max_depth=5/6`), Random Forest (100 estimators), Gradient Boosting (100/250 estimators, `learning_rate=0.1`), custom Ridge Regression (λ=1.0/100.0).
4. **Evaluation**: Metrics include MSE, RMSE, MAE, and R².
5. **Feature Importance**: Gini importance for tree-based models, absolute coefficients for Ridge Regression.

## Model Performance
| Dataset    | Model             | RMSE      | MAE       | R²     |
|------------|-------------------|-----------|-----------|--------|
| Boston     | Ridge Regression  | 4.93      | 3.19      | 0.6685 |
|            | Decision Tree     | 4.09      | 3.18      | 0.7714 |
|            | Random Forest     | 2.81      | 2.04      | 0.8921 |
|            | Gradient Boosting | 2.52      | 1.93      | 0.9134 |
| California | Ridge Regression  | 76392.66  | 53206.85  | 0.5384 |
|            | Decision Tree     | 68445.17  | 47977.34  | 0.6295 |
|            | Random Forest     | 49427.64  | 31897.26  | 0.8068 |
|            | Gradient Boosting | 49412.30  | 31766.12  | 0.8069 |
| Ames       | Ridge Regression  | 46740.14  | 23977.66  | 0.6443 |
|            | Decision Tree     | 45290.58  | 31337.98  | 0.6660 |
|            | Random Forest     | 41894.77  | 19231.80  | 0.7142 |
|            | Gradient Boosting | 46412.97  | 20145.21  | 0.6493 |

## Dependencies
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `kagglehub` (for California dataset)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebooks or Python scripts for each dataset:
   ```bash
   jupyter notebook boston_housing.ipynb
   jupyter notebook california_housing.ipynb
   jupyter notebook ames_housing.ipynb
   ```
5. Datasets are included in the repository or can be downloaded via `kagglehub` for California Housing.

## Repository Structure
- `boston_housing.ipynb`: Boston Housing dataset analysis and modeling.
- `california_housing.ipynb`: California Housing dataset analysis and modeling.
- `ames_housing.ipynb`: Ames Housing dataset analysis and modeling.
- `data/`: Directory containing datasets (or instructions to download).
- `requirements.txt`: List of required Python libraries.
- `figures/`: Generated plots (e.g., feature importance, RMSE comparisons).

## Key Findings
- **Gradient Boosting** and **Random Forest** consistently outperformed Decision Tree and Ridge Regression across all datasets, with Gradient Boosting leading in Boston (R² = 0.9134) and California (R² = 0.8069), and Random Forest in Ames (R² = 0.7142).
- Key features: `RM` and `LSTAT` (Boston), `median_income` and `ocean_proximity` (California), `OverallQual` and `GrLivArea` (Ames).
- Temporal analysis (Ames) revealed a 2009 price dip and seasonal trends (peaks in May–July).

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or contributions, please open an issue or contact [your email or GitHub handle].

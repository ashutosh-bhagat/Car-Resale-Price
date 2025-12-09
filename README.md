# Car Resale Price Prediction

## Overview

Car resale prices fluctuate widely depending on usage, features, and economic trends. This project explores historical listings data and trains a Random Forest regression model to predict log-transformed prices. The repository contains exploratory notebooks, cleaned datasets, and machine learning experiments focused on practical resale valuation insights.

## Data Sources

- `used_car.csv`: Raw scraped listings with inconsistent fields.
- `used_car_cleaned.csv`: Curated dataset after handling duplicates, missing values, and standardizing categorical labels.
- `data.csv`: Feature-engineered table used directly by `model.ipynb` (includes `log_price`).

## Repository Structure

```
├── app.ipynb            # Streamlit / app exploration notebook
├── model.ipynb          # RandomForest modeling workflow
├── data.csv             # Training-ready features
├── used_car*.csv        # Raw and cleaned datasets
├── README.md            # Project documentation
└── vir/                 # Python virtual environment (not versioned ideally)
```

## Environment & Dependencies

- Python 3.10+
- Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Modeling: `scikit-learn` (RandomForestRegressor, GradientBoosting, GridSearchCV, pipelines)
- Visualization: `matplotlib`, `seaborn`

Activate the provided virtual environment (PowerShell):

```powershell
cd "c:/Users/Ashu/Desktop/Data/Car Resale Price"
vir\Scripts\Activate.ps1
```

Then install/update dependencies as needed:

```powershell
pip install -r requirements.txt  # create this file if missing
```

## Modeling Workflow (model.ipynb)

1. Import libraries and load `data.csv`.
2. Inspect schema via `.info()` / `.describe()` and preview rows.
3. Drop high-cardinality categorical columns (`car_make`, `car_model`, `car_spec`, `Year`) for the baseline feature set and define target `log_price`.
4. Split into train/test sets (80/20).
5. Run `GridSearchCV` over Random Forest hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`).
6. Fit the tuned model, generate predictions, and compute MAE/MSE/RMSE and $R^2$.
7. Validate robustness using `cross_val_score` with a `StandardScaler`+RandomForest pipeline.
8. Plot learning curves, residuals, and actual-vs-predicted charts to diagnose variance and bias.

## Key Results

- Cross-validated RMSE remains stable across folds, indicating low variance.
- Learning curves show convergence between training and validation RMSE, suggesting the model generalizes well within the available feature space.
- Residual plots highlight scatter around zero, with mild underestimation for top-end prices, hinting at possible feature enrichment opportunities.

## How to Reproduce

1. Ensure datasets reside at the workspace root (already tracked).
2. Launch Jupyter: `jupyter notebook model.ipynb`.
3. Execute cells sequentially to train/tune the Random Forest model and visualize diagnostics.
4. (Optional) Experiment with additional regressors (Gradient Boosting, XGBoost) or add engineered features (vehicle age, kilometers-per-year) to compare performance.

## Future Enhancements

- Integrate categorical encoders (target encoding, one-hot) to leverage `car_make/model/spec` fields instead of dropping them.
- Log and compare multiple model families using MLflow or Weights & Biases.
- Deploy the trained model through a Streamlit dashboard (`app.ipynb`) for interactive valuation queries.
- Automate data refreshes and add unit tests for preprocessing utilities.

## License

Specify licensing terms here (e.g., MIT) once finalized.

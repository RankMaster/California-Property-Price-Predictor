# 🏡 California Property Price Predictor

> A machine learning pipeline that predicts California district housing prices using regression models — from simple income-based forecasting to a tuned Random Forest with 77% explained variance.

![Build Status](https://img.shields.io/github/actions/workflow/status/your-username/property-price-predictor/ci.yml?branch=main&style=flat-square)
![License](https://img.shields.io/github/license/your-username/property-price-predictor?style=flat-square)
![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-yellow?style=flat-square)

---

## ✨ Features

- **Simple Linear Regression** — Baseline model using `median_income` as the sole predictor to establish a performance floor.
- **Multiple Linear Regression (MLR)** — Full-feature regression incorporating income, location, room counts, and ocean proximity for improved accuracy.
- **Random Forest Regressor** — Ensemble model applied to both simple and multiple feature sets, with significant accuracy gains over linear baselines.
- **Hyperparameter Tuning** — `RandomizedSearchCV` and `GridSearchCV` used to optimize Random Forest parameters (`n_estimators`, `max_depth`, `min_samples_leaf`).
- **Comprehensive EDA** — Pairplots, correlation heatmaps, distribution histograms, boxplots, and geospatial scatter plots for deep feature understanding.
- **One-Hot Encoding** — Categorical `ocean_proximity` feature transformed for use in regression models.
- **Interactive Prediction** — Command-line prompt to enter a median income value and receive an instant house price estimate.
- **Model Comparison** — Side-by-side MSE, RMSE, and R² evaluation across all models for informed selection.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (LinearRegression, RandomForestRegressor, GridSearchCV, RandomizedSearchCV) |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |
| **Dataset** | California Housing Dataset (`Property Price Prediction.csv`) |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.9 or higher
- `pip` (Python package manager)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RankMaster/California-Property-Price-Predictor.git
   cd property-price-predictor
   ```

2. **Create and activate a virtual environment** *(recommended)*
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place the dataset** in the project root directory:
   ```
   property-price-predictor/
   └── Property Price Prediction.csv   ← put it here
   ```

### Running the Project

Launch Jupyter Notebook and open the main notebook:

```bash
jupyter notebook ML_Property_Price_Prediction.ipynb
```

Run all cells sequentially (`Kernel → Restart & Run All`) to reproduce the full analysis, model training, and evaluation.

---

## 📖 Usage Example

### Predicting House Price from Median Income

After training the simple linear regression model, you can get an instant prediction interactively:

```python
# The notebook will prompt you:
Enter The Medium Income Value: 5.5

# Output:
The Predicted House value is $243,871.45
```

### Using the Best Model (Tuned Random Forest on MLR features) Programmatically

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv("Property Price Prediction.csv").dropna()
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True, dtype=int)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

# Train best model
best_params = {'n_estimators': 300, 'max_depth': 25, 'min_samples_leaf': 1}
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Predict
sample = X_test.iloc[[0]]
print(f"Predicted Price: ${model.predict(sample)[0]:,.2f}")
```

**Model Performance Summary:**

| Model | R² Score | Notes |
|---|---|---|
| Simple Linear Regression | ~0.47 | Income only |
| Random Forest (Simple) | ~0.47 | No improvement over LR with 1 feature |
| Multiple Linear Regression | Higher | All features |
| **Random Forest (MLR) — Tuned** | **~0.77** | ✅ Best model |

---

## 🗺️ Roadmap

- [ ] **Feature Engineering** — Add interaction terms (e.g., `rooms_per_household`, `bedrooms_per_room`) to improve model signal.
- [ ] **Gradient Boosting Models** — Benchmark XGBoost and LightGBM against the current Random Forest baseline.
- [ ] **Web Application Deployment** — Build a lightweight Flask/FastAPI REST API and a Streamlit front-end for real-time price predictions.
- [ ] **Cross-Validation Pipeline** — Replace the single train/test split with a full k-fold CV pipeline for more robust performance estimates.

---

## 🤝 Contributing

Contributions are warmly welcome! Whether it's fixing a bug, improving the EDA, or adding a new model — all PRs are appreciated.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please make sure your code follows existing style conventions and includes relevant comments. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

> Built with 🐍 Python and ❤️ for data science. If this project helped you, consider giving it a ⭐ on GitHub!

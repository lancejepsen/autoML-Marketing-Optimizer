import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# ============================
# Hard-coded CSV values
# ============================

marketing_spend = np.array([
    10000, 12000, 5000, 8000, 9000, 15000,
    16000, 17000, 14000, 11000, 7000
])

product_sales = np.array([
    40000, 18000, 13000, 21000, 5600, 23000,
    31000, 29000, 34000, 27000, 19000
])

# Reshape for sklearn
X = marketing_spend.reshape(-1, 1)
y = product_sales

# ============================
# TRAIN / TEST SPLIT
# ============================

X_train, X_test = X[:8], X[8:]
y_train, y_test = y[:8], y[8:]

# ============================
# LINEAR REGRESSION (No tuning)
# ============================

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)


# ============================
# OPTUNA TUNING: CATBOOST
# ============================

def objective_catboost(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "loss_function": "RMSE",
        "verbose": False
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


print("\n🔧 Running Optuna hyperparameter tuning for CatBoost (50 trials)...")
study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(objective_catboost, n_trials=50)
best_params_cat = study_cat.best_params
print("Best CatBoost Params:", best_params_cat)

# Train tuned CatBoost model
cat_model = CatBoostRegressor(
    **best_params_cat,
    loss_function="RMSE",
    verbose=False
)
cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_test)


# ============================
# OPTUNA TUNING: RANDOM FOREST
# ============================

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": 42
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


print("\n🔧 Running Optuna hyperparameter tuning for Random Forest (50 trials)...")
study_rf = optuna.create_study(direction="minimize")
study_rf.optimize(objective_rf, n_trials=50)
best_params_rf = study_rf.best_params
print("Best Random Forest Params:", best_params_rf)

# Train tuned Random Forest model
rf_model = RandomForestRegressor(**best_params_rf, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# ============================
# METRICS FUNCTION
# ============================

def metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


lin_metrics = metrics(y_test, y_pred_lin)
cat_metrics = metrics(y_test, y_pred_cat)
rf_metrics = metrics(y_test, y_pred_rf)

# ============================
# PRINT METRICS
# ============================

print("\n=== ACCURACY COMPARISON (RMSE is the official accuracy score) ===")

print("\nLinear Regression:")
for k, v in lin_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTuned CatBoost:")
for k, v in cat_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTuned Random Forest:")
for k, v in rf_metrics.items():
    print(f"{k}: {v:.4f}")

# ============================
# PERFECTLY ALIGNED PREDICTION TABLE
# ============================

print("\n=== TEST SET PREDICTION TABLE WITH ERRORS ===")

# Column widths
col = 12

print(
    f"{'Marketing':>{col}} | {'Actual':>{col}} | "
    f"{'Lin Pred':>{col}} | {'Cat Pred':>{col}} | {'RF Pred':>{col}} | "
    f"{'Lin Err':>{col}} | {'Cat Err':>{col}} | {'RF Err':>{col}}"
)
print("-" * (col * 8 + 21))

for mkt, actual, plin, pcat, prf in zip(
        X_test.flatten(), y_test, y_pred_lin, y_pred_cat, y_pred_rf):
    err_lin = actual - plin
    err_cat = actual - pcat
    err_rf = actual - prf

    print(
        f"{mkt:>{col},.0f} | "
        f"{actual:>{col},.0f} | "
        f"{plin:>{col},.0f} | "
        f"{pcat:>{col},.0f} | "
        f"{prf:>{col},.0f} | "
        f"{err_lin:>{col},.0f} | "
        f"{err_cat:>{col},.0f} | "
        f"{err_rf:>{col},.0f}"
    )

# ============================
# DETERMINE MOST ACCURATE MODEL (RMSE)
# ============================

rmse_scores = {
    "Linear Regression": lin_metrics["RMSE"],
    "CatBoost": cat_metrics["RMSE"],
    "Random Forest": rf_metrics["RMSE"]
}

best_model = min(rmse_scores, key=rmse_scores.get)

print("\n=== MOST ACCURATE MODEL (Lowest RMSE) ===")
print(f"Linear Regression RMSE: {lin_metrics['RMSE']:.4f}")
print(f"CatBoost RMSE:          {cat_metrics['RMSE']:.4f}")
print(f"Random Forest RMSE:     {rf_metrics['RMSE']:.4f}")
print(f"\n🎯 BEST MODEL FOR THIS DATASET: **{best_model}** 🎯")

# ============================
# PLOTTING ALL MODELS
# ============================

x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.figure(figsize=(12, 7))
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="orange", label="Testing Data")

plt.plot(x_range, lin_model.predict(x_range), color="red", label="Linear Regression")
plt.plot(x_range, cat_model.predict(x_range), color="green", label="Tuned CatBoost")
plt.plot(x_range, rf_model.predict(x_range), color="purple", label="Tuned Random Forest")

plt.title("Model Comparison: Linear vs CatBoost vs Random Forest (Tuned with Optuna)")
plt.xlabel("Marketing Spend ($)")
plt.ylabel("Product Sales ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import scipy.stats as stats
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import time
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

print("Bayesian Optimization For Hyperparameter tuning of Gradient Boosting Machines\n")

print("Loading California Housing dataset:")
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Target range: ${y.min():.0f} - ${y.max():.0f}")

df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y

print("\nDataset description:")
print(df.describe())

print("\nPreprocessing data:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

print("\nTraining baseline model with default parameters:")
baseline_model = XGBRegressor(random_state=42, n_jobs=-1)
baseline_model.fit(X_train_scaled, y_train)
y_pred_baseline = baseline_model.predict(X_test_scaled)
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_r2 = r2_score(y_test, y_pred_baseline)

print(f"Baseline Model RMSE: {baseline_rmse:.4f}")
print(f"Baseline Model R²: {baseline_r2:.4f}")

print("\nPerforming Random Search:")
xgb = XGBRegressor(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': stats.randint(100, 600),
    'max_depth': stats.randint(3, 10),
    'learning_rate': stats.uniform(0.01, 0.3),
    'subsample': stats.uniform(0.5, 0.5),
    'colsample_bytree': stats.uniform(0.5, 0.5),
    'reg_alpha': stats.uniform(0, 1),
    'reg_lambda': stats.uniform(0, 1)
}

start_time = time.time()
random_search = RandomizedSearchCV(
    xgb, param_dist, n_iter=50, 
    scoring='neg_mean_squared_error', 
    cv=5, n_jobs=-1, random_state=42,
    verbose=0
)
random_search.fit(X_train_scaled, y_train)
random_search_time = time.time() - start_time

print(f"Random Search completed in {random_search_time:.2f} seconds")
print(f"Random Search Best CV Score (RMSE): {np.sqrt(-random_search.best_score_):.4f}")
print("Random Search Best Params:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

rs_best_model = XGBRegressor(**random_search.best_params_, random_state=42, n_jobs=-1)
rs_best_model.fit(X_train_scaled, y_train)
y_pred_rs = rs_best_model.predict(X_test_scaled)
rs_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rs))
rs_test_r2 = r2_score(y_test, y_pred_rs)

print("\nPerforming Bayesian Optimization:")

space = [
    Integer(100, 600, name='n_estimators'),
    Integer(3, 10, name='max_depth'),
    Real(0.01, 0.3, name='learning_rate'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
    Real(0, 1, name='reg_alpha'),
    Real(0, 1, name='reg_lambda')
]

@use_named_args(space)
def objective(**params):
    xgb = XGBRegressor(**params, random_state=42, n_jobs=-1)
    score = -np.mean(cross_val_score(
        xgb, X_train_scaled, y_train, 
        cv=5, n_jobs=-1, 
        scoring='neg_mean_squared_error'
    ))
    return score

start_time = time.time()
res_gp = gp_minimize(
    objective, space, n_calls=50, 
    random_state=42, verbose=False,
    n_initial_points=10
)
bo_time = time.time() - start_time

print(f"Bayesian Optimization completed in {bo_time:.2f} seconds")
print(f"Bayesian Optimization Best CV Score (RMSE): {res_gp.fun:.4f}")

best_params = {}
dim_names = [dim.name for dim in space]
for i, name in enumerate(dim_names):
    best_params[name] = res_gp.x[i]
    print(f"  {name}: {res_gp.x[i]}")

bo_best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
bo_best_model.fit(X_train_scaled, y_train)
y_pred_bo = bo_best_model.predict(X_test_scaled)
bo_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_bo))
bo_test_r2 = r2_score(y_test, y_pred_bo)

print("\nResults Comparison")

results_comparison = {
    'Model': ['Baseline (Default)', 'Random Search', 'Bayesian Optimization'],
    'Test RMSE': [baseline_rmse, rs_test_rmse, bo_test_rmse],
    'Test R²': [baseline_r2, rs_test_r2, bo_test_r2],
    'CV Score (RMSE)': [
        'N/A', 
        f"{np.sqrt(-random_search.best_score_):.4f}", 
        f"{res_gp.fun:.4f}"
    ],
    'Time (s)': ['N/A', f"{random_search_time:.2f}", f"{bo_time:.2f}"],
    '# Evaluations': [1, 50, 50]
}

results_df = pd.DataFrame(results_comparison)
print(results_df.to_string(index=False))

print("\nGenerating convergence plots:")

random_scores = random_search.cv_results_['mean_test_score']
random_best_scores = [np.max(random_scores[:i+1]) for i in range(len(random_scores))]
random_rmse_progress = np.sqrt(-np.array(random_best_scores))

bo_rmse_progress = [np.min(res_gp.func_vals[:i+1]) for i in range(len(res_gp.func_vals))]

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(range(1, 51), random_rmse_progress, 'r-', label='Random Search', linewidth=2)
plt.plot(range(1, 51), bo_rmse_progress, 'b-', label='Bayesian Optimization', linewidth=2)
plt.xlabel('Number of Iterations')
plt.ylabel('Best RMSE Score')
plt.title('Convergence Plot: Hyperparameter Optimization Methods')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
models = ['Baseline', 'Random Search', 'Bayesian Optimization']
test_rmse = [baseline_rmse, rs_test_rmse, bo_test_rmse]
colors = ['gray', 'red', 'blue']
bars = plt.bar(models, test_rmse, color=colors, alpha=0.7)
plt.ylabel('Test RMSE')
plt.title('Test Set Performance Comparison')
for bar, value in zip(bars, test_rmse):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom')

plt.subplot(2, 2, 3)
feature_importance = bo_best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])
plt.xlabel('Feature Importance Score')
plt.title('Feature Importance (BO-tuned Model)')
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_bo, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'BO Model: Actual vs Predicted\nR² = {bo_test_r2:.4f}')
plt.tight_layout()
plt.show()
print("\nImprovement Analysis:")
improvement_rs = ((baseline_rmse - rs_test_rmse) / baseline_rmse) * 100
improvement_bo = ((baseline_rmse - bo_test_rmse) / baseline_rmse) * 100
improvement_bo_vs_rs = ((rs_test_rmse - bo_test_rmse) / rs_test_rmse) * 100

print(f"Random Search improvement over baseline: {improvement_rs:.2f}%")
print(f"Bayesian Optimization improvement over baseline: {improvement_bo:.2f}%")
print(f"Bayesian Optimization improvement over Random Search: {improvement_bo_vs_rs:.2f}%")

print("\nBest Hyperparameters Comparison:")

print("Random Search Best Parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

print("\nBayesian Optimization Best Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")


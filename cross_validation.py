from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
import numpy as np

# ================================
# Pipeline (Avoids Data Leakage)
# ================================
pipeline = Pipeline([
    ('scaler', PowerTransformer()),
    ('model', stacking)
])

# ================================
# 5-Fold Cross Validation
# ================================
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    'MAE': 'neg_mean_absolute_error',
    'RMSE': 'neg_root_mean_squared_error',
    'R2': 'r2'
}

results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

# Convert negative scores
mae = -results['test_MAE']
rmse = -results['test_RMSE']
r2 = results['test_R2']

print("\n========== 5-Fold Cross Validation ==========")

print(f"MAE  : {mae.mean():.4f} ± {mae.std():.4f}")
print(f"RMSE : {rmse.mean():.4f} ± {rmse.std():.4f}")
print(f"R²   : {r2.mean():.4f} ± {r2.std():.4f}")

print("\nFold-wise Results")
for i in range(5):
    print(f"Fold {i+1}: MAE={mae[i]:.4f}, RMSE={rmse[i]:.4f}, R²={r2[i]:.4f}")

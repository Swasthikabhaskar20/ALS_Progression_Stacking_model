from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from scipy.stats import ttest_rel



# ================================
# Stacking Pipeline
# ================================
stack_pipeline = Pipeline([
    ('scaler', PowerTransformer()),
    ('model', stacking)
])

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



# Cross-validation for Stacking
stack_results = cross_validate(
    stack_pipeline,
    X,
    y,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

rf_mae = -rf_results['test_score']
stack_mae = -stack_results['test_score']

# ================================
# Paired t-test
# ================================
t_stat, p_value = ttest_rel(rf_mae, stack_mae)

print("\n========== Paired t-test ==========")

print("Stacking MAE      :", stack_mae)

print(f"\nt-statistic : {t_stat:.4f}")
print(f"p-value     : {p_value:.6f}")

if p_value < 0.05:
    print("\n The proposed Stacking model performs significantly better (p < 0.05).")
else:
    print("\n No statistically significant difference between the models (p ≥ 0.05).")

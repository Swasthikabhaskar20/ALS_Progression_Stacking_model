from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from scipy.stats import ttest_rel
import pandas as pd

# =========================
# Repeated Cross Validation
# =========================
cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

# =========================
# Pipelines
# =========================
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', rf)
])

gb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', gb)
])

lasso_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', lasso)
])

mlp_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', mlp)
])

stack_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', stacking)
])

models = {
    "Random Forest": rf_pipe,
    "Gradient Boosting": gb_pipe,
    "Lasso": lasso_pipe,
    "MLP": mlp_pipe
}

# =========================
# Proposed Model Scores
# =========================
stack_scores = cross_validate(
    stack_pipe,
    X,
    y,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

stack_mae = -stack_scores['test_score']

results = []

print("\n==============================")
print("Paired t-test Results")
print("==============================")

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    model_mae = -scores['test_score']

    t_stat, p_value = ttest_rel(model_mae, stack_mae)

    results.append({
        "Model": name,
        "Baseline MAE": model_mae.mean(),
        "Stacking MAE": stack_mae.mean(),
        "t-statistic": t_stat,
        "p-value": p_value,
        "Significant": "Yes" if p_value < 0.05 else "No"
    })

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv("paired_t_test_results.csv", index=False)

print("\nResults saved as paired_t_test_results.csv")

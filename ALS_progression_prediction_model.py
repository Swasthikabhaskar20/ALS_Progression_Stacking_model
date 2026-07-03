import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor

# Specify the correct file path
file_path = 'C:/Users/swast/119.csv'

# Try reading with 'latin1' encoding
df = pd.read_csv(file_path, encoding='latin1')

# ------------------------------------------------------------
#  Data Preprocessing
# ------------------------------------------------------------
df = df.copy()
df.replace('-', np.nan, inplace=True)

num_cols = [
    'Age at Symptons Onset', 'OnsetYR', 'LNA_YR',
    'ALSFRS-R Baseline', 'ALSFRS-R Latest', 'Diff'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Lenght_Diag_LNA'] = pd.to_numeric(df['Lenght_Diag_LNA'], errors='coerce')

# ------------------------------------------------------------
#  Target Variable Construction
# ------------------------------------------------------------
df = df[df['Lenght_Diag_LNA'] > 0.5]
df['Progression_Rate'] = df['Diff'] / df['Lenght_Diag_LNA']
df = df[(df['Progression_Rate'] > -20) & (df['Progression_Rate'] < 5)]
df.dropna(subset=['Progression_Rate'], inplace=True)

# ------------------------------------------------------------
#  Patient Selection
# ------------------------------------------------------------
df = df.sort_values(by='LNA_YR', ascending=False)
df = df.drop_duplicates(subset='SubjectUID', keep='first')

# ------------------------------------------------------------
# Feature Selection
# ------------------------------------------------------------
y = df['Progression_Rate']

X = df.drop(columns=[
    'Progression_Rate',
    'Diff',
    'Lenght_Diag_LNA',
    'ALSFRS-R Latest',
    'Age_at_Death',
    'SubjectUID',
    'med',
    'med_revised',
    'Diagdt'
], errors='ignore')

# ------------------------------------------------------------
#  Missing Value Handling
# ------------------------------------------------------------
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = X[col].fillna(X[col].median())

for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# ------------------------------------------------------------
# One-Hot Encoding
# ------------------------------------------------------------
X = pd.get_dummies(X, drop_first=True)

# ------------------------------------------------------------
#  Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ------------------------------------------------------------
#  Feature Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
#  Base Models
# ------------------------------------------------------------
lasso = Lasso(alpha=0.01)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=800,
    random_state=42
)

# ------------------------------------------------------------
#  Stacking Ensemble
# ------------------------------------------------------------
stacking = StackingRegressor(
    estimators=[
        ('lasso', lasso),
        ('rf', rf),
        ('gb', gb)
    ],
    final_estimator=mlp,
    cv=5,
    passthrough=True,
    n_jobs=-1
)

# ------------------------------------------------------------
#  Train Model
# ------------------------------------------------------------
stacking.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
#  Prediction
# ------------------------------------------------------------
y_pred = stacking.predict(X_test_scaled)

# ------------------------------------------------------------
#  Evaluation
# ------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")




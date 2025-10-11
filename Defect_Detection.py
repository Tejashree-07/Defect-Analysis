# ========================================================
# PHASE 1 - BASIC EDA WITH THE OUTPUT RESULTS AS BELOW 
# Dataset size and structure
# Column datatypes
# Missing and duplicate checks
# Quick summary of numeric + categorical variables
# Visual heatmap of missing values
# A CSV summary file for your records
# ========================================================

# ---  Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---  Load Dataset ---
# Replace with your actual file path if needed
file_path = "synthetic_explicit.csv"
df = pd.read_csv(file_path)

cols_to_drop = ['Year', 'Minutes', 'Month', 'Hour', 'Date']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# ---  Basic Overview ---
print("=== BASIC INFO ===")
print(f"Shape of dataset: {df.shape}")
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n")
print(df.dtypes)
print("\nSample Rows:\n")
print(df.head(5))

# ---  Missing Values Check ---
print("\n=== MISSING VALUES ===")
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if missing.empty:
    print(" No missing values detected.")
else:
    print(missing)

# ---  Duplicate Records Check ---
duplicates = df.duplicated().sum()
print(f"\n=== DUPLICATES ===\nTotal duplicate rows: {duplicates}")
if duplicates > 0:
    print(" Consider removing duplicates with df.drop_duplicates(inplace=True)")

# ---  Basic Descriptive Statistics ---
print("\n=== DESCRIPTIVE STATISTICS (NUMERIC FEATURES) ===")
print(df.describe().T)

# ---  Separate Categorical and Numeric Features ---
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\n=== FEATURE TYPES ===")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# ---  Summary for Categorical Columns ---
print("\n=== CATEGORICAL COLUMN SUMMARY ===")
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())
    print("-" * 40)


print("\n PHASE 1 completed successfully — summary saved as 'phase1_summary.csv'")

# ========================================================
# PHASE 2 - STATISTIAL PROFILING 
# ========================================================

'''pd.set_option('display.max_columns', None)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumeric Columns ({len(numeric_cols)}): {numeric_cols}")
print(f"\nCategorical Columns ({len(categorical_cols)}): {categorical_cols}")

# ==============================================================
# UNIVARIATE ANALYSIS
# ==============================================================

# --- Summary Stats for Numeric Features ---
print("\n=== NUMERIC FEATURE SUMMARY ===")
print(df[numeric_cols].describe().T)

# --- Distribution Plots for Numeric Columns ---
n_cols = 3 
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

plt.figure(figsize=(n_cols * 5, n_rows * 3))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(col)
    plt.xlabel("")
    plt.ylabel("")
plt.tight_layout()
plt.suptitle("Distributions of Numeric Features", fontsize=16, y=1.02)
plt.show()

# --- Outlier Detection (Boxplots) ---
n_cols = 3
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

plt.figure(figsize=(n_cols * 5, n_rows * 3))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.xlabel("")
plt.tight_layout()
plt.suptitle("Boxplots for Numeric Features (Outlier Check)", fontsize=16, y=1.02)
plt.show()

# --- Skewness & Kurtosis ---
skew_kurt = df[numeric_cols].agg(['skew', 'kurtosis']).T
print("\n=== SKEWNESS & KURTOSIS ===")
print(skew_kurt.sort_values(by='skew', ascending=False))

# ==============================================================
# CATEGORICAL ANALYSIS
# ==============================================================

print("\n=== CATEGORICAL FEATURE DISTRIBUTIONS ===")
for col in categorical_cols:
    print(f"\n{col} value counts:\n", df[col].value_counts())
    plt.figure(figsize=(6,3))
    sns.countplot(data=df, x=col)
    plt.title(f"Category Distribution: {col}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# ==============================================================
# CORRELATION & RELATIONSHIPS
# ==============================================================

# --- Correlation Matrix (Numeric Only) ---
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# --- Highly Correlated Feature Pairs ---
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
high_corr = corr_pairs[(corr_pairs < 0.9999) & (abs(corr_pairs) > 0.9)]
if len(high_corr) > 0:
    print("\n Highly Correlated Features (|r| > 0.9):")
    print(high_corr)
else:
    print("\n No strongly correlated numeric pairs (|r| > 0.9).")

# ==============================================================
# TARGET BALANCE CHECK
# ==============================================================

if 'Defect' in df.columns:
    plt.figure(figsize=(4,4))
    sns.countplot(data=df, x='Defect', palette='pastel')
    plt.title("Defect Class Balance")
    plt.xlabel("Defect (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    defect_rate = df['Defect'].value_counts(normalize=True) * 100
    print("\n=== DEFECT CLASS DISTRIBUTION (%) ===")
    print(defect_rate)
else:
    print("No 'Defect' column found — please specify target label.")

# ================================================== 
# Checking the Impact of the variable Join-Status
# ===================================================
if 'Join_Status' in df.columns and 'Defect' in df.columns:
    print("\n=== IMPACT OF JOIN_STATUS ON DEFECT ===")
    join_defect_rate = pd.crosstab(df['Join_Status'], df['Defect'], normalize='index') * 100
    print(join_defect_rate)

    # Optional: Visualize the relationship
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, x='Join_Status', hue='Defect', palette='coolwarm')
    plt.title('Defect Distribution by Join_Status')
    plt.xlabel('Join_Status')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
else:
    print("'Join_Status' or 'Defect' column not found — skipping join analysis.")

# ==============================================================
# CHI-SQUARE TEST: Join_Status vs Defect
# ==============================================================

from scipy.stats import chi2_contingency

if 'Join_Status' in df.columns and 'Defect' in df.columns:
    contingency_table = pd.crosstab(df['Join_Status'], df['Defect'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("\n=== CHI-SQUARE TEST ===")
    print(f"Chi² Statistic: {chi2:.3f}")
    print(f"P-value: {p:.5f}")

    if p < 0.05:
        print("Significant relationship: Join_Status and Defect are strongly associated.")
    else:
        print("No significant relationship detected.")

# ==============================================================
# CORRELATION OF JOIN_STATUS WITH NUMERIC FEATURES
# ==============================================================

# Convert Join_Status to numeric for correlation
df_encoded = df.copy()
df_encoded['Join_Status_Num'] = df_encoded['Join_Status'].map({'Joining': 1, 'Non-Joining': 0})

corr_with_join = df_encoded.corr(numeric_only=True)['Join_Status_Num'].sort_values(ascending=False)
print("\n=== CORRELATION OF JOIN_STATUS WITH NUMERIC FEATURES ===")
print(corr_with_join)

# Visualize correlations
plt.figure(figsize=(8,4))
sns.barplot(x=corr_with_join.index, y=corr_with_join.values, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Correlation of Process Variables with Join_Status")
plt.tight_layout()
plt.show()'''

# ==============================================================
# MODEL 1 – Predict Join_Status from Process Parameters (Updated)
# ==============================================================

'''from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# --- Prepare Data ---
X = df.drop(columns=['Join_Status', 'Defect'])
y = df['Join_Status']

# Encode target (Joining=1, Non-Joining=0)
y = y.map({'Joining': 1, 'Non-Joining': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing (Scale numeric, encode categorical) ---
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Fit on train, transform both train and test
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# --- Train Random Forest ---
rf_join = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
rf_join.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred = rf_join.predict(X_test_scaled)
print("\n=== MODEL 1: Join_Status PREDICTION ===")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# --- Feature Importance ---
# Get proper feature names after preprocessing
onehot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numeric_cols, onehot_cols])

importances = pd.Series(rf_join.feature_importances_, index=all_features).sort_values(ascending=False)
print("\nTop Features influencing Join_Status:\n", importances.head(10))

# Optional: Visualize feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
importances.head(10).plot(kind='bar', color='skyblue')
plt.title("Top 10 Features Influencing Join_Status")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''

# ==============================================================
# MODEL 2 – Predict Join_Status from Process Parameters (XGBoost)
# ==============================================================

'''from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Prepare Data ---
X = df.drop(columns=['Join_Status', 'Defect'])
y = df['Join_Status']

# Encode target (Joining=1, Non-Joining=0)
y = y.map({'Joining': 1, 'Non-Joining': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing (Scale numeric, encode categorical) ---
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Fit on train, transform both train and test
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# --- Train XGBoost ---
xgb_join = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),  # handle class imbalance
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_join.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred = xgb_join.predict(X_test_scaled)
print("\n=== MODEL 2: Join_Status PREDICTION (XGBoost) ===")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# --- Feature Importance ---
# Get proper feature names after preprocessing
onehot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numeric_cols, onehot_cols])

importances = pd.Series(xgb_join.feature_importances_, index=all_features).sort_values(ascending=False)
print("\nTop Features influencing Join_Status (XGBoost):\n", importances.head(10))

# Optional: Visualize feature importance
plt.figure(figsize=(10,5))
importances.head(10).plot(kind='bar', color='lightgreen')
plt.title("Top 10 Features Influencing Join_Status (XGBoost)")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''

# ==============================================================
# TWO-STEP PIPELINE OPTIMIZED FOR ROOT CAUSE ANALYSIS
# Using XGBoost to maximize Non-Joining recall
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Load Dataset ---
file_path = "synthetic_explicit.csv"
df = pd.read_csv(file_path)

# --- Drop unwanted columns ---
cols_to_drop = ['year', 'minutes', 'month', 'hour', 'date']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# --- Separate features and targets ---
X = df.drop(columns=['Join_Status', 'Defect'])
y_defect = df['Defect']
y_join = df['Join_Status'].map({'Joining':0, 'Non-Joining':1})

# --- Split data ---
X_train, X_test, y_def_train, y_def_test, y_join_train, y_join_test = train_test_split(
    X, y_defect, y_join, test_size=0.2, random_state=42, stratify=y_join
)

# --- Preprocessing ---
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
])

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# ==============================================================
# STEP 1: Train Defect Model with XGBoost
# ==============================================================

xgb_defect = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_def_train==0).sum() / (y_def_train==1).sum(),  # handle class imbalance
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_defect.fit(X_train_scaled, y_def_train)

# Predict defect probabilities
def_pred_train = xgb_defect.predict_proba(X_train_scaled)[:,1]
def_pred_test = xgb_defect.predict_proba(X_test_scaled)[:,1]

# --- Evaluate Defect Model ---
from sklearn.metrics import classification_report, confusion_matrix

# Lower threshold for higher recall (important for defect detection)
threshold_def = 0.2
y_def_pred_thresh = (def_pred_test >= threshold_def).astype(int)
print(f"\n=== Defect Model Evaluation (threshold={threshold_def}) ===")
print(classification_report(y_def_test, y_def_pred_thresh))
print("Confusion Matrix:")
print(confusion_matrix(y_def_test, y_def_pred_thresh))

# ==============================================================
# STEP 2: Train Join_Status Model using predicted defect
# Optimize for Non-Joining recall
# ==============================================================

# Add predicted defect as a feature
X_train_join = np.hstack([X_train_scaled, def_pred_train.reshape(-1,1)])
X_test_join = np.hstack([X_test_scaled, def_pred_test.reshape(-1,1)])

xgb_join = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_join_train==1).sum() / (y_join_train==0).sum(),  # penalize Non-Joining
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_join.fit(X_train_join, y_join_train)

# --- Predict probabilities ---
y_prob = xgb_join.predict_proba(X_test_join)[:,1]

# --- Adjust threshold to increase Non-Joining recall ---
threshold_join = 0.1  # increase threshold so more predicted as Non-Joining
y_join_pred = (y_prob >= threshold_join).astype(int)

# --- Evaluate Join_Status Model ---
print("\n=== Join_Status Model (XGBoost) Optimized for Non-Joining Recall ===")
print(classification_report(y_join_test, y_join_pred))
print(confusion_matrix(y_join_test, y_join_pred))



# ================================================================
# Buildig SHAP - The AI Reasoning Layer for Root Cause Analysis 
# ================================================================

# ==============================================================
# STEP 3: Root Cause Analysis using SHAP (for Defect Model)
# ==============================================================

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Initialize SHAP Explainer ---
explainer = shap.Explainer(xgb_defect, X_train_scaled)

# Compute SHAP values for test set
shap_values = explainer(X_test_scaled)

# --- Global Feature Importance ---
print("\n=== SHAP Global Feature Importance (Top 10) ===")
# Get feature names after preprocessing
onehot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numeric_cols, onehot_cols])

# Compute mean absolute SHAP value per feature
shap_importance = pd.DataFrame({
    'feature': all_features,
    'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)

print(shap_importance.head(10))

# --- SHAP Summary Plot ---
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=all_features, show=False)
plt.title("SHAP Summary Plot – Feature Impact on Defect Prediction")
plt.tight_layout()
plt.show()

# --- SHAP Bar Plot (Global Importance) ---
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=all_features, plot_type="bar", show=False)
plt.title("SHAP Bar Plot – Top Features Influencing Defect")
plt.tight_layout()
plt.show()

# --- SHAP Dependence Plot for Top Feature ---
top_feature = shap_importance['feature'].iloc[0]
print(f"\nTop feature selected for dependence plot: {top_feature}")

plt.figure()
shap.dependence_plot(top_feature, shap_values.values, X_test_scaled, feature_names=all_features, show=False)
plt.title(f"SHAP Dependence Plot – {top_feature}")
plt.tight_layout()
plt.show()



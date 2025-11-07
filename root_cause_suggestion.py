# ======================================================
# AGENTIC AI SYSTEM FOR AUTOMATION OF DEFECT ANALYSIS 
# ======================================================

# full pipeline with orchestrator and RCA only 
import os
import numpy as np
import pandas as pd
from collections import Counter
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier

# -----------------------------
# Configuration / file paths
# -----------------------------
file_path = "synthetic_explicit.csv"
output_log = "agentic_decision_log_rca.csv"

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(file_path)

# Drop irrelevant columns
cols_to_drop = ['year', 'minutes', 'month', 'hour', 'date']
cols_lower = {c.lower(): c for c in df.columns}
for drop in cols_to_drop:
    if drop in cols_lower:
        df.drop(columns=[cols_lower[drop]], inplace=True)

# Required columns check
required = ['Join_Status', 'Defect']
for r in required:
    if r not in df.columns:
        raise ValueError(f"Required column '{r}' not found in dataset.")

# -----------------------------
# Feature and target separation
# -----------------------------
X = df.drop(columns=['Join_Status', 'Defect'])
y_defect = df['Defect'].astype(int)
y_join = df['Join_Status'].map({'Joining': 0, 'Non-Joining': 1})
if y_join.isnull().any():
    y_join = pd.factorize(df['Join_Status'])[0]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_def_train, y_def_test, y_join_train, y_join_test = train_test_split(
    X, y_defect, y_join, test_size=0.2, random_state=42, stratify=y_join
)

# -----------------------------
# Preprocessing
# -----------------------------
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
], remainder='drop')

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

onehot_cols = []
if categorical_cols:
    onehot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features_full = np.concatenate([numeric_cols, onehot_cols]) if len(onehot_cols) > 0 else np.array(numeric_cols)
if X_train_scaled.shape[1] != len(all_features_full):
    all_features_full = np.array([f"f_{i}" for i in range(X_train_scaled.shape[1])])

# -----------------------------
# Defect prediction model
# -----------------------------
scale_pos = (y_def_train == 0).sum() / max((y_def_train == 1).sum(), 1)
xgb_defect = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
xgb_defect.fit(X_train_scaled, y_def_train)

def_pred_train = xgb_defect.predict_proba(X_train_scaled)[:, 1]
def_pred_test = xgb_defect.predict_proba(X_test_scaled)[:, 1]

threshold_def = 0.2
y_def_pred_thresh = (def_pred_test >= threshold_def).astype(int)
print("\n=== Defect Model Evaluation ===")
print(classification_report(y_def_test, y_def_pred_thresh))
print(confusion_matrix(y_def_test, y_def_pred_thresh))

# -----------------------------
# Join status model
# -----------------------------
X_train_join = np.hstack([X_train_scaled, def_pred_train.reshape(-1, 1)])
X_test_join = np.hstack([X_test_scaled, def_pred_test.reshape(-1, 1)])
scale_pos_join = (y_join_train == 1).sum() / max((y_join_train == 0).sum(), 1)

xgb_join = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_join,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
xgb_join.fit(X_train_join, y_join_train)

y_prob = xgb_join.predict_proba(X_test_join)[:, 1]
threshold_join = 0.1
y_join_pred = (y_prob >= threshold_join).astype(int)

print("\n=== Join_Status Model Evaluation ===")
print(classification_report(y_join_test, y_join_pred))
print(confusion_matrix(y_join_test, y_join_pred))

# ---------------------------------------------------
# SHAP explainer for automating root cause analysis 
# ---------------------------------------------------
X_train_df = pd.DataFrame(X_train_scaled, columns=all_features_full)
X_test_df = pd.DataFrame(X_test_scaled, columns=all_features_full)
explainer = shap.Explainer(xgb_defect, X_train_df)
shap_vals = explainer(X_test_df)

shap_values_array = shap_vals.values
if shap_values_array.ndim == 3 and shap_values_array.shape[1] == 1:
    shap_values_array = shap_values_array[:, 0, :]

drop_feats = ['hour', 'minutes', 'month', 'year', 'date']
mask = np.array([not any(df in feat.lower() for df in drop_feats) for feat in all_features_full])
shap_values_filtered = shap_values_array[:, mask]

# -----------------------------
# Process map for RCA
# -----------------------------
process_map = {
    "lithography": ["stage_alignment", "uv", "vibration", "rotation", "particle", "chamber_temp"],
    "deposition": ["gas_flow", "vacuum", "rf", "chamber_temp", "particle"],
    "etching": ["rf", "vacuum", "chamber_temp", "etch_depth", "particle", "gas_flow"]
}

def map_rcas_to_processes(rca_list, process_map):
    rca_list = [str(r).strip() for r in rca_list if str(r).strip() != ""]
    counts = Counter()
    matched_details = []
    for rca in rca_list:
        rca_low = rca.lower().replace('-', '').replace(' ', '')
        matched = set()
        for proc, keywords in process_map.items():
            for kw in keywords:
                if kw in rca_low:
                    matched.add(proc)
                    break
        if not matched:
            matched.add("unknown")
        for m in matched:
            counts[m] += 1
        matched_details.append((rca, list(matched)))

    proc_counts = {p: counts[p] for p in process_map.keys()}
    total_hits = sum(proc_counts.values())
    if total_hits == 0:
        primary = ["unknown"]
        secondary = []
    else:
        sorted_procs = sorted(proc_counts.items(), key=lambda x: x[1], reverse=True)
        max_count = sorted_procs[0][1]
        primary = [p for p, c in sorted_procs if c == max_count and c > 0]
        secondaries = [p for p, c in sorted_procs if c < max_count and c > 0]
        secondary = secondaries if secondaries else []

    return primary, secondary, matched_details

# -----------------------------
# Decision logic
# -----------------------------
def agent_decision(defect_prob, join_prob, shap_top_features, threshold_def=0.2, threshold_join=0.1):
    if defect_prob >= threshold_def and join_prob >= threshold_join:
        status = "High Risk Defect & Join Failure"
        action = "Hold for Root Cause Analysis"
    elif defect_prob >= threshold_def:
        status = "Potential Defect"
        action = "Flag for Reinspection"
    elif join_prob >= threshold_join:
        status = "Joining Risk"
        action = "Check bonding or alignment process"
    else:
        status = "Stable"
        action = "Proceed to next step"
    return {"status": status, "action": action, "root_causes": shap_top_features}

# -----------------------------
# RCA-only orchestrator
# -----------------------------
def orchestrator_rca(X_input_df, model_def, model_join, explainer_obj, preproc, thresholds):
    X_scaled = preproc.transform(X_input_df)
    X_df = pd.DataFrame(X_scaled, columns=all_features_full)

    defect_prob = model_def.predict_proba(X_df.values)[:, 1]
    X_join = np.hstack([X_df.values, defect_prob.reshape(-1, 1)])
    join_prob = model_join.predict_proba(X_join)[:, 1]

    shap_vals_local = explainer_obj(X_df)
    shap_arr = shap_vals_local.values
    if shap_arr.ndim == 3 and shap_arr.shape[1] == 1:
        shap_arr = shap_arr[:, 0, :]

    mask_new = np.array([not any(df in feat.lower() for df in drop_feats) for feat in all_features_full])
    shap_filtered = shap_arr[:, mask_new]

    logs = []
    for i in range(len(X_input_df)):
        top_idx = np.argsort(np.abs(shap_filtered[i]))[::-1]
        feats_masked = all_features_full[mask_new]
        ranked_feats = [feats_masked[j] for j in top_idx]
        filtered_feats = [f for f in ranked_feats if not any(ex in f.lower() for ex in ['minute', 'hour', 'month', 'year', 'date'])]

        is_defect = defect_prob[i] >= thresholds['defect']
        is_join = join_prob[i] >= thresholds['join']
        top_features = filtered_feats[:3] if (is_defect or is_join) else []

        dec = agent_decision(defect_prob[i], join_prob[i], top_features,
                             threshold_def=thresholds['defect'], threshold_join=thresholds['join'])

        if dec['status'] != "Stable" and top_features:
            primary, secondary, _ = map_rcas_to_processes(top_features, process_map)
            logs.append({
                "Sample_ID": int(X_input_df.index[i]),
                "Decision": dec['status'],
                "Action": dec['action'],
                "Root_Causes": "; ".join(top_features),
                "Predicted_Process": ", ".join(primary),
                "Secondary_Process": ", ".join(secondary) if secondary else ""
            })

    return pd.DataFrame(logs)

# -----------------------------
# Run orchestrator (RCA only)
# -----------------------------
thresholds_dict = {'defect': threshold_def, 'join': threshold_join}
log_df = orchestrator_rca(X_test, xgb_defect, xgb_join, explainer, preprocessor, thresholds_dict)
log_df.to_csv(output_log, index=False)
print(f"\nNumber of actionable wafers: {len(log_df)}")
print(f"Agentic RCA decision log saved as '{output_log}'")
print(log_df.head())

# -----------------------------
# Save models
# -----------------------------
joblib.dump(xgb_defect, "xgb_defect_model.pkl")
joblib.dump(xgb_join, "xgb_join_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
joblib.dump(all_features_full, "all_features_full.pkl")


print(" Trained models, preprocessor, and SHAP explainer saved successfully.")

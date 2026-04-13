# ============================================================
# SEMICONDUCTOR DEFECT AGENT + ANALYTICS CHARTS
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
import shap
import json
import matplotlib.pyplot as plt
import seaborn as sns

from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# Configuration
# -----------------------------
file_path = "synthetic_explicit.csv"
output_log = "agentic_rca_with_llm.csv"

# -----------------------------
# API Keys
# -----------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key:
    raise EnvironmentError("Set OPENAI_API_KEY")

if not pinecone_api_key:
    raise EnvironmentError("Set PINECONE_API_KEY")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

# -----------------------------
# Pinecone Setup
# -----------------------------
INDEX_NAME = "wafer-rca-memory"
DIMENSION = 1536

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(INDEX_NAME)

# -----------------------------
# Embedding Function
# -----------------------------
def embed_text(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# -----------------------------
# Retrieve Similar Cases (RAG)
# -----------------------------
def retrieve_similar_cases(current_payload, top_k=3):

    query_embedding = embed_text(json.dumps(current_payload))

    try:
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="rca_memory"
        )

        retrieved = []

        for match in results["matches"]:
            retrieved.append(match["metadata"])

        return retrieved

    except Exception:
        return []


# -----------------------------
# LLM RCA
# -----------------------------
def call_llm_with_rag(def_prob, join_prob, features, similar_cases):

    prompt = f"""
    You are a semiconductor process engineer.

    CURRENT CASE:
    Defect Probability: {def_prob}
    Join Probability: {join_prob}
    Top Features: {features}

    SIMILAR PAST CASES:
    {json.dumps(similar_cases, indent=2)}

    Return ONLY valid JSON:
    {{
        "rca_summary": "...",
        "root_causes": ["...", "..."],
        "confidence": "low/medium/high",
        "recommended_tests": ["...", "..."]
    }}
    """

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_output = response.choices[0].message.content.strip()

        try:
            return json.loads(raw_output)

        except:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            return json.loads(raw_output[start:end])

    except Exception:

        return {
            "rca_summary": "LLM call failed",
            "root_causes": [],
            "confidence": "low",
            "recommended_tests": []
        }


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(file_path)

X = df.drop(columns=["Join_Status", "Defect"])
y_defect = df["Defect"].astype(int)
y_join = pd.factorize(df["Join_Status"])[0]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_def_train, y_def_test, y_join_train, y_join_test = train_test_split(
    X, y_defect, y_join, test_size=0.2, random_state=42
)

# -----------------------------
# Preprocessing
# -----------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
])

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

all_features = np.concatenate([
    numeric_cols,
    preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
])

# -----------------------------
# Train Models
# -----------------------------
xgb_defect = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    eval_metric="logloss"
)

xgb_defect.fit(X_train_scaled, y_def_train)

def_pred_train = xgb_defect.predict_proba(X_train_scaled)[:, 1]

X_train_join = np.hstack([
    X_train_scaled,
    def_pred_train.reshape(-1,1)
])

xgb_join = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    eval_metric="logloss"
)

xgb_join.fit(X_train_join, y_join_train)

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.Explainer(xgb_defect, X_train_scaled)

# ============================================================
# AGENT PLANNER
# ============================================================
def agent_planner(defect_prob, join_prob, top_features):
    """
    Returns a step-by-step plan for handling a high-risk wafer.
    """
    plan = ["analyze_features"]

    if defect_prob > 0.7:
        plan.append("retrieve_similar_cases")
    if defect_prob > 0.5 or join_prob > 0.5:
        plan.append("run_llm_rca")
    plan.append("recommend_tests")
    plan.append("store_case_memory")

    return plan


# ============================================================
# ORCHESTRATOR (Agentic Version - FIXED)
# ============================================================
def orchestrator(X_input):

    logs = []
    pinecone_vectors = []

    llm_calls = 0
    max_llm_calls = 5
    MAX_BATCH_SIZE = 50

    X_scaled = preprocessor.transform(X_input)
    defect_probs = xgb_defect.predict_proba(X_scaled)[:, 1]
    join_probs = xgb_join.predict_proba(
        np.hstack([X_scaled, defect_probs.reshape(-1,1)])
    )[:,1]
    shap_values = explainer(X_scaled).values
    high_risk = np.where(defect_probs >= np.percentile(defect_probs, 80))[0]

    for i in high_risk:

        top_idx = np.argsort(np.abs(shap_values[i]))[::-1][:3]
        top_features = [all_features[j] for j in top_idx]

        # CREATE PLAN
        plan = agent_planner(defect_probs[i], join_probs[i], top_features)
        similar_cases = []

        # INITIAL RCA (non-hardcoded base)
        rca = {
            "rca_summary": f"Top contributing features: {', '.join(top_features)}",
            "root_causes": top_features,
            "confidence": "medium",
            "recommended_tests": []
        }

        # EXECUTE PLAN
        for step in plan:

            if step == "analyze_features":
                pass

            elif step == "retrieve_similar_cases":
                similar_cases = retrieve_similar_cases({
                    "defect_prob": float(defect_probs[i]),
                    "join_prob": float(join_probs[i]),
                    "top_features": top_features
                })

            # LLM CALL (controlled + preserves output)
            elif (
                step == "run_llm_rca"
                and llm_calls < max_llm_calls
                and defect_probs[i] > 0.9   # cost control
            ):
                rca = call_llm_with_rag(
                    defect_probs[i],
                    join_probs[i],
                    top_features,
                    similar_cases[:1]   # reduce prompt size
                )
                llm_calls += 1

            # FIXED TEST LOGIC
            elif step == "recommend_tests":

                # If LLM already provided tests → KEEP THEM
                if rca.get("recommended_tests"):
                    pass

                # Otherwise generate dynamic tests (NO hardcoding)
                else:
                    tests = []

                    for f in top_features:
                        base = f.replace("_", " ")

                        tests.append(f"Validate stability of {base}")
                        tests.append(f"Check calibration related to {base}")
                        tests.append(f"Review process logs for {base}")

                    rca["recommended_tests"] = list(set(tests))[:5]

            elif step == "store_case_memory":
                embedding = embed_text(json.dumps({
                    "sample_id": int(i),
                    "defect_prob": float(defect_probs[i]),
                    "join_prob": float(join_probs[i]),
                    "top_features": top_features,
                    "rca_summary": rca.get("rca_summary",""),
                    "root_causes": rca.get("root_causes",[]),
                    "confidence": rca.get("confidence",""),
                    "recommended_tests": rca.get("recommended_tests",[])
                }))
                pinecone_vectors.append({
                    "id": f"sample-{i}",
                    "values": embedding,
                    "metadata": {
                        "defect_prob": float(defect_probs[i]),
                        "join_prob": float(join_probs[i])
                    }
                })

        # SAVE LOG
        flat_log = {
            "sample_id": int(i),
            "defect_prob": float(defect_probs[i]),
            "join_prob": float(join_probs[i]),
            "top_feature_1": top_features[0] if len(top_features) > 0 else "",
            "top_feature_2": top_features[1] if len(top_features) > 1 else "",
            "top_feature_3": top_features[2] if len(top_features) > 2 else "",
            "agent_plan": " -> ".join(plan),
            "rca_summary": rca.get("rca_summary",""),
            "root_causes": " | ".join(rca.get("root_causes",[])),
            "confidence": rca.get("confidence",""),
            "recommended_tests": " | ".join(rca.get("recommended_tests",[]))
        }
        logs.append(flat_log)

    # UPSERT TO PINECONE
    if pinecone_vectors:
        for j in range(0, len(pinecone_vectors), MAX_BATCH_SIZE):
            batch = pinecone_vectors[j:j+MAX_BATCH_SIZE]
            pinecone_index.upsert(vectors=batch, namespace="rca_memory")

    print(f"LLM Calls Used: {llm_calls}")
    return pd.DataFrame(logs)

# ============================================================
# ANALYTICS CHARTS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ------------------------------------------------------------
# Correlation Heatmap for All Features
# ------------------------------------------------------------
def correlation_heatmap(df):
    """
    Generates a correlation heatmap for process features only.
    Removes time-related columns like Year, Month, Date, Hour, Minute.
    """

    # Remove time columns
    remove_cols = ["Year", "Month", "Date", "Hour", "Minute"]

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Keep only process features
    process_features = [col for col in numeric_cols if col not in remove_cols]

    # Correlation matrix
    corr_matrix = df[process_features].corr()

    plt.figure(figsize=(10,8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar=True
    )

    plt.title("Process Feature Correlation Heatmap")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig("feature_correlation_heatmap.png")
    plt.close()

# ------------------------------------------------------------
# Main Chart Generator
# ------------------------------------------------------------
def generate_charts(process_df, result_df):
    """
    Generates all analytics charts:
    1. Defect probability histogram
    2. Process risk map (join vs defect)
    3. Defect density map
    4. Root cause Pareto chart
    5. Correlation heatmap of all numeric features
    """

    sns.set(style="whitegrid")

    # =========================================================
    # 1. Defect Probability Histogram
    # =========================================================
    plt.figure()
    sns.histplot(
        result_df["defect_prob"],  # predictions from orchestrator
        bins=20,
        kde=True
    )
    plt.title("Defect Probability Distribution")
    plt.xlabel("Predicted Defect Probability")
    plt.ylabel("Number of Samples")
    plt.savefig("defect_probability_histogram.png")
    plt.close()

    # =========================================================
    # 2. Process Risk Map (Join vs Defect)
    # =========================================================

    # Create risk category
    risk_level = []

    for d, j in zip(result_df["defect_prob"], result_df["join_prob"]):

        if d >= 0.8 and j >= 0.8:
            risk_level.append("High Risk")

        elif d >= 0.5 or j >= 0.5:
            risk_level.append("Medium Risk")

        else:
            risk_level.append("Low Risk")

    result_df["risk_level"] = risk_level


    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x=result_df["join_prob"],
        y=result_df["defect_prob"],
        hue=result_df["risk_level"],
        palette={
            "Low Risk":"green",
            "Medium Risk":"orange",
            "High Risk":"red"
        }
    )

    plt.axhline(0.5, linestyle="--")
    plt.axvline(0.5, linestyle="--")

    plt.title("Process Risk Map: Join vs Defect Probability")

    plt.xlabel("Join Probability")
    plt.ylabel("Defect Probability")

    plt.tight_layout()

    plt.savefig("join_vs_defect_risk_map.png")

    plt.close() 

    # =========================================================
    # 3. Defect Density Map
    # =========================================================
    plt.figure()
    sns.kdeplot(
        x=result_df["join_prob"],
        y=result_df["defect_prob"],
        fill=True
    )
    plt.title("Defect Density in Process Space")
    plt.xlabel("Join Probability")
    plt.ylabel("Defect Probability")
    plt.savefig("process_density_map.png")
    plt.close()

    # =========================================================
    # 4. Root Cause Pareto Chart
    # =========================================================

    root_counts = (
        result_df["root_causes"]
        .str.split("|")
        .explode()
        .str.strip()
    )

    # Remove time-based or irrelevant features
    remove_features = ["Minute", "Year", "Month", "Date", "Hour"]

    root_counts = root_counts[~root_counts.isin(remove_features)]

    # Count root cause occurrences
    root_counts = root_counts.value_counts().head(10)

    # Convert to percentage contribution
    percent_contribution = (root_counts / root_counts.sum()) * 100

    plt.figure(figsize=(10,6))

    sns.barplot(
        x=percent_contribution.index,
        y=percent_contribution.values
    )

    plt.xlabel("Process Feature")
    plt.ylabel("Contribution to Defects (%)")
    plt.title("Process Parameter Contribution to Defects")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig("root_cause_contribution_chart.png")

    plt.close()

    # =========================================================
    # 5. Feature Correlation Heatmap
    # =========================================================
    correlation_heatmap(process_df)  # uses original raw features

    print("All analytics charts generated successfully!")

# ============================================================
# RUN SYSTEM
# ============================================================

# Run orchestrator on test data
result_df = orchestrator(X_test)

# Save results to CSV
result_df.to_csv(output_log, index=False)

# Generate charts using both process_df and result_df
#    - process_df: raw input features (for heatmaps)
#    - result_df: predictions and RCA logs (for defect/join plots)
generate_charts(process_df=df, result_df=result_df)

# Save trained models and preprocessor
joblib.dump(xgb_defect, "xgb_defect_model.pkl")
joblib.dump(xgb_join, "xgb_join_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("SYSTEM EXECUTION COMPLETE")



# ============================================================
# SEMICONDUCTOR DEFECT ANALYSIS AGENT 
# ============================================================

"""
This module implements an agentic pipeline for real-time root cause analysis
(RCA) of semiconductor wafer defects. It combines:

  - XGBoost classifiers (defect probability and join status prediction)
  - SHAP explainability (identifying top contributing sensor features)
  - A CNN classifier (classifying wafer map surface defect patterns)
  - Physics-based pattern expectations (per fault type / process stage)
  - RAG memory via Pinecone (retrieving similar past defect cases)
  - GPT-4o-mini LLM (generating structured RCA reports)
  - Batch drift detection (monitoring defect rate trends across a lot)

Typical entry point:
  Run this file directly after configuring file paths and API keys.
  The DigitalTwinSimulator (digital_twin_simulator.py) streams batches
  of 50 simulated wafers into the orchestrator, which scores, analyses,
  and logs results to an Excel file.

Dependencies:
  pip install openai pinecone-client xgboost shap scikit-learn
              torch torchvision pillow pandas numpy matplotlib seaborn

Environment variables required:
  OPENAI_API_KEY    – OpenAI API key (for embeddings + LLM RCA)
  PINECONE_API_KEY  – Pinecone API key (for vector memory store)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from pinecone import Pinecone, ServerlessSpec

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = r"D:\MS\venv"

FILE_PATH   = os.path.join(BASE_DIR, "synthetic_explicit.csv")
OUTPUT_LOG  = os.path.join(BASE_DIR, "agentic_rca_with_llm.csv")
CNN_MODEL   = os.path.join(BASE_DIR, "wafer_cnn_model.pt")

IMG_SIZE    = 32
CNN_CLASSES = [
    "Center", "Donut", "Edge Local", "Edge Ring",
    "Local", "Scratch", "near full", "none", "random",
]


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------

openai_api_key   = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key:
    raise EnvironmentError("Environment variable OPENAI_API_KEY is not set.")
if not pinecone_api_key:
    raise EnvironmentError("Environment variable PINECONE_API_KEY is not set.")

client = OpenAI(api_key=openai_api_key)
pc     = Pinecone(api_key=pinecone_api_key)


# ---------------------------------------------------------------------------
# Pinecone vector store setup
# ---------------------------------------------------------------------------

INDEX_NAME = "wafer-rca-memory"
DIMENSION  = 1536   # text-embedding-3-small output dimension

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(INDEX_NAME)


# ---------------------------------------------------------------------------
# CNN Model – Wafer Map Pattern Classification
# ---------------------------------------------------------------------------

class WaferCNN(nn.Module):
    """
    Lightweight CNN that classifies a wafer map image into one of nine
    defect pattern categories (see CNN_CLASSES above).

    Architecture: three conv-BN-ReLU-MaxPool blocks → two FC layers.
    Input: 32×32 RGB image (normalised to [-1, 1]).
    Output: logits for each of the nine pattern classes.
    """

    def __init__(self, num_classes: int = 9):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Load the pre-trained CNN weights once at module startup.
DEVICE    = torch.device("cpu")
cnn_model = None

if os.path.exists(CNN_MODEL):
    cnn_model = WaferCNN(num_classes=len(CNN_CLASSES)).to(DEVICE)
    cnn_model.load_state_dict(torch.load(CNN_MODEL, map_location=DEVICE))
    cnn_model.eval()
    print(f"CNN model loaded from {CNN_MODEL}")
else:
    print(f"WARNING: CNN model not found at {CNN_MODEL} — image classification disabled.")

# Standard image pre-processing pipeline expected by WaferCNN.
cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ---------------------------------------------------------------------------
# CNN Inference
# ---------------------------------------------------------------------------

def classify_wafer_image(image_path: str) -> tuple[str, float, dict]:
    """
    Run the CNN on a wafer map image and return the predicted pattern.

    Parameters
    ----------
    image_path : str
        Absolute path to a .jpg / .png wafer map image.

    Returns
    -------
    pattern    : str   – predicted class label (e.g. "Edge Ring")
    confidence : float – softmax probability of the top class [0, 1]
    all_probs  : dict  – {class_label: probability} for all nine classes
    """
    if cnn_model is None or not image_path or not os.path.exists(image_path):
        return "unknown", 0.0, {}

    try:
        img    = Image.open(image_path).convert("RGB")
        tensor = cnn_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = cnn_model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().numpy()

        pred_idx   = int(probs.argmax())
        pattern    = CNN_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        all_probs  = {CNN_CLASSES[i]: float(probs[i]) for i in range(len(CNN_CLASSES))}

        return pattern, confidence, all_probs

    except Exception as exc:
        print(f"  CNN inference error: {exc}")
        return "unknown", 0.0, {}


# ---------------------------------------------------------------------------
# Physics–CNN Agreement Check
# ---------------------------------------------------------------------------

def check_agreement(
    physics_pattern: str,
    cnn_pattern: str,
    confidence: float,
    defect_prob: float = 0.0,
    fault_type: str = "none",
) -> tuple[str, str]:
    """
    Compare the physics-expected wafer map pattern with the CNN prediction.

    Three agreement levels are returned:
      "high"    – both sources agree on the same pattern
      "partial" – patterns differ but belong to a known related family
                  (e.g. both are edge-type or both are centre-type)
      "conflict" – patterns are incompatible; engineer review recommended

    A special "conflict" case is also flagged when XGBoost predicts a high
    defect probability but the CNN sees a clean wafer and no fault was logged
    — this typically indicates sensor drift or a multi-parameter interaction
    not captured by the single-fault mapping.

    Parameters
    ----------
    physics_pattern : str   – pattern expected from the fault-type lookup table
    cnn_pattern     : str   – pattern classified by the CNN from the wafer image
    confidence      : float – CNN confidence score
    defect_prob     : float – XGBoost defect probability for this wafer
    fault_type      : str   – fault label from the simulator ("none" if clean)

    Returns
    -------
    agreement : str  – "high" | "partial" | "conflict"
    note      : str  – human-readable explanation
    """
    # Unexplained defect: high XGBoost score but CNN and fault log show clean
    if cnn_pattern == "none" and fault_type == "none" and defect_prob > 0.7:
        return "conflict", (
            f"UNEXPLAINED: XGBoost defect_prob={defect_prob:.3f} "
            f"but CNN shows a clean wafer and no fault was logged — "
            f"possible sensor drift or multi-parameter interaction. "
            f"Manual sensor log review recommended."
        )

    if physics_pattern == cnn_pattern:
        return "high", f"Physics and CNN both indicate '{cnn_pattern}'."

    # Patterns from the same physical family are treated as partial agreement
    related_groups = [
        {"Edge Ring", "Edge Local"},
        {"Center", "Donut"},
        {"Local", "random"},
        {"near full", "random"},
    ]
    for group in related_groups:
        if physics_pattern in group and cnn_pattern in group:
            return "partial", (
                f"Physics predicted '{physics_pattern}', "
                f"CNN found '{cnn_pattern}' — related pattern family."
            )

    return "conflict", (
        f"Physics predicted '{physics_pattern}' but CNN found '{cnn_pattern}' "
        f"({confidence * 100:.0f}% confidence) — flag for engineer review."
    )


# ---------------------------------------------------------------------------
# Batch Drift Detection
# ---------------------------------------------------------------------------

def detect_batch_drift(
    defect_probs: np.ndarray,
    batch_faults: list | None = None,
) -> tuple[bool, str, str]:
    """
    Detect whether defect probability is systematically increasing across
    a batch of wafers by comparing the first half to the second half.

    Parameters
    ----------
    defect_probs : np.ndarray  – per-wafer defect probabilities for the batch
    batch_faults : list | None – per-wafer fault labels (used in summary text)

    Returns
    -------
    drift_detected : bool  – True if any drift level was found
    message        : str   – human-readable drift summary
    severity       : str   – "none" | "low" | "medium" | "high"
    """
    n         = len(defect_probs)
    mid       = n // 2
    early_avg = defect_probs[:mid].mean()
    late_avg  = defect_probs[mid:].mean()
    trend     = late_avg - early_avg

    # Summarise the most common fault type if available
    fault_summary = ""
    if batch_faults:
        from collections import Counter
        fault_counts = Counter(f for f in batch_faults if f != "none")
        if fault_counts:
            top_fault, top_count = fault_counts.most_common(1)[0]
            fault_summary = f" Most frequent fault: {top_fault} ({top_count} times)."

    if trend > 0.20:
        return True, (
            f"SEVERE drift — defect probability rose from {early_avg:.2f} to "
            f"{late_avg:.2f} across batch (+{trend:.2f}).{fault_summary} "
            f"Recommend immediate line stop and tool inspection."
        ), "high"

    if trend > 0.10:
        return True, (
            f"MODERATE drift — defect probability rose from {early_avg:.2f} to "
            f"{late_avg:.2f} across batch (+{trend:.2f}).{fault_summary} "
            f"Recommend recalibration before the next lot."
        ), "medium"

    if trend > 0.05:
        return True, (
            f"MILD drift — defect probability rose from {early_avg:.2f} to "
            f"{late_avg:.2f} across batch (+{trend:.2f}).{fault_summary} "
            f"Monitor closely; recalibrate if the trend continues."
        ), "low"

    return False, (
        f"No systematic drift — defect probability stable "
        f"({early_avg:.2f} early vs {late_avg:.2f} late)."
    ), "none"


# ---------------------------------------------------------------------------
# Embedding and RAG Memory
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """
    Generate a 1536-dimensional embedding vector for the given text using
    OpenAI's text-embedding-3-small model.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def retrieve_similar_cases(current_payload: dict, top_k: int = 3) -> list[dict]:
    """
    Query Pinecone for the top-k most similar historical defect cases.

    The query is formed by embedding a JSON representation of the current
    wafer's features. Each returned case contains metadata stored during
    a previous orchestrator run (defect_prob, fault_type, tool_type, etc.).

    Parameters
    ----------
    current_payload : dict – features describing the current wafer
    top_k           : int  – number of similar cases to retrieve

    Returns
    -------
    list of metadata dicts from Pinecone matches (empty on failure)
    """
    query_embedding = embed_text(json.dumps(current_payload))

    try:
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="rca_memory",
        )
        return [match["metadata"] for match in results["matches"]]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Stage-Level Context Helpers (used to build LLM prompt sections)
# ---------------------------------------------------------------------------

def _get_fault_stage_tool(
    batch_meta: list | None,
    batch_idx: int,
    fallback_tool: str,
) -> str:
    """
    Identify which physical tool caused the first fault in a wafer's
    pipeline run by inspecting the per-stage results from the simulator.

    The merged sensor row always carries the Tool_Type of the final
    pipeline stage (Inspection/Deposition), so reading it directly would
    misattribute faults that occurred upstream (e.g. in Lithography or
    Etch). This function walks stage_results in pipeline order and returns
    the tool of the first stage where a fault was recorded.

    Parameters
    ----------
    batch_meta    : list[dict] | None – per-wafer metadata from simulator
    batch_idx     : int               – index of this wafer in the batch
    fallback_tool : str               – Tool_Type from the merged sensor row

    Returns
    -------
    str – tool name of the fault-causing stage, or fallback_tool
    """
    if not batch_meta or batch_idx >= len(batch_meta):
        return fallback_tool

    for stage_result in batch_meta[batch_idx].get("stage_results", []):
        if stage_result.get("fault_type", "none") != "none":
            return stage_result["tool"]

    return fallback_tool


def _build_stage_context(batch_meta: list | None, batch_idx: int) -> str:
    """
    Build a plain-text summary of every pipeline stage that experienced
    a fault or active drift for a given wafer.

    Clean stages (NORMAL, no drift) are omitted to keep the LLM prompt
    concise and focused on actionable information. The output is inserted
    verbatim into the LLM prompt under the heading
    "STAGE-LEVEL BREAKDOWN (pipeline order)".

    Parameters
    ----------
    batch_meta : list[dict] | None – per-wafer metadata from simulator
    batch_idx  : int               – index of this wafer in the batch

    Returns
    -------
    str – formatted stage summary, or "" if no data / all stages clean
    """
    if not batch_meta or batch_idx >= len(batch_meta):
        return ""

    stage_results = batch_meta[batch_idx].get("stage_results", [])
    if not stage_results:
        return ""

    lines = []
    for sr in stage_results:
        fault_type      = sr.get("fault_type",     "none")
        drift_active    = sr.get("drift_active",    False)
        drifting_sensor = sr.get("drifting_sensor", None)
        drift_mag       = sr.get("drift_magnitude", 0.0)
        sim_label       = sr.get("simulator_label", "none")
        stage           = sr.get("stage",           "unknown")
        tool            = sr.get("tool",            "unknown")

        if fault_type == "none" and not drift_active:
            continue  # stage was clean — skip

        if fault_type != "none":
            drift_note = (
                f" | drift on {drifting_sensor} mag={drift_mag:.3f} [FAULT_IN_DRIFT]"
                if drifting_sensor else ""
            )
            lines.append(
                f"  [{stage}] tool={tool} | FAULT={fault_type}"
                f" | pattern={sim_label}{drift_note}"
            )
        else:
            lines.append(
                f"  [{stage}] tool={tool} | DRIFT on {drifting_sensor}"
                f" mag={drift_mag:.3f} (no fault yet — monitor)"
            )

    if not lines:
        return ""

    return "STAGE-LEVEL BREAKDOWN (pipeline order):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Root Cause Analysis
# ---------------------------------------------------------------------------

def call_llm_with_rag(
    def_prob: float,
    join_prob: float,
    features: list[str],
    similar_cases: list[dict],
    cnn_pattern: str | None = None,
    physics_pattern: str | None = None,
    agreement: str | None = None,
    fault_type: str | None = None,
    tool_type: str | None = None,
    drift_summary: str | None = None,
    stage_context: str | None = None,
) -> dict:
    """
    Call GPT-4o-mini to generate a structured root cause analysis report.

    The prompt combines all available evidence:
      - XGBoost defect and join probabilities
      - Top SHAP-identified sensor features
      - CNN-classified surface defect pattern and physics expectation
      - Agreement status between physics model and CNN
      - Fault-causing tool and stage context from the simulator
      - Batch-level drift status
      - Top-1 similar case retrieved from Pinecone memory

    The model is instructed to return valid JSON only; the response is
    parsed and returned as a Python dict. A fallback dict is returned if
    the LLM call fails or the response cannot be parsed.

    Parameters
    ----------
    def_prob        : float       – XGBoost defect probability
    join_prob       : float       – XGBoost join-status probability
    features        : list[str]   – top 3 SHAP feature names
    similar_cases   : list[dict]  – retrieved historical cases from Pinecone
    cnn_pattern     : str | None  – CNN-classified wafer map pattern
    physics_pattern : str | None  – physics-expected pattern for this fault
    agreement       : str | None  – "high" | "partial" | "conflict"
    fault_type      : str | None  – fault label from the simulator
    tool_type       : str | None  – tool that caused the fault
    drift_summary   : str | None  – batch drift message (None if no drift)
    stage_context   : str | None  – per-stage breakdown text from _build_stage_context

    Returns
    -------
    dict with keys:
      rca_summary, root_causes, wafer_pattern, tool_responsible,
      process_step, confidence, recommended_tests, stop_line
    """
    pattern_context = ""
    if cnn_pattern and cnn_pattern != "unknown":
        pattern_context = f"""
WAFER MAP ANALYSIS:
  CNN Classified Pattern  : {cnn_pattern}
  Physics Expected Pattern: {physics_pattern or 'N/A'}
  Agreement Status        : {agreement or 'N/A'}
  Fault Type Detected     : {fault_type or 'none'}
  Tool Type               : {tool_type or 'Unknown'}
"""

    drift_context = ""
    if drift_summary:
        drift_context = f"""
BATCH DRIFT STATUS:
  {drift_summary}
"""

    stage_context_section = f"\n{stage_context}\n" if stage_context else ""

    prompt = f"""
You are an expert semiconductor process engineer performing root cause
analysis on a defective wafer.

SENSOR ANALYSIS:
  Defect Probability : {def_prob:.3f}
  Join Probability   : {join_prob:.3f}
  Top SHAP Features  : {features}
{pattern_context}{stage_context_section}{drift_context}
SIMILAR PAST CASES FROM MEMORY:
{json.dumps(similar_cases, indent=2)}

Based on ALL of the above — sensor readings, wafer map pattern, physics
mapping, stage-level fault breakdown, and similar past cases — provide a
comprehensive root cause analysis.

Return ONLY valid JSON (no markdown fences, no preamble):
{{
    "rca_summary"       : "detailed explanation of what caused this defect",
    "root_causes"       : ["specific cause 1", "specific cause 2"],
    "wafer_pattern"     : "{cnn_pattern or 'unknown'}",
    "tool_responsible"  : "which tool type caused this",
    "process_step"      : "which process step — etching/deposition/lithography/inspection",
    "confidence"        : "low/medium/high",
    "recommended_tests" : ["specific action 1", "specific action 2"],
    "stop_line"         : true or false
}} 
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except Exception:
            # Recover if the model wrapped the JSON in markdown fences
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            return json.loads(raw[start:end])

    except Exception:
        return {
            "rca_summary"      : "LLM call failed — using sensor-only analysis.",
            "root_causes"      : features,
            "wafer_pattern"    : cnn_pattern or "unknown",
            "tool_responsible" : tool_type or "unknown",
            "process_step"     : "unknown",
            "confidence"       : "low",
            "recommended_tests": [],
            "stop_line"        : False,
        }


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

df = pd.read_csv(FILE_PATH)

X        = df.drop(columns=["Join_Status", "Defect", "Year", "Month", "Date", "Hour", "Minute"])
y_defect = df["Defect"].astype(int)
y_join   = pd.factorize(df["Join_Status"])[0]

X_train, X_test, y_def_train, y_def_test, y_join_train, y_join_test = train_test_split(
    X, y_defect, y_join, test_size=0.2, random_state=42
)

numeric_cols     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                                           numeric_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
])

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled  = preprocessor.transform(X_test)

all_features = np.concatenate([
    numeric_cols,
    preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols),
])

# Stage 1: predict defect probability from sensor features
xgb_defect = XGBClassifier(n_estimators=200, max_depth=6, eval_metric="logloss")
xgb_defect.fit(X_train_scaled, y_def_train)

# Stage 2: predict join status, augmented with the defect probability
def_pred_train = xgb_defect.predict_proba(X_train_scaled)[:, 1]
X_train_join   = np.hstack([X_train_scaled, def_pred_train.reshape(-1, 1)])

xgb_join = XGBClassifier(n_estimators=200, max_depth=6, eval_metric="logloss")
xgb_join.fit(X_train_join, y_join_train)

# SHAP explainer for feature attribution
explainer = shap.Explainer(xgb_defect, X_train_scaled)


# ---------------------------------------------------------------------------
# Agent Planner
# ---------------------------------------------------------------------------

def agent_planner(
    defect_prob: float,
    join_prob: float,
    top_features: list[str],
    has_image: bool = False,
    agreement: str = "none",
) -> list[str]:
    """
    Decide which analysis steps to run for a given wafer.

    The plan always starts with feature analysis and ends with a test
    recommendation and memory storage step. Additional steps (image
    classification, LLM RCA, similar-case retrieval) are added based
    on whether an image is available and how high the risk scores are.

    Parameters
    ----------
    defect_prob  : float – XGBoost defect probability
    join_prob    : float – XGBoost join-status probability
    top_features : list  – top SHAP feature names (unused here, reserved)
    has_image    : bool  – whether a wafer map image was provided
    agreement    : str   – current agreement status (unused here, reserved)

    Returns
    -------
    list[str] – ordered list of step names for the orchestrator to execute
    """
    plan = ["analyze_features"]

    if has_image:
        plan.append("classify_image")
        plan.append("check_agreement")

    if defect_prob > 0.7:
        plan.append("retrieve_similar_cases")

    if defect_prob > 0.5 or join_prob > 0.5:
        plan.append("run_llm_rca")

    plan.append("recommend_tests")
    plan.append("store_case_memory")
    return plan


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def orchestrator(
    X_input: pd.DataFrame,
    batch_images: list | None = None,
    batch_patterns: list | None = None,
    batch_faults: list | None = None,
    batch_ids: list | None = None,
    batch_meta: list | None = None,
) -> pd.DataFrame:
    """
    Main agentic loop: score, analyse, and log high-risk wafers in a batch.

    For each batch of wafers (typically 50), the orchestrator:
      1. Scores all wafers with XGBoost (defect and join probabilities).
      2. Detects systematic drift across the batch.
      3. Identifies the top 20% highest-risk wafers for detailed analysis.
      4. For each high-risk wafer, executes the agent plan:
         - SHAP feature attribution
         - CNN image classification (if image provided)
         - Physics–CNN agreement check
         - Pinecone similar-case retrieval (if defect_prob > 0.7)
         - LLM root cause analysis (if defect_prob > 0.9)
         - Test recommendations
         - Memory storage in Pinecone
      5. Upserts all high-risk wafer embeddings to Pinecone.
      6. Returns a flat DataFrame of RCA results for the batch.

    Parameters
    ----------
    X_input        : pd.DataFrame – sensor readings for the batch
    batch_images   : list | None  – paths to wafer map images (per wafer)
    batch_patterns : list | None  – physics-expected patterns (per wafer)
    batch_faults   : list | None  – fault labels from the simulator (per wafer)
    batch_ids      : list | None  – wafer ID integers (per wafer)
    batch_meta     : list | None  – full per-wafer simulator metadata dicts

    Returns
    -------
    pd.DataFrame – one row per high-risk wafer with RCA columns
    """
    logs             = []
    pinecone_vectors = []
    llm_calls        = 0
    max_llm_calls    = 5
    MAX_BATCH_SIZE   = 50   # Pinecone upsert chunk size

    # Score all wafers in the batch
    X_scaled     = preprocessor.transform(X_input)
    defect_probs = xgb_defect.predict_proba(X_scaled)[:, 1]
    join_probs   = xgb_join.predict_proba(
        np.hstack([X_scaled, defect_probs.reshape(-1, 1)])
    )[:, 1]
    shap_values = explainer(X_scaled).values

    # Batch-level drift detection
    drift_detected, drift_summary, drift_severity = detect_batch_drift(
        defect_probs, batch_faults
    )

    # Analyse only the top 20% highest-risk wafers
    high_risk = np.where(defect_probs >= np.percentile(defect_probs, 80))[0]

    for i in high_risk:

        image_path      = batch_images[i]   if batch_images   else None
        physics_pattern = batch_patterns[i] if batch_patterns else None
        fault_type      = batch_faults[i]   if batch_faults   else "none"
        wafer_id        = batch_ids[i]      if batch_ids      else int(i)

        # Identify which tool actually caused the fault (may differ from
        # the merged row's Tool_Type, which always shows the last stage).
        if "Tool_Type" in X_input.columns:
            fallback_tool = X_input.iloc[i]["Tool_Type"]
        else:
            fallback_tool = "Unknown"
        tool_type = _get_fault_stage_tool(batch_meta, i, fallback_tool)

        # Build the per-stage fault/drift summary for the LLM prompt
        stage_context = _build_stage_context(batch_meta, i)

        # Top 3 SHAP features for this wafer
        top_idx      = np.argsort(np.abs(shap_values[i]))[::-1][:3]
        top_features = [all_features[j] for j in top_idx]

        cnn_pattern = None
        cnn_conf    = 0.0
        agreement   = "none"

        plan = agent_planner(
            defect_probs[i], join_probs[i], top_features,
            has_image=(image_path is not None),
        )

        similar_cases = []

        # Default RCA (used if LLM step is not triggered)
        rca = {
            "rca_summary"      : f"Top contributing features: {', '.join(top_features)}",
            "root_causes"      : top_features,
            "wafer_pattern"    : physics_pattern or "unknown",
            "tool_responsible" : tool_type,
            "process_step"     : "unknown",
            "confidence"       : "medium",
            "recommended_tests": [],
            "stop_line"        : False,
        }

        # Execute each planned step
        for step in plan:

            if step == "analyze_features":
                pass  # SHAP values already computed above

            elif step == "classify_image":
                if image_path:
                    cnn_pattern, cnn_conf, _ = classify_wafer_image(image_path)

            elif step == "check_agreement":
                if cnn_pattern and physics_pattern:
                    agreement, agree_note = check_agreement(
                        physics_pattern, cnn_pattern, cnn_conf,
                        defect_prob=float(defect_probs[i]),
                        fault_type=fault_type,
                    )
                    if agreement == "conflict":
                        rca["confidence"]   = "low"
                        rca["rca_summary"]  = (
                            f"CONFLICT: Physics predicted '{physics_pattern}' "
                            f"but CNN found '{cnn_pattern}'. "
                            f"Manual engineer review required."
                        )

            elif step == "retrieve_similar_cases":
                similar_cases = retrieve_similar_cases({
                    "defect_prob"  : float(defect_probs[i]),
                    "join_prob"    : float(join_probs[i]),
                    "top_features" : top_features,
                    "wafer_pattern": cnn_pattern or physics_pattern,
                })

            elif step == "run_llm_rca" and llm_calls < max_llm_calls and defect_probs[i] > 0.9:
                rca = call_llm_with_rag(
                    defect_probs[i],
                    join_probs[i],
                    top_features,
                    similar_cases[:1],
                    cnn_pattern     = cnn_pattern,
                    physics_pattern = physics_pattern,
                    agreement       = agreement,
                    fault_type      = fault_type,
                    tool_type       = tool_type,
                    drift_summary   = drift_summary if drift_detected else None,
                    stage_context   = stage_context,
                )
                llm_calls += 1

            elif step == "recommend_tests":
                if not rca.get("recommended_tests"):
                    tests = []
                    for f in top_features:
                        base = f.replace("_", " ")
                        tests.extend([
                            f"Validate stability of {base}",
                            f"Check calibration related to {base}",
                            f"Review process logs for {base}",
                        ])
                    if cnn_pattern and cnn_pattern != "unknown":
                        tests.append(f"Inspect wafer surface for '{cnn_pattern}' pattern.")
                    rca["recommended_tests"] = list(set(tests))[:5]

            elif step == "store_case_memory":
                embedding = embed_text(json.dumps({
                    "wafer_id"        : wafer_id,
                    "defect_prob"     : float(defect_probs[i]),
                    "join_prob"       : float(join_probs[i]),
                    "top_features"    : top_features,
                    "cnn_pattern"     : cnn_pattern or "unknown",
                    "physics_pattern" : physics_pattern or "unknown",
                    "agreement"       : agreement,
                    "fault_type"      : fault_type,
                    "tool_type"       : tool_type,
                    "rca_summary"     : rca.get("rca_summary", ""),
                    "root_causes"     : rca.get("root_causes", []),
                    "confidence"      : rca.get("confidence", ""),
                    "recommended_tests": rca.get("recommended_tests", []),
                }))
                pinecone_vectors.append({
                    "id"      : f"wafer-{wafer_id}",
                    "values"  : embedding,
                    "metadata": {
                        "defect_prob" : float(defect_probs[i]),
                        "join_prob"   : float(join_probs[i]),
                        "cnn_pattern" : cnn_pattern or "unknown",
                        "fault_type"  : fault_type,
                        "tool_type"   : tool_type,
                        "agreement"   : agreement,
                    },
                })

        # Flatten results for CSV/Excel logging
        flat_log = {
            "wafer_id"         : wafer_id,
            "defect_prob"      : float(defect_probs[i]),
            "join_prob"        : float(join_probs[i]),
            "top_feature_1"    : top_features[0] if len(top_features) > 0 else "",
            "top_feature_2"    : top_features[1] if len(top_features) > 1 else "",
            "top_feature_3"    : top_features[2] if len(top_features) > 2 else "",
            "physics_pattern"  : physics_pattern or "unknown",
            "cnn_pattern"      : cnn_pattern or "unknown",
            "cnn_confidence"   : round(cnn_conf, 3),
            "agreement"        : agreement,
            "fault_type"       : fault_type,
            "tool_type"        : tool_type,
            "agent_plan"       : " -> ".join(plan),
            "rca_summary"      : rca.get("rca_summary", ""),
            "root_causes"      : " | ".join(rca.get("root_causes", [])),
            "wafer_pattern"    : rca.get("wafer_pattern", ""),
            "tool_responsible" : rca.get("tool_responsible", ""),
            "process_step"     : rca.get("process_step", ""),
            "confidence"       : rca.get("confidence", ""),
            "recommended_tests": " | ".join(rca.get("recommended_tests", [])),
            "stop_line"        : rca.get("stop_line", False),
            "drift_detected"   : drift_detected,
            "drift_severity"   : drift_severity,
            "drift_summary"    : drift_summary,
        }
        logs.append(flat_log)

    # Upsert to Pinecone in chunks of 50
    for j in range(0, len(pinecone_vectors), MAX_BATCH_SIZE):
        pinecone_index.upsert(
            vectors=pinecone_vectors[j : j + MAX_BATCH_SIZE],
            namespace="rca_memory",
        )

    # Console summary for this batch
    batch_num   = (batch_ids[0] // 50 + 1) if batch_ids else "?"
    wafer_start = batch_ids[0]  if batch_ids else "?"
    wafer_end   = batch_ids[-1] if batch_ids else "?"

    print(f"\n{'=' * 55}")
    print(f"  BATCH {batch_num} | Wafers #{wafer_start}–#{wafer_end}")
    print(f"{'=' * 55}")
    print(
        f"  Drift Status : [{drift_severity.upper()}] {drift_summary}"
        if drift_detected
        else "  Drift Status : STABLE"
    )
    print(f"  High-risk    : {len(high_risk)} of 50 wafers ({len(high_risk) * 2}%)")
    print(f"  LLM Calls    : {llm_calls}/{max_llm_calls}")

    if logs:
        print(f"\n  HIGH RISK WAFERS:")
        print(f"  {'-' * 51}")
        for log in logs:
            symbol   = {"high": "✓", "partial": "~", "conflict": "⚠"}.get(log["agreement"], " ")
            stop_tag = " [STOP LINE]" if log["stop_line"] else ""
            print(
                f"  Wafer #{log['wafer_id']:04d} | defect: {log['defect_prob']:.3f} | "
                f"pattern: {log['cnn_pattern']:12s} | agree: {symbol} {log['agreement']:8s} | "
                f"fault: {log['fault_type']}{stop_tag}"
            )
            print(f"    Causes : {log['top_feature_1']}, {log['top_feature_2']}, {log['top_feature_3']}")
            if log["agreement"] == "conflict":
                print(
                    f"    ⚠ CONFLICT: Physics={log['physics_pattern']} "
                    f"vs CNN={log['cnn_pattern']} — engineer review needed"
                )
            if log["stop_line"]:
                print("    ★ STOP LINE RECOMMENDED")
    else:
        print("  No high-risk wafers flagged this batch.")

    print(f"{'=' * 55}\n")

    return pd.DataFrame(logs)


# ---------------------------------------------------------------------------
# Analytics Charts
# ---------------------------------------------------------------------------

def correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot and save a Pearson correlation heatmap for all process sensor
    features, excluding time-index columns (Year, Month, Date, Hour, Minute).
    """
    exclude       = {"Year", "Month", "Date", "Hour", "Minute"}
    numeric_cols  = [c for c in df.select_dtypes(include=["int64", "float64"]).columns
                     if c not in exclude]
    corr_matrix   = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, cbar=True)
    plt.title("Process Feature Correlation Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("feature_correlation_heatmap.png")
    plt.close()


def generate_charts(process_df: pd.DataFrame, result_df: pd.DataFrame) -> None:
    """
    Generate and save all analytics charts to the working directory.

    Charts produced:
      defect_probability_histogram.png   – distribution of predicted defect probabilities
      join_vs_defect_risk_map.png        – scatter plot of join vs defect probability,
                                           coloured by risk tier (Low / Medium / High)
      process_density_map.png            – KDE density of the join × defect space
      root_cause_contribution_chart.png  – top 10 sensor features by defect contribution
      cnn_pattern_distribution.png       – wafer map pattern frequency (CNN predictions)
      agreement_distribution.png         – physics–CNN agreement level counts
      feature_correlation_heatmap.png    – sensor feature correlation matrix

    Parameters
    ----------
    process_df : pd.DataFrame – raw sensor data (from the CSV, used for correlation heatmap)
    result_df  : pd.DataFrame – RCA results returned by the orchestrator
    """
    sns.set(style="whitegrid")

    # Defect probability histogram
    plt.figure()
    sns.histplot(result_df["defect_prob"], bins=20, kde=True)
    plt.title("Defect Probability Distribution")
    plt.xlabel("Predicted Defect Probability")
    plt.ylabel("Number of Samples")
    plt.savefig("defect_probability_histogram.png")
    plt.close()

    # Risk tiers
    def _risk_tier(defect, join):
        if defect >= 0.8 and join >= 0.8:
            return "High Risk"
        if defect >= 0.5 or join >= 0.5:
            return "Medium Risk"
        return "Low Risk"

    result_df = result_df.copy()
    result_df["risk_level"] = [
        _risk_tier(d, j) for d, j in zip(result_df["defect_prob"], result_df["join_prob"])
    ]

    # Risk scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=result_df["join_prob"],
        y=result_df["defect_prob"],
        hue=result_df["risk_level"],
        palette={"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"},
    )
    plt.axhline(0.5, linestyle="--")
    plt.axvline(0.5, linestyle="--")
    plt.title("Process Risk Map: Join vs Defect Probability")
    plt.xlabel("Join Probability")
    plt.ylabel("Defect Probability")
    plt.tight_layout()
    plt.savefig("join_vs_defect_risk_map.png")
    plt.close()

    # KDE density map
    plt.figure()
    sns.kdeplot(x=result_df["join_prob"], y=result_df["defect_prob"], fill=True)
    plt.title("Defect Density in Process Space")
    plt.xlabel("Join Probability")
    plt.ylabel("Defect Probability")
    plt.savefig("process_density_map.png")
    plt.close()

    # Root-cause feature contribution bar chart
    exclude_features = {"Minute", "Year", "Month", "Date", "Hour"}
    root_counts = (
        result_df["root_causes"]
        .str.split("|")
        .explode()
        .str.strip()
    )
    root_counts   = root_counts[~root_counts.isin(exclude_features)]
    root_counts   = root_counts.value_counts().head(10)
    pct_contrib   = (root_counts / root_counts.sum()) * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(x=pct_contrib.index, y=pct_contrib.values)
    plt.xlabel("Process Feature")
    plt.ylabel("Contribution to Defects (%)")
    plt.title("Process Parameter Contribution to Defects")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("root_cause_contribution_chart.png")
    plt.close()

    # CNN pattern distribution
    if "cnn_pattern" in result_df.columns:
        pattern_counts = result_df["cnn_pattern"].value_counts()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=pattern_counts.index, y=pattern_counts.values)
        plt.title("Surface Defect Pattern Distribution (CNN)")
        plt.xlabel("Wafer Map Pattern")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("cnn_pattern_distribution.png")
        plt.close()

    # Physics–CNN agreement distribution
    if "agreement" in result_df.columns:
        agree_counts = result_df["agreement"].value_counts()
        colours = {"high": "green", "partial": "orange", "conflict": "red", "none": "gray"}
        plt.figure(figsize=(6, 4))
        sns.barplot(
            x=agree_counts.index,
            y=agree_counts.values,
            palette=[colours.get(a, "blue") for a in agree_counts.index],
        )
        plt.title("Physics vs CNN Agreement Distribution")
        plt.xlabel("Agreement Level")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("agreement_distribution.png")
        plt.close()

    # Sensor feature correlation heatmap
    correlation_heatmap(process_df)

    print("All analytics charts saved successfully.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Persist models to disk for use by downstream scripts or notebooks
    joblib.dump(xgb_defect,   "xgb_defect_model.pkl")
    joblib.dump(xgb_join,     "xgb_join_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")

    from digital_twin_simulator import DigitalTwinSimulator

    twin = DigitalTwinSimulator(
        csv_path  = FILE_PATH,
        image_dir = os.path.join(BASE_DIR, "WM811k_Dataset"),
    )

    results = twin.stream(
        orchestrator,
        interval_seconds = 0,
        max_wafers       = 650,
        batch_size       = 50,
    )

    final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    final_df.to_excel(os.path.join(BASE_DIR, "digital_twin_results.xlsx"), index=False)

    if not final_df.empty:
        generate_charts(process_df=df, result_df=final_df)

    print("Execution complete.")

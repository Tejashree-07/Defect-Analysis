# ============================================================
# DIGITAL TWIN SIMULATOR
# ============================================================

"""
digital_twin_simulator.py
=========================
Digital Twin Simulator for Semiconductor Wafer Fabrication

Simulates a four-stage wafer fabrication pipeline:
  Lithography → Etch → Deposition → Inspection

Each wafer is run sequentially through all four stages. At every stage the
simulator:
  1. Samples sensor readings from a stage-specific multivariate Gaussian
     manifold fitted to real CSV data (active sensors only, inactive sensors
     pinned to their stage mean).
  2. Optionally injects a realistic tool drift episode on one active sensor.
  3. May inject a fault whose type is causally linked to the drifting sensor
     (when drift is active) or chosen at random otherwise.
  4. Derives a ground-truth wafer map pattern label from the stage × fault
     probability table.
  5. Runs a Mahalanobis anomaly check against the stage manifold.

Wafers are streamed in configurable batches to an orchestrator callback
(see DigitalTwinSimulator.stream).

Dependencies:
  pip install numpy pandas scipy scikit-learn
"""

import os
import time
import random
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import truncnorm, chi2


# ---------------------------------------------------------------------------
# Pipeline Constants
# ---------------------------------------------------------------------------

PROCESS_STAGES = ["Lithography", "Etch", "Deposition", "Inspection"]

# Maps CSV Tool_Type values to stage names (used when fitting stage manifolds)
TOOL_TYPE_TO_STAGE = {
    "Lithography" : "Lithography",
    "Etching"     : "Etch",
    "Deposition"  : "Deposition",
}

# Maps each stage to the canonical Tool_Type string used in the CSV.
# Inspection shares the Deposition chamber (same physical tool).
STAGE_TO_TOOL_TYPE = {
    "Lithography" : "Lithography",
    "Etch"        : "Etching",
    "Deposition"  : "Deposition",
    "Inspection"  : "Deposition",
}

# Sensors that are actively varying at each stage.
# Inactive sensors are pinned to their stage mean (stable background noise).
STAGE_ACTIVE_SENSORS = {
    "Lithography" : [
        "UV_Exposure_Intensity",
        "Stage_Alignment_Error",
        "Vibration_Level",
        "Chamber_Temperature",
    ],
    "Etch" : [
        "RF_Power",
        "Etch_Depth",
        "Vacuum_Pressure",
        "Gas_Flow_Rate",
        "Chamber_Temperature",
    ],
    "Deposition" : [
        "Chamber_Temperature",
        "Gas_Flow_Rate",
        "Vacuum_Pressure",
        "Rotation_Speed",
        "Particle_Count",
    ],
    "Inspection" : [
        "Particle_Count",
        "Stage_Alignment_Error",
        "Vibration_Level",
    ],
}

# Fault catalogue per stage.
# Each entry: (fault_label, sensor_column, value_transform_fn)
# When a sensor is drifting, only faults targeting that sensor are eligible.
STAGE_FAULTS = {
    "Lithography": [
        ("uv_overexposure",     "UV_Exposure_Intensity", lambda v: v * random.uniform(1.20, 1.50)),
        ("alignment_error",     "Stage_Alignment_Error", lambda v: v + random.uniform(2.0,  5.0)),
        ("vibration_event",     "Vibration_Level",       lambda v: v * random.uniform(3.0,  8.0)),
        ("thermal_drift_litho", "Chamber_Temperature",   lambda v: v + random.uniform(5.0, 15.0)),
    ],
    "Etch": [
        ("rf_power_drop",   "RF_Power",           lambda v: v * random.uniform(0.50, 0.75)),
        ("over_etch",       "Etch_Depth",          lambda v: v * random.uniform(1.15, 1.40)),
        ("vacuum_loss",     "Vacuum_Pressure",     lambda v: v * random.uniform(1.50, 3.00)),
        ("gas_starvation",  "Gas_Flow_Rate",       lambda v: v * random.uniform(0.40, 0.70)),
        ("temp_spike_etch", "Chamber_Temperature", lambda v: v * random.uniform(1.10, 1.25)),
    ],
    "Deposition": [
        ("temperature_spike", "Chamber_Temperature", lambda v: v * random.uniform(1.10, 1.25)),
        ("particle_burst",    "Particle_Count",      lambda v: v + random.uniform(300,  700)),
        ("gas_excess",        "Gas_Flow_Rate",       lambda v: v * random.uniform(1.20, 1.60)),
        ("rotation_stall",    "Rotation_Speed",      lambda v: v * random.uniform(0.40, 0.70)),
        ("vacuum_loss_dep",   "Vacuum_Pressure",     lambda v: v * random.uniform(1.30, 2.50)),
    ],
    "Inspection": [
        ("particle_burst",  "Particle_Count",        lambda v: v + random.uniform(200, 500)),
        ("alignment_error", "Stage_Alignment_Error", lambda v: v + random.uniform(1.0, 3.0)),
        ("vibration_event", "Vibration_Level",       lambda v: v * random.uniform(2.0, 5.0)),
    ],
}

# Stage × fault_type → wafer map pattern probability distribution
STAGE_FAULT_PATTERNS = {
    "Lithography": {
        "uv_overexposure"     : {"Center": 0.55, "near full": 0.25, "Donut": 0.20},
        "alignment_error"     : {"Edge Local": 0.50, "Edge Ring": 0.30, "Scratch": 0.20},
        "vibration_event"     : {"Scratch": 0.60, "Local": 0.25, "random": 0.15},
        "thermal_drift_litho" : {"Center": 0.40, "Donut": 0.35, "near full": 0.25},
        "drift"               : {"Edge Local": 0.45, "Center": 0.30, "Donut": 0.25},
        "none"                : {"none": 1.00},
    },
    "Etch": {
        "rf_power_drop"   : {"Center": 0.50, "Local": 0.25, "Donut": 0.25},
        "over_etch"       : {"Edge Ring": 0.45, "near full": 0.35, "random": 0.20},
        "vacuum_loss"     : {"Edge Ring": 0.45, "near full": 0.30, "random": 0.25},
        "gas_starvation"  : {"Center": 0.40, "Donut": 0.35, "Local": 0.25},
        "temp_spike_etch" : {"Edge Ring": 0.40, "Center": 0.35, "Scratch": 0.25},
        "drift"           : {"Edge Ring": 0.40, "Edge Local": 0.35, "Donut": 0.25},
        "none"            : {"none": 1.00},
    },
    "Deposition": {
        "temperature_spike" : {"Edge Ring": 0.40, "Center": 0.25, "Scratch": 0.20, "near full": 0.15},
        "particle_burst"    : {"random": 0.55, "Local": 0.25, "near full": 0.20},
        "gas_excess"        : {"near full": 0.50, "random": 0.30, "Edge Ring": 0.20},
        "rotation_stall"    : {"Center": 0.55, "Donut": 0.30, "Local": 0.15},
        "vacuum_loss_dep"   : {"Edge Ring": 0.45, "near full": 0.30, "random": 0.25},
        "drift"             : {"Edge Local": 0.40, "Edge Ring": 0.35, "Donut": 0.25},
        "none"              : {"none": 1.00},
    },
    "Inspection": {
        "particle_burst"  : {"random": 0.60, "Local": 0.40},
        "alignment_error" : {"Edge Local": 0.60, "Scratch": 0.40},
        "vibration_event" : {"Scratch": 0.70, "Local": 0.30},
        "drift"           : {"Edge Local": 0.55, "Scratch": 0.45},
        "none"            : {"none": 1.00},
    },
}

# Hard physical bounds enforced after every injection or drift step
PHYSICAL_BOUNDS = {
    "Chamber_Temperature"   : (20.0,  250.0),
    "Gas_Flow_Rate"         : (0.0,   500.0),
    "RF_Power"              : (0.0,   1000.0),
    "Etch_Depth"            : (0.0,   2000.0),
    "Rotation_Speed"        : (0.0,   5000.0),
    "Vacuum_Pressure"       : (1e-6,  10.0),
    "Stage_Alignment_Error" : (0.0,   20.0),
    "Vibration_Level"       : (0.0,   5.0),
    "UV_Exposure_Intensity" : (0.0,   500.0),
    "Particle_Count"        : (0.0,   5000.0),
}


# ---------------------------------------------------------------------------
# Wafer State Machine
# ---------------------------------------------------------------------------

class WaferState:
    """
    Four mutually exclusive states for a single stage-pass.

    NORMAL         – sensor readings within normal operating range
    DRIFT          – a tool is drifting but no fault has been injected yet
    FAULT          – a fault was injected with no preceding drift context
    FAULT_IN_DRIFT – a fault was injected while the tool was already drifting
                     (drift magnitude > 0.15 when the fault fired)
    """
    NORMAL         = "NORMAL"
    DRIFT          = "DRIFT"
    FAULT          = "FAULT"
    FAULT_IN_DRIFT = "FAULT_IN_DRIFT"


# ---------------------------------------------------------------------------
# Per-Tool Drift State
# ---------------------------------------------------------------------------

class ToolDriftState:
    """
    Tracks the current drift episode for a single physical tool.

    Each tool has its own independent ToolDriftState instance, so
    multiple tools can drift simultaneously without interfering.

    Attributes
    ----------
    active      : bool   – whether a drift episode is currently running
    sensor_col  : str    – name of the sensor that is drifting
    sensor_idx  : int    – index of sensor_col in the global numeric_cols list
    per_wafer   : float  – signed shift applied to the sensor each wafer
    accumulated : float  – cumulative shift applied so far
    remaining   : int    – number of wafers left in this drift episode
    magnitude   : float  – normalised drift severity, clamped to [0, 1]
                           computed as |accumulated| / (3 × sensor_std)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.active      = False
        self.sensor_col  = None
        self.sensor_idx  = None
        self.per_wafer   = 0.0
        self.accumulated = 0.0
        self.remaining   = 0
        self.magnitude   = 0.0


# ---------------------------------------------------------------------------
# Digital Twin Simulator
# ---------------------------------------------------------------------------

class DigitalTwinSimulator:
    """
    Simulates a semiconductor fab line at the wafer level.

    On construction the simulator:
      - Loads the real process CSV and fits per-stage Gaussian manifolds.
      - Calibrates fault and drift probabilities from the CSV defect rate
        and the WM-811K surface defect rate.
      - Initialises one ToolDriftState per physical tool.
      - Optionally learns wafer-map pattern weights from the WM-811K folder.

    Use generate() to produce one complete wafer (all four stages), or
    stream() to feed batches to an orchestrator callback continuously.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str | None = None,
        fault_probability: float | None = None,
        drift_probability: float | None = None,
    ):
        """
        Parameters
        ----------
        csv_path          : path to synthetic_explicit.csv
        image_dir         : path to WM811k_Dataset folder (None = auto-detect)
        fault_probability : per-stage fault injection probability (None = auto)
        drift_probability : per-tool drift episode start probability (None = 0.10)
        """
        print("=" * 65)
        print("  DIGITAL TWIN SIMULATOR — Learning from data")
        print("=" * 65)

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} real wafers from CSV")

        self.numeric_cols = [
            "Chamber_Temperature",
            "Gas_Flow_Rate",
            "RF_Power",
            "Etch_Depth",
            "Rotation_Speed",
            "Vacuum_Pressure",
            "Stage_Alignment_Error",
            "Vibration_Level",
            "UV_Exposure_Intensity",
            "Particle_Count",
        ]

        # Fit per-stage Gaussian manifolds from real data
        self.stage_stats = {}
        self._fit_stage_manifolds(df)

        # Global stats (kept for Mahalanobis fallback when stage data is sparse)
        numeric_data     = df[self.numeric_cols].copy()
        self.global_mean = numeric_data.mean().values
        self.global_cov  = numeric_data.cov().values
        self.global_stds = numeric_data.std().values

        try:
            self.global_cov_inv = np.linalg.inv(self.global_cov)
        except np.linalg.LinAlgError:
            self.global_cov_inv = np.linalg.pinv(self.global_cov)

        self.mahal_threshold = float(
            np.sqrt(chi2.ppf(0.95, df=len(self.numeric_cols)))
        )

        # Calibrate probabilities
        csv_defect_rate     = float(df["Defect"].mean())
        wm811k_surface_rate = 0.15
        self.fault_probability = (
            fault_probability
            if fault_probability is not None
            else round((csv_defect_rate + wm811k_surface_rate) / 2, 3)
        )
        self.drift_probability = (
            drift_probability if drift_probability is not None else 0.10
        )

        # One drift state object per physical tool
        self.tool_drift = {
            "Lithography" : ToolDriftState(),
            "Etching"     : ToolDriftState(),
            "Deposition"  : ToolDriftState(),
        }

        # Stage → physical tool name (Inspection shares the Deposition tool)
        self.stage_to_tool = {
            "Lithography" : "Lithography",
            "Etch"        : "Etching",
            "Deposition"  : "Deposition",
            "Inspection"  : "Deposition",
        }

        self.wafer_count = 0

        # WM-811K image directory
        if image_dir is None:
            image_dir = os.path.join(
                os.path.dirname(os.path.abspath(csv_path)), "WM811k_Dataset"
            )
        self.image_dir = image_dir
        self._learn_pattern_weights()

        print(f"\n  Calibrated parameters:")
        print(f"    CSV defect rate      : {csv_defect_rate:.1%}")
        print(f"    WM-811K surface rate : {wm811k_surface_rate:.1%}")
        print(f"    Fault probability    : {self.fault_probability:.1%}")
        print(f"    Drift probability    : {self.drift_probability:.1%}")
        print(f"  Mahalanobis threshold  : {self.mahal_threshold:.3f}")
        print("=" * 65)


    # -----------------------------------------------------------------------
    # Stage Manifold Fitting
    # -----------------------------------------------------------------------

    def _fit_stage_manifolds(self, df: pd.DataFrame) -> None:
        """
        Fit a separate multivariate Gaussian for each process stage by
        filtering the CSV on Tool_Type, then extract the active-sensor
        sub-mean and sub-covariance for statistically correct sampling.

        Per-stage stats stored in self.stage_stats[stage]:
          mean       : full-sensor mean (10-vector)
          cov        : full-sensor covariance (10×10)
          stds       : full-sensor standard deviations
          cov_inv    : inverse covariance (for Mahalanobis)
          active_idx : integer indices of active sensors for this stage
          sub_mean   : active-sensor mean
          sub_cov    : active-sensor covariance
          sub_stds   : active-sensor standard deviations
          trunc_a/b  : z-score bounds for truncated-Gaussian fallback sampling
          n_rows     : number of CSV rows used for fitting

        Inspection inherits the Deposition manifold (same physical chamber)
        but uses Inspection's own active sensor list.
        """
        print(f"\n  Fitting per-stage manifolds (active-sensor subspace):")

        for tool_type, stage in TOOL_TYPE_TO_STAGE.items():
            subset = df[df["Tool_Type"] == tool_type][self.numeric_cols].copy()

            if len(subset) < 30:
                print(f"    {stage:12s}: insufficient rows ({len(subset)}) — using global stats")
                self.stage_stats[stage] = None
                continue

            mean    = subset.mean().values
            cov     = subset.cov().values
            stds    = subset.std().values

            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)

            active_sensors = STAGE_ACTIVE_SENSORS[stage]
            active_idx     = [
                self.numeric_cols.index(col)
                for col in active_sensors
                if col in self.numeric_cols
            ]

            sub_mean = mean[active_idx]
            sub_cov  = cov[np.ix_(active_idx, active_idx)]
            sub_stds = stds[active_idx]

            # Precompute z-score truncation bounds for physical limits
            trunc_a = np.zeros(len(active_idx))
            trunc_b = np.full(len(active_idx), np.inf)
            for i, col_idx in enumerate(active_idx):
                col    = self.numeric_cols[col_idx]
                lo, hi = PHYSICAL_BOUNDS[col]
                mu, sigma = sub_mean[i], sub_stds[i]
                trunc_a[i] = (lo - mu) / sigma if sigma > 0 else 0.0
                trunc_b[i] = (hi - mu) / sigma if sigma > 0 else np.inf

            self.stage_stats[stage] = {
                "mean"       : mean,
                "cov"        : cov,
                "stds"       : stds,
                "cov_inv"    : cov_inv,
                "active_idx" : active_idx,
                "sub_mean"   : sub_mean,
                "sub_cov"    : sub_cov,
                "sub_stds"   : sub_stds,
                "trunc_a"    : trunc_a,
                "trunc_b"    : trunc_b,
                "n_rows"     : len(subset),
            }

            key_sensors = ["RF_Power", "Vacuum_Pressure", "Particle_Count",
                           "Chamber_Temperature", "UV_Exposure_Intensity"]
            sensor_str  = "  ".join(
                f"{s.split('_')[0]}={mean[self.numeric_cols.index(s)]:.1f}"
                for s in key_sensors if s in self.numeric_cols
            )
            print(f"    {stage:12s}: n={len(subset):6,}  "
                  f"active_sensors={len(active_idx)}  {sensor_str}")

        # Inspection: inherit Deposition manifold, override active-sensor subspace
        if self.stage_stats.get("Deposition"):
            dep  = self.stage_stats["Deposition"]
            insp_sensors = STAGE_ACTIVE_SENSORS["Inspection"]
            insp_idx     = [
                self.numeric_cols.index(col)
                for col in insp_sensors
                if col in self.numeric_cols
            ]
            insp_sub_mean = dep["mean"][insp_idx]
            insp_sub_cov  = dep["cov"][np.ix_(insp_idx, insp_idx)]
            insp_sub_stds = dep["stds"][insp_idx]

            trunc_a = np.zeros(len(insp_idx))
            trunc_b = np.full(len(insp_idx), np.inf)
            for i, col_idx in enumerate(insp_idx):
                col    = self.numeric_cols[col_idx]
                lo, hi = PHYSICAL_BOUNDS[col]
                mu, sigma = insp_sub_mean[i], insp_sub_stds[i]
                trunc_a[i] = (lo - mu) / sigma if sigma > 0 else 0.0
                trunc_b[i] = (hi - mu) / sigma if sigma > 0 else np.inf

            self.stage_stats["Inspection"] = {
                "mean"       : dep["mean"],
                "cov"        : dep["cov"],
                "stds"       : dep["stds"],
                "cov_inv"    : dep["cov_inv"],
                "active_idx" : insp_idx,
                "sub_mean"   : insp_sub_mean,
                "sub_cov"    : insp_sub_cov,
                "sub_stds"   : insp_sub_stds,
                "trunc_a"    : trunc_a,
                "trunc_b"    : trunc_b,
                "n_rows"     : dep["n_rows"],
            }
            print(f"    {'Inspection':12s}: inherits Deposition manifold "
                  f"(active_sensors={len(insp_idx)}, proxy)")
        else:
            self.stage_stats["Inspection"] = None


    # -----------------------------------------------------------------------
    # WM-811K Pattern Weight Learning
    # -----------------------------------------------------------------------

    def _learn_pattern_weights(self) -> None:
        """
        Count images per defect-pattern class in the WM-811K folder and
        store the resulting frequency weights in self.pattern_weights.
        These weights are used when selecting a matching image for a
        simulated wafer label.
        """
        if not os.path.exists(self.image_dir):
            self.pattern_weights = {}
            print(f"\n  WM-811K folder not found — image stream disabled.")
            return

        pattern_counts  = {}
        total_defective = 0
        for cls in os.listdir(self.image_dir):
            cls_path = os.path.join(self.image_dir, cls)
            if os.path.isdir(cls_path) and cls != "none":
                count = len([
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ])
                pattern_counts[cls]  = count
                total_defective     += count

        self.pattern_weights = (
            {cls: round(c / total_defective, 4) for cls, c in pattern_counts.items()}
            if total_defective > 0
            else {}
        )

        print(f"\n  Pattern weights from WM-811K ({total_defective:,} images):")
        for cls, w in sorted(self.pattern_weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(w * 30)
            print(f"    {cls:15s}: {w:.3f}  {bar}")


    # -----------------------------------------------------------------------
    # Physical Bounds Enforcement
    # -----------------------------------------------------------------------

    def _apply_physical_bounds(self, wafer: dict) -> dict:
        """Clip all sensor values to their hard physical limits."""
        for col, (lo, hi) in PHYSICAL_BOUNDS.items():
            if col in wafer:
                wafer[col] = float(np.clip(wafer[col], lo, hi))
        return wafer


    # -----------------------------------------------------------------------
    # Stage Baseline Sampling
    # -----------------------------------------------------------------------

    def _sample_stage_baseline(self, stage: str) -> dict:
        """
        Sample sensor readings for one stage-pass from the stage-specific
        active-sensor subspace manifold.

        Active sensors are drawn from multivariate_normal(sub_mean, sub_cov)
        using rejection sampling to enforce physical bounds. If rejection
        sampling exhausts MAX_TRIES, independent truncated Gaussians are used
        as a fallback (physical bounds guaranteed; inter-sensor correlations
        lost). Inactive sensors are pinned to their stage mean.

        Tool_Type is assigned deterministically from STAGE_TO_TOOL_TYPE so
        that the merged sensor row always carries the correct tool name.

        Returns
        -------
        dict : {sensor_name: float, "Tool_Type": str}
        """
        stats     = self.stage_stats.get(stage)
        active    = STAGE_ACTIVE_SENSORS[stage]
        MAX_TRIES = 50

        if stats is not None:
            sub_mean   = stats["sub_mean"]
            sub_cov    = stats["sub_cov"]
            sub_stds   = stats["sub_stds"]
            active_idx = stats["active_idx"]
            trunc_a    = stats["trunc_a"]
            trunc_b    = stats["trunc_b"]
            full_mean  = stats["mean"]
        else:
            # Sparse-data fallback: use global stats for the active subspace
            active_idx = [self.numeric_cols.index(c) for c in active if c in self.numeric_cols]
            full_mean  = self.global_mean
            sub_mean   = self.global_mean[active_idx]
            sub_cov    = self.global_cov[np.ix_(active_idx, active_idx)]
            sub_stds   = self.global_stds[active_idx]
            trunc_a    = np.array([
                (PHYSICAL_BOUNDS[self.numeric_cols[k]][0] - sub_mean[i]) / sub_stds[i]
                if sub_stds[i] > 0 else 0.0
                for i, k in enumerate(active_idx)
            ])
            trunc_b    = np.array([
                (PHYSICAL_BOUNDS[self.numeric_cols[k]][1] - sub_mean[i]) / sub_stds[i]
                if sub_stds[i] > 0 else np.inf
                for i, k in enumerate(active_idx)
            ])

        # Rejection sampling in the active subspace
        active_values = None
        for _ in range(MAX_TRIES):
            candidate = np.random.multivariate_normal(sub_mean, sub_cov)
            if all(
                PHYSICAL_BOUNDS[self.numeric_cols[active_idx[i]]][0]
                <= candidate[i]
                <= PHYSICAL_BOUNDS[self.numeric_cols[active_idx[i]]][1]
                for i in range(len(active_idx))
            ):
                active_values = candidate
                break

        if active_values is None:
            # Fallback: independent truncated Gaussians (rare)
            active_values = np.array([
                truncnorm.rvs(
                    a=trunc_a[i], b=trunc_b[i],
                    loc=sub_mean[i], scale=sub_stds[i],
                )
                for i in range(len(active_idx))
            ])

        # Build the full sensor dict
        wafer = {}
        for k, col in enumerate(self.numeric_cols):
            if k in active_idx:
                wafer[col] = float(active_values[active_idx.index(k)])
            else:
                wafer[col] = float(full_mean[k])

        wafer["Tool_Type"] = STAGE_TO_TOOL_TYPE[stage]
        return wafer


    # -----------------------------------------------------------------------
    # Tool Drift Processing
    # -----------------------------------------------------------------------

    def _process_tool_drift(
        self, wafer: dict, stage: str
    ) -> tuple[dict, "ToolDriftState", str, str | None]:
        """
        Manage the drift episode for the tool assigned to this stage.

        If no drift is active, a new episode may start with probability
        self.drift_probability. The episode duration is capped so the
        drifting sensor never breaches its physical bound.

        One drift increment is applied per wafer. The drifting sensor name
        is captured before any potential episode reset so that downstream
        fault injection can use it reliably (the episode may end on this
        very wafer).

        Parameters
        ----------
        wafer : dict  – current sensor readings (modified in place)
        stage : str   – current pipeline stage

        Returns
        -------
        wafer           : dict            – updated sensor readings
        tds             : ToolDriftState  – persistent drift state object
        tool_name       : str             – e.g. "Etching"
        drifting_sensor : str | None      – sensor name snapshot before reset
        """
        tool_name = self.stage_to_tool[stage]
        tds       = self.tool_drift[tool_name]
        stats     = self.stage_stats.get(stage)
        active    = STAGE_ACTIVE_SENSORS[stage]

        # Possibly start a new drift episode
        if not tds.active and random.random() < self.drift_probability:
            active_idxs = [k for k, col in enumerate(self.numeric_cols) if col in active]
            k           = random.choice(active_idxs)
            col         = self.numeric_cols[k]

            sigma = (
                float(np.sqrt(stats["cov"][k, k]))
                if stats else float(self.global_stds[k])
            )

            direction      = random.choice([+1, -1])
            per_wafer_step = direction * sigma * random.uniform(0.05, 0.12)
            natural_dur    = random.randint(20, 50)

            # Truncate episode duration to stay within physical bounds
            lo, hi       = PHYSICAL_BOUNDS.get(col, (-np.inf, np.inf))
            current_val  = wafer[col]
            bound        = hi if direction > 0 else lo
            headroom     = abs(bound - current_val)
            step_abs     = abs(per_wafer_step)

            if step_abs > 0:
                discriminant = 1 + 8 * headroom / step_abs
                max_safe     = max(1, int(((-1 + discriminant ** 0.5) / 2) - 1))
            else:
                max_safe = natural_dur

            tds.active      = True
            tds.sensor_col  = col
            tds.sensor_idx  = k
            tds.per_wafer   = per_wafer_step
            tds.accumulated = 0.0
            tds.remaining   = min(natural_dur, max_safe)
            tds.magnitude   = 0.0

            arrow = "↑" if direction > 0 else "↓"
            print(
                f"  TOOL DRIFT started | tool: {tool_name} | "
                f"sensor: '{col}' | {arrow} {abs(tds.per_wafer):.4f}/wafer "
                f"| duration: {tds.remaining} wafers"
            )

        # Snapshot the drifting sensor before any decrement or reset
        drifting_sensor = tds.sensor_col if tds.active else None

        # Apply one increment if drift is active
        if tds.active:
            tds.accumulated       += tds.per_wafer
            wafer[tds.sensor_col] += tds.accumulated
            tds.remaining         -= 1

            sigma = (
                float(np.sqrt(stats["cov"][tds.sensor_idx, tds.sensor_idx]))
                if stats else float(self.global_stds[tds.sensor_idx])
            )
            tds.magnitude = min(abs(tds.accumulated) / (3.0 * sigma), 1.0)

            if tds.remaining <= 0:
                direction_word = "above" if tds.accumulated > 0 else "below"
                print(
                    f"  TOOL DRIFT ended   | tool: {tool_name} | "
                    f"sensor: '{tds.sensor_col}' "
                    f"| total shift: {tds.accumulated:+.3f} ({direction_word} baseline)"
                )
                tds.reset()

        return wafer, tds, tool_name, drifting_sensor


    # -----------------------------------------------------------------------
    # Stage Fault Injection
    # -----------------------------------------------------------------------

    def _inject_stage_fault(
        self,
        wafer: dict,
        stage: str,
        tds: ToolDriftState,
        drifting_sensor: str | None,
    ) -> tuple[dict, str, str]:
        """
        Inject a realistic fault into the wafer sensor readings.

        When a sensor is drifting (drifting_sensor is not None), only faults
        that target that specific sensor are eligible. This causally links
        the drift mechanism to the fault type (e.g. a drifting RF_Power can
        only produce an rf_power_drop fault, not a gas_starvation fault).
        If no fault targets the drifting sensor (can happen at Inspection),
        the full catalogue is used as fallback.

        Parameters
        ----------
        wafer           : dict             – sensor readings to modify
        stage           : str              – current pipeline stage
        tds             : ToolDriftState   – used for log message only
        drifting_sensor : str | None       – pre-reset snapshot of sensor name

        Returns
        -------
        wafer        : dict  – modified sensor readings
        fault_label  : str   – e.g. "rf_power_drop"
        fault_detail : str   – human-readable description for logging
        """
        all_faults = STAGE_FAULTS[stage]

        if drifting_sensor is not None:
            constrained = [f for f in all_faults if f[1] == drifting_sensor]
            eligible    = constrained if constrained else all_faults
        else:
            eligible = all_faults

        label, col, scale_fn = random.choice(eligible)

        original   = wafer[col]
        wafer[col] = scale_fn(original)

        fault_detail = (
            f"FAULT [{stage}]: {label} | {col}: {original:.3f} → {wafer[col]:.3f}"
            + (f" [drift-constrained on {drifting_sensor}]" if drifting_sensor else "")
        )
        print(f"  {fault_detail}")

        return wafer, label, fault_detail


    # -----------------------------------------------------------------------
    # Stage State Assignment
    # -----------------------------------------------------------------------

    def _assign_stage_state(
        self,
        wafer: dict,
        stage: str,
        tds: ToolDriftState,
        drifting_sensor: str | None,
    ) -> tuple[dict, str, str]:
        """
        Determine the exclusive WaferState for one stage-pass and
        optionally inject a fault.

        The effective fault probability is raised when the tool is drifting:
          effective_prob = base + drift_magnitude × (1 − base) × 0.60

        Parameters
        ----------
        wafer           : dict
        stage           : str
        tds             : ToolDriftState
        drifting_sensor : str | None  – pre-reset snapshot

        Returns
        -------
        wafer      : dict  – (possibly modified) sensor readings
        fault_type : str   – fault label or "none"
        state      : str   – WaferState constant
        """
        fault_type = "none"

        effective_fault_prob = (
            self.fault_probability
            + tds.magnitude * (1.0 - self.fault_probability) * 0.60
        )

        if random.random() < effective_fault_prob:
            wafer, fault_type, _ = self._inject_stage_fault(
                wafer, stage, tds, drifting_sensor
            )

        triggered_by_drift = (
            drifting_sensor is not None
            and tds.magnitude > 0.15
            and fault_type != "none"
        )

        if fault_type != "none":
            state = WaferState.FAULT_IN_DRIFT if triggered_by_drift else WaferState.FAULT
        elif tds.active:
            state = WaferState.DRIFT
        else:
            state = WaferState.NORMAL

        return wafer, fault_type, state


    # -----------------------------------------------------------------------
    # Ground-Truth Label Derivation
    # -----------------------------------------------------------------------

    def _derive_ground_truth_label(
        self, stage: str, fault_type: str, wafer_state: str
    ) -> str:
        """
        Sample a wafer map pattern label from the stage × fault distribution.

        Normal wafers always receive the label "none". Drifting wafers (no
        fault yet) draw from the "drift" distribution for their stage.

        Returns
        -------
        str – e.g. "Edge Ring", "Center", "none"
        """
        if wafer_state == WaferState.NORMAL:
            return "none"

        effective_fault = "drift" if wafer_state == WaferState.DRIFT else fault_type
        dist = STAGE_FAULT_PATTERNS.get(stage, {}).get(effective_fault, {"none": 1.0})
        return random.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0]


    # -----------------------------------------------------------------------
    # Mahalanobis Anomaly Detection
    # -----------------------------------------------------------------------

    def _check_anomaly(
        self, wafer: dict, stage: str
    ) -> tuple[bool, float, str | None]:
        """
        Compute the Mahalanobis distance between this wafer's sensor vector
        and the stage-specific operating manifold.

        A wafer is flagged as anomalous if its distance exceeds the 95th
        percentile of the chi-squared distribution for 10 degrees of freedom.

        Returns
        -------
        is_anomaly : bool
        distance   : float
        warning    : str | None  – message if anomalous, else None
        """
        stats   = self.stage_stats.get(stage)
        mean    = stats["mean"]    if stats else self.global_mean
        cov_inv = stats["cov_inv"] if stats else self.global_cov_inv

        x = np.array([wafer[col] for col in self.numeric_cols])
        try:
            dist = mahalanobis(x, mean, cov_inv)
        except Exception:
            dist = 0.0

        is_anomaly = dist > self.mahal_threshold
        warning    = (
            f"Mahal dist {dist:.3f} > {self.mahal_threshold:.3f} "
            f"[{stage} manifold] — anomaly"
        ) if is_anomaly else None

        return is_anomaly, float(dist), warning


    # -----------------------------------------------------------------------
    # WM-811K Image Lookup
    # -----------------------------------------------------------------------

    def _get_matching_image(self, simulator_label: str) -> str | None:
        """
        Return a random image path from the WM-811K folder that matches the
        given simulator label. Falls back to the "none" folder if the label
        directory does not exist or is empty.
        """
        if not os.path.exists(self.image_dir):
            return None

        pattern_dir = os.path.join(self.image_dir, simulator_label)
        if not os.path.exists(pattern_dir):
            pattern_dir = os.path.join(self.image_dir, "none")

        images = [
            f for f in os.listdir(pattern_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        return os.path.join(pattern_dir, random.choice(images)) if images else None


    # -----------------------------------------------------------------------
    # Full Wafer Pipeline
    # -----------------------------------------------------------------------

    def _run_wafer_through_pipeline(self, wafer_id: int) -> dict:
        """
        Run one wafer through all four fabrication stages in sequence.

        At each stage the simulator samples, drifts, injects faults, enforces
        bounds, derives a label, and checks for anomalies. Sensor readings
        are merged into a single row (last active-stage value for active
        sensors; stage mean for inactive sensors). The worst outcome across
        all stages determines the overall label, fault type, and state.

        Returns
        -------
        dict with keys:
          sensors         – pd.DataFrame (1 row)
          image_path      – str | None
          simulator_label – str  (worst-stage pattern label)
          fault_type      – str  (worst-stage fault label)
          wafer_state     – str  (WaferState constant)
          stage_results   – list[dict]  (per-stage breakdown)
          wafer_id        – int
          mahal_distance  – float  (max across stages)
          is_anomaly      – bool
          drift_magnitude – float  (max across stages)
        """
        print(f"\n{'═' * 65}")
        print(f"  WAFER #{wafer_id:04d}  |  Time: {datetime.now().strftime('%H:%M:%S')}")

        stage_results  = []
        merged_sensors = {}

        STATE_PRIORITY = {
            WaferState.FAULT_IN_DRIFT : 3,
            WaferState.FAULT          : 2,
            WaferState.DRIFT          : 1,
            WaferState.NORMAL         : 0,
        }

        worst_state   = WaferState.NORMAL
        worst_label   = "none"
        worst_fault   = "none"
        max_mahal     = 0.0
        max_drift_mag = 0.0

        for stage in PROCESS_STAGES:
            print(f"  ── Stage: {stage}")

            wafer                               = self._sample_stage_baseline(stage)
            wafer, tds, tool_name, drift_sensor = self._process_tool_drift(wafer, stage)
            wafer, fault_type, stage_state      = self._assign_stage_state(wafer, stage, tds, drift_sensor)
            wafer                               = self._apply_physical_bounds(wafer)
            sim_label                           = self._derive_ground_truth_label(stage, fault_type, stage_state)
            is_anomaly, mahal_dist, warning     = self._check_anomaly(wafer, stage)

            if warning:
                print(f"    ⚠  {warning}")

            # Track worst outcome across stages
            if STATE_PRIORITY[stage_state] > STATE_PRIORITY[worst_state]:
                worst_state = stage_state
                worst_label = sim_label
                worst_fault = fault_type

            if sim_label != "none" and worst_label == "none":
                worst_label = sim_label

            max_mahal     = max(max_mahal,     mahal_dist)
            max_drift_mag = max(max_drift_mag, tds.magnitude)

            stage_results.append({
                "stage"           : stage,
                "tool"            : tool_name,
                "fault_type"      : fault_type,
                "wafer_state"     : stage_state,
                "simulator_label" : sim_label,
                "mahal_distance"  : round(mahal_dist, 4),
                "is_anomaly"      : is_anomaly,
                "drift_active"    : tds.active,
                "drift_sensor"    : tds.sensor_col,
                "drifting_sensor" : drift_sensor,
                "drift_magnitude" : round(tds.magnitude, 4),
            })

            # Merge sensor readings into the final row
            active = STAGE_ACTIVE_SENSORS[stage]
            stats  = self.stage_stats.get(stage)
            for col in self.numeric_cols:
                if col in active:
                    merged_sensors[col] = wafer[col]
                elif col not in merged_sensors:
                    k = self.numeric_cols.index(col)
                    merged_sensors[col] = (
                        float(stats["mean"][k]) if stats else float(self.global_mean[k])
                    )

            merged_sensors["Tool_Type"] = wafer["Tool_Type"]

        all_cols  = self.numeric_cols + ["Tool_Type"]
        sensor_df = pd.DataFrame([merged_sensors])[all_cols]
        image_path = self._get_matching_image(worst_label)

        return {
            "sensors"         : sensor_df,
            "image_path"      : image_path,
            "simulator_label" : worst_label,
            "fault_type"      : worst_fault,
            "wafer_state"     : worst_state,
            "stage_results"   : stage_results,
            "wafer_id"        : wafer_id,
            "mahal_distance"  : round(max_mahal, 4),
            "is_anomaly"      : max_mahal > self.mahal_threshold,
            "drift_magnitude" : round(max_drift_mag, 4),
        }


    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate(self) -> dict:
        """Generate one complete wafer (all four stages). Returns a result dict."""
        self.wafer_count += 1
        return self._run_wafer_through_pipeline(self.wafer_count)

    def stream(
        self,
        orchestrator_fn,
        interval_seconds: float = 0,
        max_wafers: int | None = None,
        batch_size: int = 50,
    ) -> list[pd.DataFrame]:
        """
        Continuously generate wafers in batches and pass each batch to
        an orchestrator callback.

        The orchestrator receives:
          batch_df       : pd.DataFrame  – merged sensor rows for the batch
          batch_images   : list[str]     – matching WM-811K image paths
          batch_patterns : list[str]     – ground-truth pattern labels
          batch_faults   : list[str]     – fault labels
          batch_ids      : list[int]     – wafer ID integers
          batch_meta     : list[dict]    – full per-wafer metadata including
                                          stage_results for stage-level attribution

        The stream stops when max_wafers is reached or the user presses Ctrl+C.
        All orchestrator results are concatenated and saved to
        digital_twin_results.csv before returning.

        Parameters
        ----------
        orchestrator_fn  : callable  – the orchestrator function to call per batch
        interval_seconds : float     – sleep time between wafers (0 = as fast as possible)
        max_wafers       : int       – stop after this many wafers (None = run forever)
        batch_size       : int       – number of wafers per batch (default 50)

        Returns
        -------
        list[pd.DataFrame] – one DataFrame per batch returned by the orchestrator
        """
        print("\n" + "=" * 65)
        print("  DIGITAL TWIN STREAM STARTED")
        print(f"  Batch size   : {batch_size} wafers × 4 stages each")
        print(f"  Fault chance : {self.fault_probability * 100:.0f}% per stage")
        print(f"  Drift chance : {self.drift_probability * 100:.0f}% per tool")
        if max_wafers:
            print(f"  Stop after   : {max_wafers} wafers")
        print("=" * 65)

        all_results  = []
        risk_counts  = {"low": 0, "high": 0}
        state_counts = Counter()

        try:
            while True:
                batch_sensors  = []
                batch_images   = []
                batch_patterns = []
                batch_faults   = []
                batch_ids      = []
                batch_meta     = []

                for _ in range(batch_size):
                    w = self.generate()
                    batch_sensors.append(w["sensors"])
                    batch_images.append(w["image_path"])
                    batch_patterns.append(w["simulator_label"])
                    batch_faults.append(w["fault_type"])
                    batch_ids.append(w["wafer_id"])
                    batch_meta.append({
                        "wafer_id"        : w["wafer_id"],
                        "wafer_state"     : w["wafer_state"],
                        "process_stage"   : "pipeline",
                        "simulator_label" : w["simulator_label"],
                        "fault_type"      : w["fault_type"],
                        "mahal_distance"  : w["mahal_distance"],
                        "is_anomaly"      : w["is_anomaly"],
                        "drift_magnitude" : w["drift_magnitude"],
                        "stage_results"   : w["stage_results"],
                    })
                    state_counts[w["wafer_state"]] += 1
                    time.sleep(interval_seconds)

                    if max_wafers and self.wafer_count >= max_wafers:
                        break

                batch_df = pd.concat(batch_sensors, ignore_index=True)

                result = orchestrator_fn(
                    batch_df,
                    batch_images   = batch_images,
                    batch_patterns = batch_patterns,
                    batch_faults   = batch_faults,
                    batch_ids      = batch_ids,
                    batch_meta     = batch_meta,
                )

                if result is not None and not result.empty:
                    risk_counts["high"] += len(result)
                    risk_counts["low"]  += batch_size - len(result)
                    all_results.append(result)
                else:
                    risk_counts["low"] += batch_size

                meta_df       = pd.DataFrame(batch_meta)
                state_dist    = meta_df["wafer_state"].value_counts()
                anomaly_count = int(meta_df["is_anomaly"].sum())

                print(f"\n{'=' * 65}")
                print(f"  BATCH COMPLETE | Wafers #{batch_ids[0]}–#{batch_ids[-1]}")
                print(f"  State dist     : {dict(state_dist)}")
                print(f"  Mahal anomalies: {anomaly_count} / {batch_size}")
                print(f"  High-risk total: {risk_counts['high']}")
                print(f"{'=' * 65}")

                if max_wafers and self.wafer_count >= max_wafers:
                    print(f"\n  Reached {max_wafers} wafers. Stopping.")
                    break

        except KeyboardInterrupt:
            print("\n\n  Stream stopped by user (Ctrl+C).")

        finally:
            print(f"\n{'─' * 50}")
            print(f"  TOTAL wafers    : {self.wafer_count}")
            print(f"  High-risk       : {risk_counts['high']}")
            print(f"  Normal          : {risk_counts['low']}")
            print(f"  State breakdown : {dict(state_counts)}")

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                final_df.to_csv("digital_twin_results.csv", index=False)
                print("  Saved to        : digital_twin_results.csv")

        return all_results


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def mock_orchestrator(
        X_input,
        batch_images=None,
        batch_patterns=None,
        batch_faults=None,
        batch_ids=None,
        batch_meta=None,
    ):
        """Minimal orchestrator for smoke-testing the simulator in isolation."""
        print(f"\n  [mock] batch shape: {X_input.shape}")
        if batch_meta:
            for m in batch_meta[:2]:
                print(
                    f"    Wafer #{m['wafer_id']:04d} | state: {m['wafer_state']:15s} | "
                    f"label: {m['simulator_label']:12s} | mahal: {m['mahal_distance']:.3f} | "
                    f"drift: {m['drift_magnitude']:.3f}"
                )
                for sr in m.get("stage_results", []):
                    print(
                        f"      {sr['stage']:12s} | fault: {sr['fault_type']:20s} | "
                        f"state: {sr['wafer_state']:15s} | label: {sr['simulator_label']:12s} | "
                        f"drift_active: {sr['drift_active']}  mag={sr['drift_magnitude']:.3f} | "
                        f"drifting_sensor: {sr['drifting_sensor']}"
                    )
        return pd.DataFrame()

    twin = DigitalTwinSimulator(
        csv_path  = r"D:\MS\venv\synthetic_explicit.csv",
        image_dir = r"D:\MS\venv\WM811k_Dataset",
    )

    twin.stream(mock_orchestrator, interval_seconds=0, max_wafers=4, batch_size=2)

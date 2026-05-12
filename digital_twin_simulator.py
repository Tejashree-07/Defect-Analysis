# ============================================================
# DIGITAL TWIN SIMULATOR
# Built specifically for synthetic_explicit.csv
# ============================================================
#
# Your dataset has 42,190 wafers with these process parameters:
#
#   Chamber_Temperature   — furnace temp in Celsius    (~77C avg)
#   Gas_Flow_Rate         — gas feed rate              (~50 units)
#   RF_Power              — radio frequency power      (~355 units)
#   Etch_Depth            — how deep the etch goes     (~400 units)
#   Rotation_Speed        — wafer spin speed           (~1600 RPM)
#   Vacuum_Pressure       — chamber vacuum             (~0.6 units)
#   Stage_Alignment_Error — how far off-center         (~1.6 units)
#   Vibration_Level       — mechanical vibration       (~0.015)
#   UV_Exposure_Intensity — light intensity for litho  (~110 units)
#   Particle_Count        — particles on wafer         (~479 avg)
#   Tool_Type             — Etching / Lithography / Deposition
#
# NOTE: Year/Month/Date/Hour/Minute are intentionally excluded.
# They hurt model performance (see AUC-ROC / F1 discussion)
# and carry no causal signal for defect prediction.
#
# Defect rate in real data: 40.2%
# ============================================================

import numpy as np
import pandas as pd
import time
import random
from datetime import datetime


class DigitalTwinSimulator:

    def __init__(self, csv_path, fault_probability=0.05, drift_probability=0.15):
        """
        Reads your CSV, learns its statistics, and uses them
        to generate realistic fake wafers forever.

        fault_probability : chance of a sudden spike per wafer  (5%)
        drift_probability : chance a slow drift starts per wafer (15%)
        """

        print("=" * 55)
        print("  DIGITAL TWIN - Learning from synthetic_explicit.csv")
        print("=" * 55)

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} real wafers to learn from")

        # --------------------------------------------------
        # The 10 process sensor columns we will simulate.
        # Defect and Join_Status are excluded (those are
        # labels, not inputs). Timestamps are excluded
        # because they hurt model performance.
        # --------------------------------------------------
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
            "Particle_Count"
        ]

        # --------------------------------------------------
        # LEARN THE STATISTICS from your real data.
        #
        # mean       = average value of each sensor
        # cov_matrix = how sensors move TOGETHER
        #
        # Example of what cov catches:
        #   If high RF_Power tends to increase Etch_Depth
        #   in your real data, the simulator will respect
        #   that relationship too. Random noise alone
        #   would not do this.
        # --------------------------------------------------
        numeric_data    = df[self.numeric_cols].copy()
        self.mean       = numeric_data.mean().values
        self.cov_matrix = numeric_data.cov().values

        # --------------------------------------------------
        # LEARN Tool_Type frequencies.
        # Real data: Etching 38%, Lithography 33%, Deposition 28%
        # We will sample with those same proportions.
        # --------------------------------------------------
        tool_counts      = df["Tool_Type"].value_counts(normalize=True)
        self.tool_types  = tool_counts.index.tolist()
        self.tool_probs  = tool_counts.values.tolist()

        # --------------------------------------------------
        # SAFE OPERATING RANGES per sensor.
        # These are the bounds within which a wafer is
        # considered "normal". Anything outside triggers
        # a warning in the output log.
        #
        # We define them as: mean +/- 2 standard deviations
        # (covers ~95% of your real data = "normal zone")
        # --------------------------------------------------
        stds = numeric_data.std().values
        self.safe_min = self.mean - 2 * stds
        self.safe_max = self.mean + 2 * stds

        print(f"  Numeric features : {len(self.numeric_cols)}")
        print(f"  Tool types       : {self.tool_types}")
        print(f"  Safe temp range  : "
              f"{self.safe_min[0]:.1f}C - {self.safe_max[0]:.1f}C")
        print(f"  Safe RF range    : "
              f"{self.safe_min[2]:.1f} - {self.safe_max[2]:.1f}")

        # --------------------------------------------------
        # DRIFT STATE
        # Only one drift episode can be active at a time.
        # It targets a specific sensor and creeps slowly.
        # --------------------------------------------------
        self.drift_active      = False
        self.drift_col_idx     = None   # which sensor is drifting
        self.drift_per_wafer   = 0.0    # how much it shifts per wafer
        self.drift_remaining   = 0      # wafers left in this episode
        self.drift_accumulated = 0.0    # total shift so far

        # --------------------------------------------------
        # SIMULATION CLOCK
        # We generate realistic timestamps starting from now.
        # Each wafer takes ~30 seconds of simulated fab time.
        # --------------------------------------------------
        self.wafer_count = 0

        self.fault_probability = fault_probability
        self.drift_probability = drift_probability

        print("=" * 55)
        print("  Simulator ready.\n")


    # ==============================================================
    # INTERNAL: Generate a baseline (healthy) wafer
    # ==============================================================
    def _sample_baseline(self):
        """
        Samples all 10 sensors simultaneously using the
        multivariate normal distribution learned from your CSV.

        'Multivariate' = all sensors sampled together,
        so their correlations are preserved.
        """
        values = np.random.multivariate_normal(self.mean, self.cov_matrix)

        wafer = {}
        for col, val in zip(self.numeric_cols, values):
            wafer[col] = val

        # Tool type sampled by real frequency
        wafer["Tool_Type"] = np.random.choice(
            self.tool_types, p=self.tool_probs
        )
        return wafer


    # ==============================================================
    # INTERNAL: Apply gradual drift to one sensor
    # ==============================================================
    def _apply_drift(self, wafer):
        """
        Drift = one sensor slowly wandering away from normal.

        Real-world cause: furnace element aging, gas valve
        slowly clogging, mechanical wear on rotation stage.

        Once started, it runs for 20-50 wafers.
        The sensor reading shifts by a small fixed amount
        every wafer -- small enough that any single wafer
        looks normal, but the trend is detectable.
        """

        # Maybe START a new drift episode
        if not self.drift_active:
            if random.random() < self.drift_probability:

                self.drift_active    = True
                self.drift_col_idx   = random.randint(0, len(self.numeric_cols) - 1)
                self.drift_remaining = random.randint(20, 50)

                # Drift amount = small fraction of that sensor's std dev
                # Direction is random: +1 (creeping up) or -1 (creeping down)
                # A furnace can overshoot OR undershoot calibration equally
                sensor_std            = np.sqrt(self.cov_matrix[
                    self.drift_col_idx, self.drift_col_idx
                ])
                direction             = random.choice([+1, -1])
                self.drift_per_wafer  = direction * sensor_std * random.uniform(0.05, 0.12)
                self.drift_accumulated = 0.0

                drifting        = self.numeric_cols[self.drift_col_idx]
                direction_label = "UP" if direction == 1 else "DOWN"
                print(f"  DRIFT started  | sensor: '{drifting}' "
                      f"| {direction_label} {abs(self.drift_per_wafer):.4f}/wafer "
                      f"| duration: {self.drift_remaining} wafers")

        # APPLY drift if active
        if self.drift_active:
            col = self.numeric_cols[self.drift_col_idx]
            self.drift_accumulated += self.drift_per_wafer
            wafer[col]             += self.drift_accumulated
            self.drift_remaining   -= 1

            if self.drift_remaining <= 0:
                direction_word = "above" if self.drift_accumulated > 0 else "below"
                print(f"  DRIFT ended    | sensor: '{col}' "
                      f"| total shift: {self.drift_accumulated:.3f} ({direction_word} baseline)")
                self.drift_active      = False
                self.drift_accumulated = 0.0

        return wafer


    # ==============================================================
    # INTERNAL: Inject a sudden fault on one sensor
    # ==============================================================
    def _inject_fault(self, wafer):
        """
        Fault = sudden spike on one sensor for a single wafer.

        Real-world cause: momentary gas valve stutter,
        RF power supply glitch, particle contamination event,
        mechanical vibration from nearby equipment.

        Each fault type targets a physically relevant sensor:
          - Temperature fault  -> Chamber_Temperature spikes
          - Particle burst     -> Particle_Count jumps
          - Power fault        -> RF_Power drops
          - Vacuum loss        -> Vacuum_Pressure rises (worse vacuum)
          - Vibration event    -> Vibration_Level spikes
        """

        fault_types = [
            "temperature_spike",
            "particle_burst",
            "rf_power_drop",
            "vacuum_loss",
            "vibration_event"
        ]
        fault = random.choice(fault_types)

        if fault == "temperature_spike":
            original = wafer["Chamber_Temperature"]
            wafer["Chamber_Temperature"] *= random.uniform(1.1, 1.25)
            print(f"  FAULT: temp spike      | "
                  f"{original:.1f}C -> {wafer['Chamber_Temperature']:.1f}C")

        elif fault == "particle_burst":
            original = wafer["Particle_Count"]
            wafer["Particle_Count"] += random.randint(300, 700)
            print(f"  FAULT: particle burst  | "
                  f"{original:.0f} -> {wafer['Particle_Count']:.0f} particles")

        elif fault == "rf_power_drop":
            original = wafer["RF_Power"]
            wafer["RF_Power"] *= random.uniform(0.5, 0.75)
            print(f"  FAULT: RF power drop   | "
                  f"{original:.1f} -> {wafer['RF_Power']:.1f}")

        elif fault == "vacuum_loss":
            original = wafer["Vacuum_Pressure"]
            wafer["Vacuum_Pressure"] *= random.uniform(1.5, 3.0)
            print(f"  FAULT: vacuum loss     | "
                  f"{original:.3f} -> {wafer['Vacuum_Pressure']:.3f}")

        elif fault == "vibration_event":
            original = wafer["Vibration_Level"]
            wafer["Vibration_Level"] *= random.uniform(3.0, 8.0)
            print(f"  FAULT: vibration       | "
                  f"{original:.4f} -> {wafer['Vibration_Level']:.4f}")

        return wafer


    # ==============================================================
    # INTERNAL: Check if any sensor is outside safe range
    # ==============================================================
    def _check_out_of_range(self, wafer):
        """
        After fault/drift are applied, check which sensors
        have gone outside the normal operating window.
        Returns a list of warning strings (empty if all normal).
        """
        warnings = []
        for i, col in enumerate(self.numeric_cols):
            val = wafer[col]
            if val < self.safe_min[i] or val > self.safe_max[i]:
                warnings.append(
                    f"{col}: {val:.3f} "
                    f"(safe: {self.safe_min[i]:.2f} - {self.safe_max[i]:.2f})"
                )
        return warnings


    # ==============================================================
    # PUBLIC: Generate one wafer
    # ==============================================================
    def generate(self):
        """
        Generates one wafer as a single-row DataFrame.
        This is the same format as X_test in your orchestrator --
        so your existing code needs ZERO changes.

        Returns: pd.DataFrame with 1 row, all feature columns
        """

        self.wafer_count += 1

        print(f"\n{'-'*55}")
        print(f"  Wafer #{self.wafer_count:04d}  |  "
              f"Real time: {datetime.now().strftime('%H:%M:%S')}")

        # Step 1: start with a normal wafer
        wafer = self._sample_baseline()

        # Step 2: maybe apply slow drift
        wafer = self._apply_drift(wafer)

        # Step 3: maybe inject a sudden fault
        if random.random() < self.fault_probability:
            wafer = self._inject_fault(wafer)

        # Step 4: check what is out of range
        warnings = self._check_out_of_range(wafer)
        if warnings:
            print(f"  WARNING: {len(warnings)} sensor(s) out of safe range:")
            for w in warnings:
                print(f"    -> {w}")

        # Step 5: return as single-row DataFrame
        # NOTE: timestamps (Year/Month/Date/Hour/Minute) are intentionally
        # excluded — your original code drops them from analysis anyway,
        # and your preprocessor only needs the process sensor columns.
        all_cols = self.numeric_cols + ["Tool_Type"]
        return pd.DataFrame([wafer])[all_cols]


    # ==============================================================
    # PUBLIC: Stream forever, feeding into your orchestrator
    # ==============================================================
    def stream(self, orchestrator_fn, interval_seconds=3, max_wafers=None):
        """
        Runs the digital twin continuously.
        Calls your existing orchestrator() on every wafer.

        orchestrator_fn  : your existing orchestrator function
        interval_seconds : how fast to generate wafers
        max_wafers       : stop after N wafers (None = run forever)
        """

        print("\n" + "=" * 55)
        print("  DIGITAL TWIN STREAM STARTED")
        print(f"  Speed          : 1 wafer every {interval_seconds}s")
        print(f"  Fault chance   : {self.fault_probability*100:.0f}% per wafer")
        print(f"  Drift chance   : {self.drift_probability*100:.0f}% per wafer")
        if max_wafers:
            print(f"  Will stop after: {max_wafers} wafers")
        else:
            print(f"  Press Ctrl+C to stop")
        print("=" * 55)

        all_results = []
        risk_counts = {"low": 0, "high": 0}

        try:
            while True:

                # Collect 10 wafers into a batch first.
                # Your orchestrator uses np.percentile(defect_probs, 80)
                # which needs multiple rows to be meaningful.
                # With 1 row, every wafer flags as high risk -- wrong.
                # With 10 rows, only the genuinely worst ones flag.
                batch = []
                for _ in range(10):
                    wafer_df = self.generate()
                    batch.append(wafer_df)

                # Stack all 10 into one DataFrame -- same shape as X_test
                batch_df = pd.concat(batch, ignore_index=True)
                result = orchestrator_fn(batch_df)

                # Log result
                if result is not None and not result.empty:
                    risk_counts["high"] += 1
                    all_results.append(result)
                    print(f"  HIGH RISK  | "
                          f"defect_prob: {result['defect_prob'].values[0]:.3f} | "
                          f"High-risk total: {risk_counts['high']}")
                else:
                    risk_counts["low"] += 1
                    print(f"  NORMAL     | Normal total: {risk_counts['low']}")

                # Stop if max reached
                if max_wafers and self.wafer_count >= max_wafers:
                    print(f"\n  Reached {max_wafers} wafers. Stopping.")
                    break

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n  Stream stopped by user.")

        finally:
            print(f"\n  {'─'*45}")
            print(f"  SUMMARY")
            print(f"  Total wafers processed : {self.wafer_count}")
            print(f"  High-risk detected     : {risk_counts['high']}")
            print(f"  Normal wafers          : {risk_counts['low']}")

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                out_path = "digital_twin_results.csv"
                final_df.to_csv(out_path, index=False)
                print(f"  Results saved to       : {out_path}")

        return all_results


# ============================================================
# HOW TO PLUG THIS INTO YOUR EXISTING CODE
# ============================================================
#
# In your main script, replace the bottom section:
#
#   OLD (runs once on static data):
#   --------------------------------
#   result_df = orchestrator(X_test)
#   result_df.to_csv(output_log, index=False)
#   generate_charts(process_df=df, result_df=result_df)
#
#   NEW (runs forever as a live digital twin):
#   --------------------------------
#   from digital_twin_simulator import DigitalTwinSimulator
#
#   twin = DigitalTwinSimulator(
#       csv_path="synthetic_explicit.csv",
#       fault_probability=0.05,
#       drift_probability=0.15
#   )
#
#   # Option A: Stream forever
#   twin.stream(orchestrator, interval_seconds=3)
#
#   # Option B: Generate one wafer for testing
#   wafer = twin.generate()
#   print(wafer)
#
# Your orchestrator, preprocessor, models -- nothing changes.
# ============================================================


# ============================================================
# QUICK STANDALONE TEST
# Run this file directly:  python digital_twin_simulator.py
# ============================================================

if __name__ == "__main__":

    def mock_orchestrator(X_input):
        """
        Fake orchestrator for testing.
        Replace with your real orchestrator() when ready.
        """
        print(f"  Orchestrator got: shape={X_input.shape} | "
              f"Temp={X_input['Chamber_Temperature'].values[0]:.1f}C | "
              f"RF={X_input['RF_Power'].values[0]:.1f} | "
              f"Particles={X_input['Particle_Count'].values[0]:.0f}")
        return pd.DataFrame()   # empty = low risk

    twin = DigitalTwinSimulator(
        csv_path="synthetic_explicit.csv",
        fault_probability=0.30,   # high so you see faults in the demo
        drift_probability=0.40    # high so you see drift in the demo
    )

    print("\n=== Generating 8 demo wafers ===")
    twin.stream(mock_orchestrator, interval_seconds=1, max_wafers=30)

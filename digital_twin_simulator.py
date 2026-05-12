# ============================================================
# DIGITAL TWIN SIMULATOR
# ============================================================

import numpy as np
import pandas as pd
import time
import random
from datetime import datetime


class DigitalTwinSimulator:

    def __init__(self, csv_path, fault_probability=0.05, drift_probability=0.15):

        print("=" * 55)
        print("  DIGITAL TWIN - Learning from synthetic_explicit.csv")
        print("=" * 55)

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} real wafers to learn from")

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

        numeric_data    = df[self.numeric_cols].copy()
        self.mean       = numeric_data.mean().values
        self.cov_matrix = numeric_data.cov().values

        tool_counts      = df["Tool_Type"].value_counts(normalize=True)
        self.tool_types  = tool_counts.index.tolist()
        self.tool_probs  = tool_counts.values.tolist()
        
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
        # --------------------------------------------------
        self.drift_active      = False
        self.drift_col_idx     = None   # which sensor is drifting
        self.drift_per_wafer   = 0.0    # how much it shifts per wafer
        self.drift_remaining   = 0      # wafers left in this episode
        self.drift_accumulated = 0.0    # total shift so far

        # --------------------------------------------------
        # SIMULATION CLOCK
        # --------------------------------------------------
        self.wafer_count = 0

        self.fault_probability = fault_probability
        self.drift_probability = drift_probability

        print("=" * 55)
        print("  Simulator ready.\n")


    # ==============================================================
    # Generate a baseline (healthy) wafer
    # ==============================================================
    def _sample_baseline(self):
        values = np.random.multivariate_normal(self.mean, self.cov_matrix)

        wafer = {}
        for col, val in zip(self.numeric_cols, values):
            wafer[col] = val

        wafer["Tool_Type"] = np.random.choice(
            self.tool_types, p=self.tool_probs
        )
        return wafer


    def _apply_drift(self, wafer):

        if not self.drift_active:
            if random.random() < self.drift_probability:

                self.drift_active    = True
                self.drift_col_idx   = random.randint(0, len(self.numeric_cols) - 1)
                self.drift_remaining = random.randint(20, 50)

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


    def _inject_fault(self, wafer):

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

        self.wafer_count += 1

        print(f"\n{'-'*55}")
        print(f"  Wafer #{self.wafer_count:04d}  |  "
              f"Real time: {datetime.now().strftime('%H:%M:%S')}")
        wafer = self._sample_baseline()
        wafer = self._apply_drift(wafer)
        if random.random() < self.fault_probability:
            wafer = self._inject_fault(wafer)
        warnings = self._check_out_of_range(wafer)
        if warnings:
            print(f"  WARNING: {len(warnings)} sensor(s) out of safe range:")
            for w in warnings:
                print(f"    -> {w}")
        all_cols = self.numeric_cols + ["Tool_Type"]
        return pd.DataFrame([wafer])[all_cols]

    def stream(self, orchestrator_fn, interval_seconds=3, max_wafers=None):

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

                batch = []
                for _ in range(10):
                    wafer_df = self.generate()
                    batch.append(wafer_df)

                batch_df = pd.concat(batch, ignore_index=True)
                result = orchestrator_fn(batch_df)

                if result is not None and not result.empty:
                    risk_counts["high"] += 1
                    all_results.append(result)
                    print(f"  HIGH RISK  | "
                          f"defect_prob: {result['defect_prob'].values[0]:.3f} | "
                          f"High-risk total: {risk_counts['high']}")
                else:
                    risk_counts["low"] += 1
                    print(f"  NORMAL     | Normal total: {risk_counts['low']}")

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
# QUICK STANDALONE TEST
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
        return pd.DataFrame()  

    twin = DigitalTwinSimulator(
        csv_path="synthetic_explicit.csv",
        fault_probability=0.30,  
        drift_probability=0.40    
    )

    print("\n=== Generating 8 demo wafers ===")
    twin.stream(mock_orchestrator, interval_seconds=1, max_wafers=30)

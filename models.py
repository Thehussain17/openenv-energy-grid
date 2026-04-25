"""
Data models for the Energy Grid Environment — v2.1 (22-Dim Observation).

Action space  : 4-dimensional MultiDiscreteAction vector
Observation   : 22-element normalized float vector [0, 1]
State         : Extended GridState with per-sector and black-swan tracking

Observation vector — 8 functional blocks (22 total):
  Block 1  — Temporal Context         (fields  1-2 )  sin/cos time encoding
  Block 2  — Real-Time Generation     (fields  3-4 )  solar, wind normalized
  Block 3  — Energy Storage State     (fields  5-6 )  SoC, BESS health
  Block 4  — Sector Demand/Perf       (fields  7-12)  per-sector demand + served ratios
  Block 5  — Grid Physics             (fields 13-14)  frequency norm, grid import norm
  Block 6  — Emergency Flags          (fields 15-16)  solar/wind black-swan binary flags
  Block 7  — Episode Context          (fields 17-18)  step progress, shed budget ratio
  Block 8  — Predictive Forecasts     (fields 19-22)  solar/wind at t+1h and t+3h

OpenEnv type hierarchy:
  GridAction       extends Action         (+ metadata)
  GridObservation  extends Observation    (+ done, reward, metadata)
  GridState        extends State          (+ episode_id, step_count)
"""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


# ---------------------------------------------------------------------------
# Action Model — Multi-Discrete 4-Dim Vector
# ---------------------------------------------------------------------------

class GridAction(Action):
    """
    4-dimensional multi-discrete dispatch action.

    Dimensions
    ----------
    bess        int  BESS control
                     0=Idle  1=Charge25%  2=Charge50%
                     3=Discharge25%  4=Discharge50%  5=Discharge100%
    hospital    int  Hospital supply level
                     0=100%  1=98% (emergency safety margin only)
    industrial  int  Industrial supply level
                     0=100%  1=90%  2=80%  3=70%
    residential int  Residential supply level
                     0=100%  1=85%  2=70%  3=60%
    """

    bess: int = Field(
        default=0,
        ge=0,
        le=5,
        description=(
            "BESS control: 0=Idle, 1=Charge25%, 2=Charge50%, "
            "3=Discharge25%, 4=Discharge50%, 5=Discharge100%."
        ),
    )
    hospital: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Hospital supply level: 0=100%, 1=98% (emergency margin only).",
    )
    industrial: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Industrial supply level: 0=100%, 1=90%, 2=80%, 3=70%.",
    )
    residential: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Residential supply level: 0=100%, 1=85%, 2=70%, 3=60%.",
    )

    @field_validator("bess")
    @classmethod
    def _validate_bess(cls, v: int) -> int:
        if v not in range(6):
            raise ValueError(f"bess must be 0–5, got {v}")
        return v

    @field_validator("hospital")
    @classmethod
    def _validate_hospital(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"hospital must be 0 or 1, got {v}")
        return v

    @field_validator("industrial")
    @classmethod
    def _validate_industrial(cls, v: int) -> int:
        if v not in range(4):
            raise ValueError(f"industrial must be 0–3, got {v}")
        return v

    @field_validator("residential")
    @classmethod
    def _validate_residential(cls, v: int) -> int:
        if v not in range(4):
            raise ValueError(f"residential must be 0–3, got {v}")
        return v


# ---------------------------------------------------------------------------
# Action level → supply-ratio lookup tables
# ---------------------------------------------------------------------------

HOSPITAL_RATIOS:    List[float] = [1.00, 0.98]
INDUSTRIAL_RATIOS:  List[float] = [1.00, 0.90, 0.80, 0.70]
RESIDENTIAL_RATIOS: List[float] = [1.00, 0.85, 0.70, 0.60]

# BESS: (is_charge, fraction_of_max_power)
BESS_MAP: List[tuple] = [
    (None,  0.00),   # 0 Idle
    (True,  0.25),   # 1 Charge 25%
    (True,  0.50),   # 2 Charge 50%
    (False, 0.25),   # 3 Discharge 25%
    (False, 0.50),   # 4 Discharge 50%
    (False, 1.00),   # 5 Discharge 100%
]


# ---------------------------------------------------------------------------
# Observation Model — 22-element normalized vector + rich raw fields
# ---------------------------------------------------------------------------

class GridObservation(Observation):
    """
    Normalized 22-element observation vector for the energy-grid environment v2.1.

    All scalar fields are in [0, 1] unless noted.
    Inherits `done`, `reward`, `metadata` from Observation base.

    BLOCK LAYOUT
    ============
    Block 1  — Temporal Context       (2 fields,  idx 0-1)
    Block 2  — Real-Time Generation   (2 fields,  idx 2-3)
    Block 3  — Energy Storage State   (2 fields,  idx 4-5)
    Block 4  — Sector Demand/Perf     (6 fields,  idx 6-11)
    Block 5  — Grid Physics           (2 fields,  idx 12-13)
    Block 6  — Emergency Flags        (2 fields,  idx 14-15)
    Block 7  — Episode Context        (2 fields,  idx 16-17)
    Block 8  — Predictive Forecasts   (4 fields,  idx 18-21)
                                      TOTAL = 22
    """

    # ── Block 1: Temporal Context ───────────────────────────────────────────
    time_sin: float = Field(
        default=0.0,
        description="sin(2π * hour / 24). Encodes cyclic time without a discontinuity.",
    )
    time_cos: float = Field(
        default=1.0,
        description="cos(2π * hour / 24). Combined with time_sin gives full phase.",
    )

    # ── Block 2: Real-Time Generation ──────────────────────────────────────
    solar_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="P_solar / P_solar_max (200kW). Current solar output normalized.",
    )
    wind_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="P_wind / P_wind_max (150kW). Current wind output normalized.",
    )

    # ── Block 3: Energy Storage State ──────────────────────────────────────
    battery_soc: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="BESS State of Charge [0, 1]. Naturally dimensionless.",
    )
    bess_health: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="BESS cycle-life health proxy [0, 1]. Degrades from 1.0 to 0.0.",
    )

    # ── Block 4: Sector Demand & Performance ───────────────────────────────
    hosp_demand_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Hospital demand / 500kW (total microgrid capacity).",
    )
    hosp_served_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Hospital: Load_Served / Load_Demand. Must stay >= 0.95.",
    )
    ind_demand_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Industrial demand / 500kW.",
    )
    ind_served_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Industrial: Load_Served / Load_Demand.",
    )
    res_demand_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Residential demand / 500kW.",
    )
    res_served_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Residential: Load_Served / Load_Demand.",
    )

    # ── Block 5: Grid Physics & Stability ──────────────────────────────────
    frequency_norm: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "(frequency - 49.0) / 2.0. Maps 49–51 Hz → 0.0–1.0. "
            "0.5 = nominal 50 Hz."
        ),
    )
    grid_import_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Net grid import / 500kW. 0.0 = fully islanded, 1.0 = full import.",
    )

    # ── Block 6: Emergency & Stochastic Flags ──────────────────────────────
    solar_swan_active: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1.0 if solar-collapse black-swan is active, else 0.0.",
    )
    wind_swan_active: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1.0 if wind-failure black-swan is active, else 0.0.",
    )

    # ── Block 7: Episode Context & Constraints ─────────────────────────────
    step_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="step / 24. Normalized episode progress [0, 1].",
    )
    cumulative_shed_ratio_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "cumulative_shed_ratio / 0.20, clamped to 1.0. "
            "Values near 1.0 indicate the 20% shedding budget is almost exceeded."
        ),
    )

    # ── Block 8: Predictive Forecasts ──────────────────────────────────────
    forecast_solar_1h: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Noisy forecast of solar output 1 hour ahead, normalized by 200kW. "
            "10-15% Gaussian noise applied to force uncertainty management."
        ),
    )
    forecast_solar_3h: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Noisy forecast of solar output 3 hours ahead, normalized by 200kW. "
            "Higher noise than 1h due to extended horizon uncertainty."
        ),
    )
    forecast_wind_1h: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Noisy forecast of wind output 1 hour ahead, normalized by 150kW. "
            "10-15% Gaussian noise applied."
        ),
    )
    forecast_wind_3h: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Noisy forecast of wind output 3 hours ahead, normalized by 150kW. "
            "Higher noise than 1h due to extended horizon uncertainty."
        ),
    )

    def to_vector(self) -> List[float]:
        """
        Return the canonical 22-element observation vector.

        Order matches the 8-block specification exactly:
          [0-1]   Block 1: Temporal
          [2-3]   Block 2: Generation
          [4-5]   Block 3: BESS
          [6-11]  Block 4: Sectors (hosp, ind, res — demand + served each)
          [12-13] Block 5: Physics
          [14-15] Block 6: Flags
          [16-17] Block 7: Episode
          [18-21] Block 8: Forecasts (solar_1h, solar_3h, wind_1h, wind_3h)
        """
        return [
            # Block 1 — Temporal
            self.time_sin,
            self.time_cos,
            # Block 2 — Generation
            self.solar_norm,
            self.wind_norm,
            # Block 3 — BESS
            self.battery_soc,
            self.bess_health,
            # Block 4 — Sectors
            self.hosp_demand_norm,
            self.hosp_served_ratio,
            self.ind_demand_norm,
            self.ind_served_ratio,
            self.res_demand_norm,
            self.res_served_ratio,
            # Block 5 — Physics
            self.frequency_norm,
            self.grid_import_norm,
            # Block 6 — Flags
            self.solar_swan_active,
            self.wind_swan_active,
            # Block 7 — Episode
            self.step_norm,
            self.cumulative_shed_ratio_norm,
            # Block 8 — Forecasts
            self.forecast_solar_1h,
            self.forecast_solar_3h,
            self.forecast_wind_1h,
            self.forecast_wind_3h,
        ]


# ---------------------------------------------------------------------------
# State Model — Extended episode-level tracking
# ---------------------------------------------------------------------------

class GridState(State):
    """
    Episode-level metadata — upgraded for multi-sector, scenario, and black-swan tracking.

    Inherits `episode_id` and `step_count` from State base.
    """

    # Scenario tracking
    scenario_id: int = Field(
        default=0, ge=0, le=4,
        description=(
            "Active scenario: 0=Normal, 1=Heatwave, 2=Storm, 3=Surge, 4=Fault. "
            "Set at reset(); never exposed in the observation vector."
        ),
    )

    # Cumulative rewards / costs
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of shaped rewards since last reset.",
    )
    total_cost: float = Field(
        default=0.0, ge=0.0,
        description="Total cost incurred from grid imports in this episode.",
    )

    # Hospital tracking
    hospital_failure_steps: int = Field(
        default=0, ge=0,
        description="Total steps where hospital served ratio < 0.95.",
    )
    consecutive_hospital_failures: int = Field(
        default=0, ge=0,
        description="Current streak of consecutive hospital under-supply steps.",
    )
    hospital_terminal_triggered: bool = Field(
        default=False,
        description="True if episode ended due to 3 consecutive hospital failures (V1 terminal).",
    )

    # Shed budget tracking
    total_energy_shed_kwh: float = Field(
        default=0.0, ge=0.0,
        description="Cumulative kWh shed across all sectors this episode.",
    )
    total_energy_demanded_kwh: float = Field(
        default=0.0, ge=0.0,
        description="Cumulative kWh demanded across all sectors this episode.",
    )

    # Renewable tracking
    renewable_energy_used: float = Field(
        default=0.0, ge=0.0,
        description="Total kWh served from renewables (solar + wind) in this episode.",
    )

    # Frequency
    frequency_violation_steps: int = Field(
        default=0, ge=0,
        description="Steps where |freq − 50| > 0.5 Hz (V2 hard zone).",
    )

    # BESS health
    bess_health: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Running BESS cycle-life health proxy [0, 1].",
    )

    # Black-swan event tracking
    solar_swan_triggered: bool = Field(
        default=False,
        description="True if solar-collapse event has occurred this episode.",
    )
    wind_swan_triggered: bool = Field(
        default=False,
        description="True if wind-failure event has occurred this episode.",
    )
    wind_swan_steps_remaining: int = Field(
        default=0, ge=0,
        description="Steps remaining for wind-failure black-swan.",
    )

    # Verifier logs (logged separately per spec)
    v1_triggers: int = Field(
        default=0, ge=0,
        description="Count of V1 (hospital) penalty activations this episode.",
    )
    v2_triggers: int = Field(
        default=0, ge=0,
        description="Count of V2 (frequency > 0.5Hz) hard-zone violations this episode.",
    )
    v3_triggers: int = Field(
        default=0, ge=0,
        description="Count of V3 (BESS SoC out-of-bounds) penalty activations.",
    )
    v4_triggers: int = Field(
        default=0, ge=0,
        description="Count of V4 (shed budget exceeded) penalty activations.",
    )

    # Legacy / grader compatibility
    blackout_count: int = Field(
        default=0, ge=0,
        description="Steps where total supply < 95% of total demand.",
    )
    total_steps: int = Field(
        default=0, ge=0,
        description="Total steps completed in this episode.",
    )
    demand_spike_hours: Optional[list] = Field(
        default=None,
        description="Surge scenario: hours with demand spike active.",
    )

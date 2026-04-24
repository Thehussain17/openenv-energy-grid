"""
Data models for the Energy Grid Environment — v2 (Multi-Sector Upgrade).

Action space  : 4-dimensional MultiDiscreteAction vector
Observation   : 18-element normalized float vector [0, 1]
State         : Extended GridState with per-sector and black-swan tracking

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
                     0=100%  1=98% (emergency safety margin)
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
        description="Hospital supply level: 0=100%, 1=98% (emergency margin).",
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
# Observation Model — 18-element normalized vector + rich raw fields
# ---------------------------------------------------------------------------

class GridObservation(Observation):
    """
    Normalized observation vector for the upgraded energy-grid environment.

    All scalar fields are in [0, 1] unless noted.
    Inherits `done`, `reward`, `metadata` from Observation base.
    """

    # ── Time encoding ──────────────────────────────────────────────────────
    time_sin: float = Field(default=0.0, description="sin(2πt/24) ∈ [-1,1].")
    time_cos: float = Field(default=1.0, description="cos(2πt/24) ∈ [-1,1].")

    # ── Generation ─────────────────────────────────────────────────────────
    solar_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="P_solar / 200 kW.",
    )
    wind_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="P_wind / 150 kW.",
    )

    # ── BESS ───────────────────────────────────────────────────────────────
    battery_soc: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="State of charge [0, 1].",
    )
    bess_health: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="BESS cycle-life health proxy [0, 1].",
    )

    # ── Hospital sector ────────────────────────────────────────────────────
    hosp_demand_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Hospital demand / 500 kW.",
    )
    hosp_served_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of hospital demand actually served.",
    )

    # ── Industrial sector ──────────────────────────────────────────────────
    ind_demand_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Industrial demand / 500 kW.",
    )
    ind_served_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of industrial demand actually served.",
    )

    # ── Residential sector ─────────────────────────────────────────────────
    res_demand_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Residential demand / 500 kW.",
    )
    res_served_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of residential demand actually served.",
    )

    # ── Grid physics ───────────────────────────────────────────────────────
    frequency_norm: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="(frequency − 49.0) / 2.0; maps 49–51 Hz → 0–1.",
    )
    grid_import_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Grid import / 500 kW.",
    )

    # ── Black-swan flags ───────────────────────────────────────────────────
    solar_swan_active: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1.0 if solar-collapse black-swan is active, else 0.0.",
    )
    wind_swan_active: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1.0 if wind-failure black-swan is active, else 0.0.",
    )

    # ── Episode progress & budget ──────────────────────────────────────────
    step_norm: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="step / 24.",
    )
    cumulative_shed_ratio_norm: float = Field(
        default=0.0,
        description="(cumulative_shed_ratio / 0.20); >1.0 means over budget.",
    )

    def to_vector(self) -> List[float]:
        """Return the canonical 18-element observation vector."""
        return [
            self.time_sin, self.time_cos,
            self.solar_norm, self.wind_norm,
            self.battery_soc, self.bess_health,
            self.hosp_demand_norm, self.hosp_served_ratio,
            self.ind_demand_norm, self.ind_served_ratio,
            self.res_demand_norm, self.res_served_ratio,
            self.frequency_norm, self.grid_import_norm,
            self.solar_swan_active, self.wind_swan_active,
            self.step_norm,
            self.cumulative_shed_ratio_norm,
        ]


# ---------------------------------------------------------------------------
# State Model — Extended episode-level tracking
# ---------------------------------------------------------------------------

class GridState(State):
    """
    Episode-level metadata — upgraded for multi-sector, black-swan tracking.

    Inherits `episode_id` and `step_count` from State base.
    """

    # Cumulative rewards / costs
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of shaped rewards since last reset.",
    )
    total_cost: float = Field(
        default=0.0, ge=0.0,
        description="Total ₹ spent on grid imports in this episode.",
    )

    # Hospital tracking
    hospital_failure_steps: int = Field(
        default=0, ge=0,
        description="Steps where hospital served ratio < 0.95.",
    )
    consecutive_hospital_failures: int = Field(
        default=0, ge=0,
        description="Current streak of consecutive hospital under-supply steps.",
    )
    hospital_terminal_triggered: bool = Field(
        default=False,
        description="True if episode ended due to 3 consecutive hospital failures.",
    )

    # Shed budget tracking
    total_energy_shed_kwh: float = Field(
        default=0.0, ge=0.0,
        description="Cumulative kWh shed across all sectors.",
    )
    total_energy_demanded_kwh: float = Field(
        default=0.0, ge=0.0,
        description="Cumulative kWh demanded across all sectors.",
    )

    # Renewable tracking
    renewable_energy_used: float = Field(
        default=0.0, ge=0.0,
        description="Total kWh served from renewables (solar + wind) in this episode.",
    )

    # Frequency
    frequency_violation_steps: int = Field(
        default=0, ge=0,
        description="Steps where |freq − 50| > 0.5 Hz.",
    )

    # BESS health
    bess_health: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Running BESS cycle-life health proxy [0,1].",
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
        description="Legacy: hours with amplified demand (unused in v2).",
    )

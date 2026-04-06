"""
Data models for the Energy Grid Environment.

Follows the OpenEnv type system:
  - GridAction extends Action (which already has `metadata`)
  - GridObservation extends Observation (which already has `done`, `reward`, `metadata`)
  - GridState extends State (which already has `episode_id`, `step_count`)
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class GridAction(Action):
    """
    Dispatch decision for the energy grid operator.

    The agent selects one of five operations and a magnitude
    (fraction of maximum capacity) to apply.
    """

    decision: Literal[
        "battery_discharge",
        "battery_charge",
        "buy_external",
        "curtail_load",
        "idle",
    ] = Field(
        ...,
        description=(
            "Dispatch operation to perform: "
            "'battery_discharge' draw stored energy, "
            "'battery_charge' store excess energy, "
            "'buy_external' purchase from external grid, "
            "'curtail_load' reduce consumer demand, "
            "'idle' take no action."
        ),
    )

    magnitude: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of maximum capacity to apply, in [0.0, 1.0].",
    )


class GridObservation(Observation):
    """
    Current grid state observed by the dispatch agent.

    All physical units are noted in field descriptions.
    Inherits `done`, `reward`, and `metadata` from Observation base.
    """

    solar_output: float = Field(
        default=0.0,
        description="Solar generation (kW). Follows a sine curve peaking at midday.",
    )
    wind_output: float = Field(
        default=0.0,
        description="Wind generation (kW). Gaussian noise around a mean value.",
    )
    demand: float = Field(
        default=0.0,
        description="Consumer electricity demand (kW). Peaks in morning and evening.",
    )
    battery_soc: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Battery state of charge, dimensionless in [0.0, 1.0].",
    )
    grid_frequency: float = Field(
        default=50.0,
        description="Grid frequency (Hz). Nominal 50 Hz; deviates under imbalance.",
    )
    time_of_day: int = Field(
        default=0,
        ge=0,
        le=23,
        description="Hour of day in 24-hour format (0–23).",
    )
    electricity_price: float = Field(
        default=5.0,
        description="Electricity spot price (₹/kWh). Higher during peak hours.",
    )
    renewable_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of supply currently met by renewables (solar + wind).",
    )


class GridState(State):
    """
    Episode-level metadata for the energy grid environment.

    Inherits `episode_id` and `step_count` from State base.
    Additional tracking fields are added for grader access.
    """

    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of shaped rewards accumulated since last reset.",
    )
    blackout_count: int = Field(
        default=0,
        ge=0,
        description="Number of timesteps where consumer demand was unmet (blackout).",
    )
    total_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Total ₹ spent on external grid purchases in this episode.",
    )
    renewable_energy_used: float = Field(
        default=0.0,
        ge=0.0,
        description="Total kWh served from renewables (solar + wind) in this episode.",
    )
    frequency_violation_steps: int = Field(
        default=0,
        ge=0,
        description="Number of steps where grid frequency deviated > 0.5 Hz from 50 Hz.",
    )
    total_steps: int = Field(
        default=0,
        ge=0,
        description="Total steps completed in this episode.",
    )
    demand_spike_hours: Optional[list] = Field(
        default=None,
        description="Hours with artificially amplified demand (used in Task 2).",
    )

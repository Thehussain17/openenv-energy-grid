"""
Energy Grid Environment — Core Server Implementation.

Simulates a 24-hour grid dispatch cycle in hourly timesteps.
The operator controls solar, wind, and battery storage to balance
variable consumer demand, minimising cost and avoiding blackouts.

Physical constants (representative values for a small microgrid):
  - Max solar capacity : 500 kW
  - Max wind capacity  : 300 kW
  - Battery capacity   : 1000 kWh  (SoC × 1000 → kWh stored)
  - Max battery power  : 200 kW    (charge / discharge rate)
  - Max external buy   : 400 kW
  - Demand base        : 250 kW
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import GridAction, GridObservation, GridState
except ImportError:
    import sys as _sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from models import GridAction, GridObservation, GridState

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MAX_SOLAR_KW: float = 500.0
MAX_WIND_KW: float = 300.0
BATTERY_CAPACITY_KWH: float = 1000.0   # total storage
MAX_BATTERY_POWER_KW: float = 200.0    # charge/discharge rate cap
MAX_BUY_KW: float = 400.0
BASE_DEMAND_KW: float = 250.0
BATTERY_EFFICIENCY: float = 0.90       # round-trip per step

# Time-of-use pricing schedule (₹/kWh) — peak hours cost more
_TOU_PRICE: dict[int, float] = {
    0: 3.0,  1: 3.0,  2: 3.0,  3: 3.0,  4: 3.0,  5: 3.5,
    6: 4.5,  7: 7.0,  8: 8.5,  9: 7.0, 10: 5.5, 11: 5.0,
    12: 4.5, 13: 4.5, 14: 5.0, 15: 5.5, 16: 6.5, 17: 8.0,
    18: 9.0, 19: 9.0, 20: 8.0, 21: 7.0, 22: 5.5, 23: 4.0,
}

# Demand multiplier for Task 2 spike hours (applied on top of base profile)
SPIKE_MULTIPLIER: float = 1.8


class EnergyGridEnvironment(Environment[GridAction, GridObservation, GridState]):
    """
    OpenEnv-compliant energy grid dispatch environment.

    Each episode simulates one 24-hour cycle (step 0 = hour 0, …, step 23 = hour 23).
    The episode terminates after step 23 (24 hourly decisions).

    Supports concurrent sessions via per-instance state.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state: GridState = GridState(episode_id=str(uuid4()), step_count=0)
        self._battery_soc: float = 0.5
        self._rng: random.Random = random.Random()
        self._demand_spike_hours: list[int] = []

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        demand_spike_hours: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> GridObservation:
        """
        Reset environment to a new episode.

        Args:
            seed: RNG seed for reproducible episodes.
            episode_id: Custom episode identifier.
            demand_spike_hours: Hours with amplified demand (Task 2).
        """
        if seed is not None:
            self._rng.seed(seed)
            random.seed(seed)

        self._battery_soc = self._rng.uniform(0.3, 0.7)
        self._demand_spike_hours = demand_spike_hours or []

        self._state = GridState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            cumulative_reward=0.0,
            blackout_count=0,
            total_cost=0.0,
            renewable_energy_used=0.0,
            frequency_violation_steps=0,
            total_steps=0,
            demand_spike_hours=self._demand_spike_hours,
        )

        return self._observe(hour=0, reward=0.0, done=False)

    def step(
        self,
        action: GridAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GridObservation:
        """
        Advance the environment by one hour.

        Args:
            action: Dispatch decision + magnitude.

        Returns:
            Updated GridObservation with shaped reward.
        """
        hour: int = self._state.step_count % 24

        # --- Generate renewable supply ---
        solar = self._solar(hour)
        wind = self._wind()
        renewable_supply = solar + wind

        # --- Consumer demand ---
        demand = self._demand(hour)

        # --- Apply the agent's action ---
        bought_kw: float = 0.0
        new_soc: float = self._battery_soc

        decision = action.decision
        mag = float(action.magnitude)

        if decision == "battery_discharge":
            discharge_kw = mag * MAX_BATTERY_POWER_KW
            available_kwh = new_soc * BATTERY_CAPACITY_KWH
            actual_discharge_kw = min(discharge_kw, available_kwh)  # 1-hr step → kW ≈ kWh
            new_soc -= actual_discharge_kw / BATTERY_CAPACITY_KWH
            renewable_supply += actual_discharge_kw  # treat stored renewables as clean

        elif decision == "battery_charge":
            charge_kw = mag * MAX_BATTERY_POWER_KW
            available_space = (1.0 - new_soc) * BATTERY_CAPACITY_KWH
            actual_charge_kw = min(charge_kw, available_space, renewable_supply)
            new_soc += (actual_charge_kw * BATTERY_EFFICIENCY) / BATTERY_CAPACITY_KWH
            renewable_supply -= actual_charge_kw  # energy diverted to battery

        elif decision == "buy_external":
            bought_kw = mag * MAX_BUY_KW

        elif decision == "curtail_load":
            demand *= (1.0 - mag * 0.5)  # curtail up to 50 % of demand

        # elif decision == "idle": pass — no change

        # Clamp SoC to valid range
        self._battery_soc = max(0.0, min(1.0, new_soc))

        # --- Energy balance ---
        total_supply = renewable_supply + bought_kw
        blackout = total_supply < demand * 0.95  # 5 % tolerance

        # --- Grid frequency deviation (proportional to imbalance) ---
        imbalance_fraction = (total_supply - demand) / max(demand, 1.0)
        freq_deviation = imbalance_fraction * 2.0  # ±2 Hz at full imbalance
        grid_frequency = 50.0 + freq_deviation

        # --- Renewable fraction ---
        if total_supply > 0:
            ren_fraction = min(renewable_supply, total_supply) / total_supply
        else:
            ren_fraction = 0.0

        # --- Shaped reward ---
        price = _TOU_PRICE[hour]
        reward = self._compute_reward(
            price=price,
            bought_kw=bought_kw,
            blackout=blackout,
            ren_fraction=ren_fraction,
            soc=self._battery_soc,
            grid_freq=grid_frequency,
        )

        # --- Update episode state ---
        self._state.step_count += 1
        self._state.total_steps += 1
        self._state.cumulative_reward += reward
        self._state.total_cost += price * bought_kw * 1e-3  # kWh × ₹/kWh
        self._state.renewable_energy_used += min(renewable_supply, demand)
        if blackout:
            self._state.blackout_count += 1
        if abs(grid_frequency - 50.0) > 0.5:
            self._state.frequency_violation_steps += 1

        done = self._state.step_count >= 24

        return self._observe(
            hour=hour,
            reward=reward,
            done=done,
            solar=solar,
            wind=wind,
            demand=demand,
            grid_frequency=grid_frequency,
            ren_fraction=ren_fraction,
            price=price,
        )

    @property
    def state(self) -> GridState:
        """Return current episode metadata."""
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="energy_grid_env",
            description=(
                "24-hour energy grid dispatch simulation. "
                "The agent balances solar, wind, battery storage, and external "
                "grid purchases against variable consumer demand, minimising cost "
                "and preventing blackouts."
            ),
            version="1.0.0",
            author="OpenEnv Community",
        )

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _solar(self, hour: int) -> float:
        """Solar output (kW): sine curve peaking at midday + noise."""
        angle = math.pi * hour / 24.0
        base = MAX_SOLAR_KW * (math.sin(angle) ** 2)
        noise = self._rng.gauss(0.0, 15.0)
        return max(0.0, base + noise)

    def _wind(self) -> float:
        """Wind output (kW): mean + uncorrelated Gaussian noise."""
        mean_wind = 120.0
        noise = self._rng.gauss(0.0, 40.0)
        return max(0.0, mean_wind + noise)

    def _demand(self, hour: int) -> float:
        """
        Consumer demand (kW): realistic load profile with morning and
        evening peaks, plus optional demand spikes for Task 2.
        """
        # Morning peak (8h) and evening peak (19h) modelled as Gaussians
        morning = 180.0 * math.exp(-0.5 * ((hour - 8) / 2.0) ** 2)
        evening = 220.0 * math.exp(-0.5 * ((hour - 19) / 2.5) ** 2)
        base = BASE_DEMAND_KW + morning + evening
        noise = self._rng.gauss(0.0, 10.0)
        demand = max(50.0, base + noise)

        # Task 2: artificially amplified spike hours
        if hour in self._demand_spike_hours:
            demand *= SPIKE_MULTIPLIER

        return demand

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reward(
        price: float,
        bought_kw: float,
        blackout: bool,
        ren_fraction: float,
        soc: float,
        grid_freq: float,
    ) -> float:
        """
        Shaped reward at every step.

        Components:
          - Cost penalty     : -price × bought_kWh
          - Blackout penalty : -50 per blackout step
          - Renewable bonus  : +2 × renewable_fraction
          - Battery health   : -0.1 × |soc - 0.5|
          - Freq stability   : -5 if |freq - 50| > 0.5 Hz
        """
        # Units: price ₹/kWh, bought in kW over 1-hr step → kWh = kW
        cost_penalty = -price * bought_kw * 1e-3
        blackout_penalty = -50.0 if blackout else 0.0
        renewable_bonus = 2.0 * ren_fraction
        battery_penalty = -0.1 * abs(soc - 0.5)
        freq_penalty = -5.0 if abs(grid_freq - 50.0) > 0.5 else 0.0

        return cost_penalty + blackout_penalty + renewable_bonus + battery_penalty + freq_penalty

    # ------------------------------------------------------------------
    # Helper: build observation
    # ------------------------------------------------------------------

    def _observe(
        self,
        hour: int,
        reward: float,
        done: bool,
        solar: Optional[float] = None,
        wind: Optional[float] = None,
        demand: Optional[float] = None,
        grid_frequency: float = 50.0,
        ren_fraction: float = 0.0,
        price: Optional[float] = None,
    ) -> GridObservation:
        if solar is None:
            solar = self._solar(hour)
        if wind is None:
            wind = self._wind()
        if demand is None:
            demand = self._demand(hour)
        if price is None:
            price = _TOU_PRICE[hour]

        return GridObservation(
            solar_output=round(solar, 2),
            wind_output=round(wind, 2),
            demand=round(demand, 2),
            battery_soc=round(self._battery_soc, 4),
            grid_frequency=round(grid_frequency, 4),
            time_of_day=hour,
            electricity_price=price,
            renewable_fraction=round(ren_fraction, 4),
            reward=reward,
            done=done,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "blackout_count": self._state.blackout_count,
                "total_cost": round(self._state.total_cost, 4),
            },
        )

"""
Energy Grid Environment v2 — Multi-Sector Physics Engine.

Upgrades over v1:
  * Three-sector load: Hospital / Industrial / Residential
  * New solar/wind equations + Black Swan events (solar collapse, wind failure)
  * Virtual-inertia frequency physics (df = ΔP / M, M=1000)
  * Multi-Discrete 4-dim action space
  * 18-element normalized observation vector
  * Layered reward function with hard hospital clamp and black-swan multiplier
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        GridAction, GridObservation, GridState,
        BESS_MAP, HOSPITAL_RATIOS, INDUSTRIAL_RATIOS, RESIDENTIAL_RATIOS,
    )
except ImportError:
    import sys as _sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from models import (
        GridAction, GridObservation, GridState,
        BESS_MAP, HOSPITAL_RATIOS, INDUSTRIAL_RATIOS, RESIDENTIAL_RATIOS,
    )

# ---------------------------------------------------------------------------
# Physical constants — 500 kW microgrid
# ---------------------------------------------------------------------------
MAX_SOLAR_KW        = 200.0    # Peak solar capacity
MAX_WIND_KW         = 150.0    # Peak wind capacity
BESS_CAPACITY_KWH   = 1000.0   # Total BESS storage
MAX_BESS_POWER_KW   = 200.0    # Max charge / discharge rate
MAX_IMPORT_KW       = 500.0    # Max grid import
BESS_EFFICIENCY     = 0.90     # Round-trip efficiency

# Sector base demands (kW)
HOSP_BASE_KW  = 100.0   # 20% of 500
IND_BASE_KW   = 250.0   # 50% of 500
RES_BASE_KW   = 150.0   # 30% of 500

# Frequency physics
VIRTUAL_INERTIA_M = 1000.0   # MW·s — virtual inertia constant
FREQ_NOM          = 50.0
FREQ_MIN          = 49.0
FREQ_MAX          = 51.0

# Black-swan probabilities (checked once per step after step 6)
P_SOLAR_COLLAPSE  = 0.02
P_WIND_FAILURE    = 0.02

# BESS health wear per kWh throughput (rough proxy, per kWh / BESS_CAPACITY_KWH)
BESS_WEAR_PER_KWH = 1.0 / (500.0 * BESS_CAPACITY_KWH)   # ~2000 full cycles lifetime


# ---------------------------------------------------------------------------
# Beta(5,2) sampler (no scipy dependency)
# ---------------------------------------------------------------------------
def _beta_sample(rng: random.Random, a: float = 5.0, b: float = 2.0) -> float:
    """Sample from Beta(a, b) using two Gamma variates."""
    x = rng.gammavariate(a, 1.0)
    y = rng.gammavariate(b, 1.0)
    return x / (x + y)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class EnergyGridEnvironment(Environment[GridAction, GridObservation, GridState]):
    """
    OpenEnv-compliant v2 energy grid dispatch environment.

    Episode: 24 hourly timesteps (step 0 = hour 0 … step 23 = hour 23).
    Terminates:
      - After 24 steps (normal)
      - Immediately when hospital supply < 0.95 for 3 consecutive steps
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__()
        self._rng: random.Random = random.Random()
        self._state: GridState = GridState(episode_id=str(uuid4()), step_count=0)
        self._battery_soc: float = 0.5
        self._grid_frequency: float = FREQ_NOM
        self._bess_health: float = 1.0
        # Black-swan state
        self._solar_swan_active: bool = False
        self._wind_swan_active: bool = False
        self._wind_swan_steps_left: int = 0
        # Shed budget accumulator
        self._cumulative_shed_kwh: float = 0.0
        self._cumulative_demand_kwh: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GridObservation:
        if seed is not None:
            self._rng.seed(seed)
            random.seed(seed)

        self._battery_soc    = self._rng.uniform(0.3, 0.7)
        self._grid_frequency = FREQ_NOM
        self._bess_health    = 1.0

        self._solar_swan_active   = False
        self._wind_swan_active    = False
        self._wind_swan_steps_left = 0
        self._cumulative_shed_kwh  = 0.0
        self._cumulative_demand_kwh = 0.0

        self._state = GridState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            cumulative_reward=0.0,
            total_cost=0.0,
            hospital_failure_steps=0,
            consecutive_hospital_failures=0,
            hospital_terminal_triggered=False,
            total_energy_shed_kwh=0.0,
            total_energy_demanded_kwh=0.0,
            renewable_energy_used=0.0,
            frequency_violation_steps=0,
            bess_health=1.0,
            solar_swan_triggered=False,
            wind_swan_triggered=False,
            wind_swan_steps_remaining=0,
            blackout_count=0,
            total_steps=0,
        )

        return self._build_obs(hour=0, reward=0.0, done=False)

    def step(
        self,
        action: GridAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GridObservation:
        hour: int = self._state.step_count % 24

        # ── 1. Generate renewable supply ───────────────────────────────
        solar = self._solar(hour)
        wind  = self._wind(hour)

        # ── 2. Black-swan events (after step 6) ───────────────────────
        any_swan = self._tick_black_swans(solar, wind)

        # Re-evaluate after swan mutations
        solar = 0.0 if self._solar_swan_active else solar
        if self._wind_swan_active:
            wind = wind * 0.10

        p_gen = solar + wind

        # ── 3. Sector demands ─────────────────────────────────────────
        hosp_demand = self._hospital_demand()
        ind_demand  = self._industrial_demand(hour)
        res_demand  = self._residential_demand(hour)
        total_demand = hosp_demand + ind_demand + res_demand

        # ── 4. Decode action ──────────────────────────────────────────
        hosp_ratio  = HOSPITAL_RATIOS[action.hospital]
        ind_ratio   = INDUSTRIAL_RATIOS[action.industrial]
        res_ratio   = RESIDENTIAL_RATIOS[action.residential]

        hosp_served  = hosp_demand  * hosp_ratio
        ind_served   = ind_demand   * ind_ratio
        res_served   = res_demand   * res_ratio
        total_served = hosp_served + ind_served + res_served

        # BESS dispatch
        bess_mode, bess_frac = BESS_MAP[action.bess]
        bess_power_kw = 0.0   # positive = discharge to grid
        wear_increment = 0.0

        if bess_mode is True:   # charging
            charge_kw = bess_frac * MAX_BESS_POWER_KW
            available_space = (1.0 - self._battery_soc) * BESS_CAPACITY_KWH
            actual_charge = min(charge_kw, available_space, p_gen)
            self._battery_soc += (actual_charge * BESS_EFFICIENCY) / BESS_CAPACITY_KWH
            p_gen -= actual_charge
            wear_increment = actual_charge / BESS_CAPACITY_KWH

        elif bess_mode is False:  # discharging
            discharge_kw = bess_frac * MAX_BESS_POWER_KW
            available_kwh = self._battery_soc * BESS_CAPACITY_KWH
            actual_discharge = min(discharge_kw, available_kwh)
            self._battery_soc -= actual_discharge / BESS_CAPACITY_KWH
            bess_power_kw = actual_discharge
            wear_increment = actual_discharge / BESS_CAPACITY_KWH

        self._battery_soc = max(0.0, min(1.0, self._battery_soc))

        # BESS health degradation
        self._bess_health = max(0.0, self._bess_health - BESS_WEAR_PER_KWH * wear_increment * BESS_CAPACITY_KWH)
        self._state.bess_health = self._bess_health

        # ── 5. Grid import to cover deficit ───────────────────────────
        available_supply = p_gen + bess_power_kw
        deficit = max(0.0, total_served - available_supply)
        p_import = min(deficit, MAX_IMPORT_KW)

        # Actual supply after import
        total_supply = available_supply + p_import

        # ── 6. Frequency physics (virtual-inertia model) ───────────────
        p_export = max(0.0, total_supply - total_served)
        df = (p_gen + p_import - p_export - total_served) / VIRTUAL_INERTIA_M
        self._grid_frequency = max(FREQ_MIN, min(FREQ_MAX, self._grid_frequency + df))

        # ── 7. Renewable fraction ─────────────────────────────────────
        renew_supplied = min(p_gen + bess_power_kw, total_supply)
        renew_frac = renew_supplied / max(total_supply, 1.0)

        # ── 8. Per-sector actual serve ratios ─────────────────────────
        # If there is a supply shortfall after import, we can't fully serve
        # all sectors; deduct from lowest-priority first
        actual_total = min(total_supply, total_served)
        shortfall = max(0.0, total_served - actual_total)

        # Distribute shortfall: residential absorbs first, then industrial
        res_actual  = max(0.0, res_served - min(shortfall, res_served))
        shortfall   = max(0.0, shortfall - (res_served - res_actual))
        ind_actual  = max(0.0, ind_served - min(shortfall, ind_served))
        shortfall   = max(0.0, shortfall - (ind_served - ind_actual))
        hosp_actual = max(0.0, hosp_served - min(shortfall, hosp_served))

        h_ratio = hosp_actual / max(hosp_demand, 1.0)
        i_ratio = ind_actual  / max(ind_demand,  1.0)
        r_ratio = res_actual  / max(res_demand,  1.0)

        # ── 9. Shed budget ────────────────────────────────────────────
        shed_this_step = (
            (hosp_demand  - hosp_actual)
            + (ind_demand  - ind_actual)
            + (res_demand  - res_actual)
        )
        self._cumulative_shed_kwh   += shed_this_step
        self._cumulative_demand_kwh += total_demand
        cum_shed_ratio = self._cumulative_shed_kwh / max(self._cumulative_demand_kwh, 1.0)

        # ── 10. Hospital consecutive-failure tracking ──────────────────
        if h_ratio < 0.95:
            self._state.consecutive_hospital_failures += 1
            self._state.hospital_failure_steps += 1
        else:
            self._state.consecutive_hospital_failures = 0

        hospital_terminal = self._state.consecutive_hospital_failures >= 3

        # ── 11. Reward ────────────────────────────────────────────────
        reward, is_terminal = self._compute_reward(
            h_ratio=h_ratio,
            i_ratio=i_ratio,
            r_ratio=r_ratio,
            freq=self._grid_frequency,
            soc=self._battery_soc,
            cum_shed_ratio=cum_shed_ratio,
            renew_frac=renew_frac,
            wear_inc=wear_increment,
            any_swan=any_swan,
            hospital_terminal=hospital_terminal,
        )

        # ── 12. Update episode state ───────────────────────────────────
        self._state.step_count              += 1
        self._state.total_steps             += 1
        self._state.cumulative_reward       += reward
        self._state.total_cost              += p_import * 0.005   # ₹5/kWh × kW × 1e-3
        self._state.total_energy_shed_kwh   = self._cumulative_shed_kwh
        self._state.total_energy_demanded_kwh = self._cumulative_demand_kwh
        self._state.renewable_energy_used   += renew_supplied
        self._state.solar_swan_triggered    = self._state.solar_swan_triggered or self._solar_swan_active
        self._state.wind_swan_triggered     = self._state.wind_swan_triggered or self._wind_swan_active
        self._state.wind_swan_steps_remaining = self._wind_swan_steps_left

        if abs(self._grid_frequency - FREQ_NOM) > 0.5:
            self._state.frequency_violation_steps += 1
        if total_supply < total_demand * 0.95:
            self._state.blackout_count += 1

        done = hospital_terminal or is_terminal or self._state.step_count >= 24
        if hospital_terminal:
            self._state.hospital_terminal_triggered = True

        return self._build_obs(
            hour=hour,
            reward=reward,
            done=done,
            solar=solar, wind=wind,
            hosp_demand=hosp_demand, hosp_ratio=h_ratio,
            ind_demand=ind_demand,   ind_ratio=i_ratio,
            res_demand=res_demand,   res_ratio=r_ratio,
            p_import=p_import,
            cum_shed_ratio=cum_shed_ratio,
        )

    @property
    def state(self) -> GridState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="energy_grid_env",
            description=(
                "v2: 24-hour 500kW microgrid dispatch with three-sector load "
                "(Hospital/Industrial/Residential), black-swan events, "
                "virtual-inertia frequency physics, and multi-discrete actions."
            ),
            version="2.0.0",
            author="OpenEnv Community",
        )

    # ------------------------------------------------------------------
    # Generation models
    # ------------------------------------------------------------------

    def _solar(self, hour: int) -> float:
        """P_solar(t) = 200kW * max(0, sin(π*(t−6)/12)) * Beta(5,2)."""
        raw = math.sin(math.pi * (hour - 6) / 12.0)
        if raw <= 0:
            return 0.0
        beta_sample = _beta_sample(self._rng, 5.0, 2.0)
        return MAX_SOLAR_KW * raw * beta_sample

    def _wind(self, hour: int) -> float:
        """P_wind(t) = 150kW * clip(N(μ(t), 0.15), 0, 1); μ varies diurnally."""
        mu = 0.6 + 0.2 * math.sin(2 * math.pi * hour / 24.0 + math.pi)
        sample = self._rng.gauss(mu, 0.15)
        sample = max(0.0, min(1.0, sample))
        return MAX_WIND_KW * sample

    # ------------------------------------------------------------------
    # Sector demand models
    # ------------------------------------------------------------------

    def _hospital_demand(self) -> float:
        """Flat ~100 kW + small noise."""
        noise = self._rng.gauss(0.0, 3.0)
        return max(90.0, HOSP_BASE_KW + noise)

    def _industrial_demand(self, hour: int) -> float:
        """Active 06:00–22:00; flat ~250 kW with light noise."""
        if hour < 6 or hour >= 22:
            return 0.0
        noise = self._rng.gauss(0.0, 15.0)
        return max(0.0, IND_BASE_KW + noise)

    def _residential_demand(self, hour: int) -> float:
        """Bimodal peaks at 08:00 and 19:00, base ~150 kW."""
        morning = 80.0  * math.exp(-0.5 * ((hour - 8)  / 2.0)  ** 2)
        evening = 100.0 * math.exp(-0.5 * ((hour - 19) / 2.5)  ** 2)
        noise   = self._rng.gauss(0.0, 8.0)
        return max(20.0, RES_BASE_KW + morning + evening + noise)

    # ------------------------------------------------------------------
    # Black-swan event logic
    # ------------------------------------------------------------------

    def _tick_black_swans(self, solar_raw: float, wind_raw: float) -> bool:
        """
        Roll black-swan events after step 6.
        Returns True if any swan is currently active.
        """
        step = self._state.step_count

        # Solar collapse: permanent for remainder of episode
        if not self._solar_swan_active and step > 6:
            if self._rng.random() < P_SOLAR_COLLAPSE:
                self._solar_swan_active = True

        # Wind failure: lasts 3-6 steps
        if not self._wind_swan_active and step > 6:
            if self._rng.random() < P_WIND_FAILURE:
                self._wind_swan_active = True
                self._wind_swan_steps_left = self._rng.randint(3, 6)

        if self._wind_swan_active:
            self._wind_swan_steps_left -= 1
            if self._wind_swan_steps_left <= 0:
                self._wind_swan_active = False

        return self._solar_swan_active or self._wind_swan_active

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        h_ratio: float,
        i_ratio: float,
        r_ratio: float,
        freq: float,
        soc: float,
        cum_shed_ratio: float,
        renew_frac: float,
        wear_inc: float,
        any_swan: bool,
        hospital_terminal: bool,
    ) -> tuple[float, bool]:
        """
        Returns (reward, is_terminal).

        V1 — Hospital hard clamp (terminal)
        V2 — Frequency physics penalty
        V3 — BESS SoC boundary penalties
        V4 — Shed budget excess penalty
        Core — Weighted service ratios + renewable bonus − BESS wear
        Multiplier — 3× during any active black-swan
        """
        # V1: Hospital hard clamp
        if h_ratio < 0.95:
            if hospital_terminal:
                return -1000.0, True
            # Non-terminal under-supply step still penalised severely
            return -1000.0, False

        # V2: Frequency physics
        freq_dev = abs(freq - FREQ_NOM)
        if freq_dev > 0.5:
            v2 = -500.0 * freq_dev
        else:
            v2 = -100.0 * freq_dev

        # V3: BESS SoC boundaries
        if soc < 0.15:
            v3 = -200.0 * (0.15 - soc)
        elif soc > 0.95:
            v3 = -50.0  * (soc - 0.95)
        else:
            v3 = 0.0

        # V4: Shed budget
        excess_shed = max(0.0, cum_shed_ratio - 0.20)
        v4 = -100.0 * excess_shed

        # Core: weighted service + renewables − wear
        core = (
            (1.0 * h_ratio + 0.6 * i_ratio + 0.3 * r_ratio) * 100.0
            + 10.0  * renew_frac
            - 50.0  * wear_inc
        )

        reward = core + v2 + v3 + v4

        # Black-swan multiplier
        if any_swan:
            reward *= 3.0

        return reward, False

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        hour: int,
        reward: float,
        done: bool,
        solar: float = 0.0,
        wind: float = 0.0,
        hosp_demand: float = HOSP_BASE_KW,
        hosp_ratio: float = 1.0,
        ind_demand: float = IND_BASE_KW,
        ind_ratio: float = 1.0,
        res_demand: float = RES_BASE_KW,
        res_ratio: float = 1.0,
        p_import: float = 0.0,
        cum_shed_ratio: float = 0.0,
    ) -> GridObservation:
        t = hour
        freq = self._grid_frequency
        step = self._state.step_count  # already incremented

        return GridObservation(
            # Time encoding
            time_sin=round(math.sin(2 * math.pi * t / 24), 6),
            time_cos=round(math.cos(2 * math.pi * t / 24), 6),
            # Generation (normalized)
            solar_norm=round(min(1.0, solar / MAX_SOLAR_KW), 4),
            wind_norm=round(min(1.0, wind  / MAX_WIND_KW),  4),
            # BESS
            battery_soc=round(self._battery_soc, 4),
            bess_health=round(self._bess_health, 4),
            # Hospital
            hosp_demand_norm=round(hosp_demand / 500.0, 4),
            hosp_served_ratio=round(max(0.0, min(1.0, hosp_ratio)), 4),
            # Industrial
            ind_demand_norm=round(ind_demand / 500.0, 4),
            ind_served_ratio=round(max(0.0, min(1.0, ind_ratio)), 4),
            # Residential
            res_demand_norm=round(res_demand / 500.0, 4),
            res_served_ratio=round(max(0.0, min(1.0, res_ratio)), 4),
            # Grid physics
            frequency_norm=round((freq - FREQ_MIN) / (FREQ_MAX - FREQ_MIN), 4),
            grid_import_norm=round(min(1.0, p_import / 500.0), 4),
            # Black swans
            solar_swan_active=1.0 if self._solar_swan_active else 0.0,
            wind_swan_active=1.0  if self._wind_swan_active  else 0.0,
            # Episode progress
            step_norm=round(min(1.0, step / 24), 4),
            cumulative_shed_ratio_norm=round(cum_shed_ratio / 0.20, 4),
            # OpenEnv base fields
            reward=round(reward, 4),
            done=done,
            metadata={
                "episode_id":                  self._state.episode_id,
                "step":                        step,
                "grid_frequency_hz":           round(freq, 4),
                "solar_kw":                    round(solar, 2),
                "wind_kw":                     round(wind, 2),
                "hosp_demand_kw":              round(hosp_demand, 2),
                "ind_demand_kw":               round(ind_demand, 2),
                "res_demand_kw":               round(res_demand, 2),
                "p_import_kw":                 round(p_import, 2),
                "cum_shed_ratio":              round(cum_shed_ratio, 4),
                "solar_swan":                  self._solar_swan_active,
                "wind_swan":                   self._wind_swan_active,
                "hospital_consecutive_fail":   self._state.consecutive_hospital_failures,
                "blackout_count":              self._state.blackout_count,
                "total_cost":                  round(self._state.total_cost, 4),
            },
        )

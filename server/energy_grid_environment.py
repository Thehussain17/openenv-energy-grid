"""
Energy Grid Environment v2.1 — Multi-Sector Physics Engine with Scenarios.

Upgrades over v2.0:
  * 22-Dim Observation Vector with 1h/3h forecasts
  * 5 Curriculum Scenarios (Normal, Heatwave, Storm, Surge, Fault)
  * Refined Reward Function (V2 deadband, V3 critical zone, V4 friction)
  * TRL GRPOTrainer compatibility
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional, Dict, Tuple, List
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
MAX_SOLAR_KW        = 200.0
MAX_WIND_KW         = 150.0
MAX_IMPORT_KW       = 500.0

HOSP_BASE_KW  = 100.0
IND_BASE_KW   = 250.0
RES_BASE_KW   = 150.0

FREQ_NOM          = 50.0
FREQ_MIN          = 49.0
FREQ_MAX          = 51.0

P_SOLAR_COLLAPSE  = 0.02
P_WIND_FAILURE    = 0.02


def _beta_sample(rng: random.Random, a: float = 5.0, b: float = 2.0) -> float:
    x = rng.gammavariate(a, 1.0)
    y = rng.gammavariate(b, 1.0)
    return x / (x + y)


class EnergyGridEnvironment(Environment[GridAction, GridObservation, GridState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._rng: random.Random = random.Random()
        self._state: GridState = GridState(episode_id=str(uuid4()), step_count=0)
        
        # Grid/Storage State
        self._battery_soc: float = 0.5
        self._grid_frequency: float = FREQ_NOM
        self._bess_health: float = 1.0
        
        # Black-swan state
        self._solar_swan_active: bool = False
        self._wind_swan_active: bool = False
        self._wind_swan_steps_left: int = 0
        
        # Budget
        self._cumulative_shed_kwh: float = 0.0
        self._cumulative_demand_kwh: float = 0.0

        # Scenario Overrides (defaults)
        self._scenario: int = 0
        self._bess_capacity_kwh: float = 1000.0
        self._bess_max_power_kw: float = 200.0
        self._bess_efficiency: float = 0.90
        self._virtual_inertia_m: float = 1000.0
        self._surge_start: int = -1
        self._surge_end: int = -1

    def _setup_scenario(self, scenario: int) -> None:
        self._scenario = scenario
        
        # Defaults
        self._bess_capacity_kwh = 1000.0
        self._bess_max_power_kw = 200.0
        self._bess_efficiency = 0.90
        self._virtual_inertia_m = 1000.0
        self._surge_start = -1
        self._surge_end = -1
        self._bess_health = 1.0

        if scenario == 1: # Heatwave
            self._bess_efficiency = 0.82
        elif scenario == 2: # Storm
            self._virtual_inertia_m = 700.0
        elif scenario == 3: # Surge
            self._surge_start = self._rng.randint(6, 18)
            self._surge_end = self._surge_start + self._rng.randint(4, 6)
        elif scenario == 4: # Fault
            self._bess_capacity_kwh = 100.0
            self._bess_max_power_kw = 20.0
            self._bess_health = 0.30

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, scenario: int = 0, **kwargs: Any) -> GridObservation:
        if seed is not None:
            self._rng.seed(seed)
            random.seed(seed)

        self._setup_scenario(scenario)

        self._battery_soc = self._rng.uniform(0.3, 0.7)
        self._grid_frequency = FREQ_NOM

        self._solar_swan_active = False
        self._wind_swan_active = False
        self._wind_swan_steps_left = 0
        self._cumulative_shed_kwh = 0.0
        self._cumulative_demand_kwh = 0.0

        self._state = GridState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scenario_id=scenario,
            cumulative_reward=0.0,
            total_cost=0.0,
            hospital_failure_steps=0,
            consecutive_hospital_failures=0,
            hospital_terminal_triggered=False,
            total_energy_shed_kwh=0.0,
            total_energy_demanded_kwh=0.0,
            renewable_energy_used=0.0,
            frequency_violation_steps=0,
            bess_health=self._bess_health,
            solar_swan_triggered=False,
            wind_swan_triggered=False,
            wind_swan_steps_remaining=0,
            v1_triggers=0,
            v2_triggers=0,
            v3_triggers=0,
            v4_triggers=0,
            blackout_count=0,
            total_steps=0,
            demand_spike_hours=list(range(self._surge_start, self._surge_end)) if self._surge_start != -1 else None
        )

        return self._build_obs(hour=0, reward=0.0, done=False)

    def _solar(self, hour: int) -> float:
        raw = math.sin(math.pi * (hour - 6) / 12.0)
        if raw <= 0: return 0.0
        
        if self._scenario == 1: # Heatwave
            beta_sample = _beta_sample(self._rng, 3.0, 3.0)
            return min(140.0, MAX_SOLAR_KW * raw * beta_sample)
        elif self._scenario == 2: # Storm
            return MAX_SOLAR_KW * raw * 0.15 # Max 15%
            
        beta_sample = _beta_sample(self._rng, 5.0, 2.0)
        return MAX_SOLAR_KW * raw * beta_sample

    def _wind(self, hour: int) -> float:
        mu = 0.6 + 0.2 * math.sin(2 * math.pi * hour / 24.0 + math.pi)
        sigma = 0.15
        
        if self._scenario == 2: # Storm
            mu *= 1.6
            sigma = 0.25
            
        sample = max(0.0, min(1.0, self._rng.gauss(mu, sigma)))
        return MAX_WIND_KW * sample

    def _hospital_demand(self, hour: int) -> float:
        noise = self._rng.gauss(0.0, 3.0)
        base = HOSP_BASE_KW
        if self._scenario == 1: base *= 1.25 # Heatwave
        elif self._scenario == 2: base *= 1.10 # Storm
        elif self._scenario == 3 and self._surge_start <= hour < self._surge_end: base *= 1.40
        return max(90.0, base + noise)

    def _industrial_demand(self, hour: int) -> float:
        if hour < 6 or hour >= 22: return 0.0
        noise = self._rng.gauss(0.0, 15.0)
        base = IND_BASE_KW
        if self._scenario == 1: base *= 1.25
        elif self._scenario == 3 and self._surge_start <= hour < self._surge_end: base *= 1.40
        return max(0.0, base + noise)

    def _residential_demand(self, hour: int) -> float:
        morning = 80.0 * math.exp(-0.5 * ((hour - 8) / 2.0)**2)
        evening = 100.0 * math.exp(-0.5 * ((hour - 19) / 2.5)**2)
        noise = self._rng.gauss(0.0, 8.0)
        base = RES_BASE_KW
        if self._scenario == 1: base *= 1.25
        elif self._scenario == 2: base *= 1.10
        elif self._scenario == 3 and self._surge_start <= hour < self._surge_end: base *= 1.40
        return max(20.0, base + morning + evening + noise)

    def _get_forecast(self, horizon: int = 6) -> Tuple[List[float], List[float]]:
        # Returns (solar_forecasts, wind_forecasts) for next `horizon` hours
        current_hour = self._state.step_count % 24
        s_fc, w_fc = [], []
        for i in range(1, horizon + 1):
            target_hour = (current_hour + i) % 24
            
            # True values
            s_true = self._solar(target_hour)
            w_true = self._wind(target_hour)
            
            # Apply noise (increases with horizon)
            noise_factor = 0.10 + (0.01 * i)
            
            s_noisy = s_true * self._rng.gauss(1.0, noise_factor)
            w_noisy = w_true * self._rng.gauss(1.0, noise_factor)
            
            # If black swans are active, forecast assumes they persist if steps remain (for wind) or forever (solar)
            if self._solar_swan_active:
                s_noisy = 0.0
            if self._wind_swan_active and i <= self._wind_swan_steps_left:
                w_noisy = w_noisy * 0.10

            s_fc.append(max(0.0, min(MAX_SOLAR_KW, s_noisy)))
            w_fc.append(max(0.0, min(MAX_WIND_KW, w_noisy)))
            
        return s_fc, w_fc

    def _tick_black_swans(self, solar_raw: float, wind_raw: float) -> bool:
        step = self._state.step_count
        if not self._solar_swan_active and step > 6:
            if self._rng.random() < P_SOLAR_COLLAPSE:
                self._solar_swan_active = True

        if not self._wind_swan_active and step > 6:
            if self._rng.random() < P_WIND_FAILURE:
                self._wind_swan_active = True
                self._wind_swan_steps_left = self._rng.randint(3, 6)

        if self._wind_swan_active:
            self._wind_swan_steps_left -= 1
            if self._wind_swan_steps_left <= 0:
                self._wind_swan_active = False

        return self._solar_swan_active or self._wind_swan_active

    def step(self, action: GridAction, timeout_s: Optional[float] = None, **kwargs: Any) -> GridObservation:
        hour = self._state.step_count % 24
        
        solar = self._solar(hour)
        wind = self._wind(hour)
        any_swan = self._tick_black_swans(solar, wind)
        
        solar = 0.0 if self._solar_swan_active else solar
        if self._wind_swan_active: wind *= 0.10
            
        p_gen = solar + wind

        hosp_demand = self._hospital_demand(hour)
        ind_demand = self._industrial_demand(hour)
        res_demand = self._residential_demand(hour)
        total_demand = hosp_demand + ind_demand + res_demand

        hosp_served = hosp_demand * HOSPITAL_RATIOS[action.hospital]
        ind_served = ind_demand * INDUSTRIAL_RATIOS[action.industrial]
        res_served = res_demand * RESIDENTIAL_RATIOS[action.residential]
        total_served = hosp_served + ind_served + res_served

        bess_mode, bess_frac = BESS_MAP[action.bess]
        bess_power_kw = 0.0
        wear_increment = 0.0

        if bess_mode is True:
            charge_kw = bess_frac * self._bess_max_power_kw
            available_space = (1.0 - self._battery_soc) * self._bess_capacity_kwh
            actual_charge = min(charge_kw, available_space, p_gen)
            self._battery_soc += (actual_charge * self._bess_efficiency) / self._bess_capacity_kwh
            p_gen -= actual_charge
            wear_increment = actual_charge / self._bess_capacity_kwh
        elif bess_mode is False:
            discharge_kw = bess_frac * self._bess_max_power_kw
            available_kwh = self._battery_soc * self._bess_capacity_kwh
            actual_discharge = min(discharge_kw, available_kwh)
            self._battery_soc -= actual_discharge / self._bess_capacity_kwh
            bess_power_kw = actual_discharge
            wear_increment = actual_discharge / self._bess_capacity_kwh

        self._battery_soc = max(0.0, min(1.0, self._battery_soc))
        
        bess_wear_per_kwh = 1.0 / (500.0 * self._bess_capacity_kwh)
        self._bess_health = max(0.0, self._bess_health - bess_wear_per_kwh * wear_increment * self._bess_capacity_kwh)
        self._state.bess_health = self._bess_health

        available_supply = p_gen + bess_power_kw
        deficit = max(0.0, total_served - available_supply)
        p_import = min(deficit, MAX_IMPORT_KW)
        total_supply = available_supply + p_import

        p_export = max(0.0, total_supply - total_served)
        df = (p_gen + p_import - p_export - total_served) / self._virtual_inertia_m
        self._grid_frequency = max(FREQ_MIN, min(FREQ_MAX, self._grid_frequency + df))

        renew_supplied = min(p_gen + bess_power_kw, total_supply)
        renew_frac = renew_supplied / max(total_supply, 1.0)

        actual_total = min(total_supply, total_served)
        shortfall = max(0.0, total_served - actual_total)

        res_actual = max(0.0, res_served - min(shortfall, res_served))
        shortfall = max(0.0, shortfall - (res_served - res_actual))
        ind_actual = max(0.0, ind_served - min(shortfall, ind_served))
        shortfall = max(0.0, shortfall - (ind_served - ind_actual))
        hosp_actual = max(0.0, hosp_served - min(shortfall, hosp_served))

        h_ratio = hosp_actual / max(hosp_demand, 1.0)
        i_ratio = ind_actual / max(ind_demand, 1.0)
        r_ratio = res_actual / max(res_demand, 1.0)

        shed_this_step = (hosp_demand - hosp_actual) + (ind_demand - ind_actual) + (res_demand - res_actual)
        self._cumulative_shed_kwh += shed_this_step
        self._cumulative_demand_kwh += total_demand
        cum_shed_ratio = self._cumulative_shed_kwh / max(self._cumulative_demand_kwh, 1.0)

        if h_ratio < 0.95:
            self._state.consecutive_hospital_failures += 1
            self._state.hospital_failure_steps += 1
        else:
            self._state.consecutive_hospital_failures = 0

        hospital_terminal = self._state.consecutive_hospital_failures >= 3

        reward, is_terminal = self._compute_reward(
            h_ratio, i_ratio, r_ratio, self._grid_frequency, self._battery_soc,
            cum_shed_ratio, shed_this_step, total_demand, renew_frac, wear_increment,
            any_swan, hospital_terminal
        )

        self._state.step_count += 1
        self._state.total_steps += 1
        self._state.cumulative_reward += reward
        self._state.total_cost += p_import * 0.005
        self._state.total_energy_shed_kwh = self._cumulative_shed_kwh
        self._state.total_energy_demanded_kwh = self._cumulative_demand_kwh
        self._state.renewable_energy_used += renew_supplied
        self._state.solar_swan_triggered = self._state.solar_swan_triggered or self._solar_swan_active
        self._state.wind_swan_triggered = self._state.wind_swan_triggered or self._wind_swan_active
        self._state.wind_swan_steps_remaining = self._wind_swan_steps_left

        if abs(self._grid_frequency - FREQ_NOM) > 0.5:
            self._state.frequency_violation_steps += 1
        if total_supply < total_demand * 0.95:
            self._state.blackout_count += 1

        done = hospital_terminal or is_terminal or self._state.step_count >= 24
        if hospital_terminal:
            self._state.hospital_terminal_triggered = True

        return self._build_obs(
            hour=hour, reward=reward, done=done,
            solar=solar, wind=wind,
            hosp_demand=hosp_demand, hosp_ratio=h_ratio,
            ind_demand=ind_demand, ind_ratio=i_ratio,
            res_demand=res_demand, res_ratio=r_ratio,
            p_import=p_import, cum_shed_ratio=cum_shed_ratio
        )

    def _compute_reward(self, h_ratio, i_ratio, r_ratio, freq, soc, cum_shed_ratio, shed_this_step, total_demand, renew_frac, wear_inc, any_swan, hospital_terminal) -> Tuple[float, bool]:
        if h_ratio < 0.95:
            self._state.v1_triggers += 1
            if hospital_terminal: return -1000.0, True
            return -1000.0, False

        freq_dev = abs(freq - FREQ_NOM)
        if freq_dev > 0.5:
            v2 = -500.0
            self._state.v2_triggers += 1
        elif freq_dev > 0.1:
            v2 = -100.0 * freq_dev
        else:
            v2 = 0.0

        if soc < 0.05:
            v3 = -300.0
            self._state.v3_triggers += 1
        elif soc < 0.15:
            v3 = -200.0 * (0.15 - soc)
            self._state.v3_triggers += 1
        elif soc > 0.95:
            v3 = -50.0 * (soc - 0.95)
            self._state.v3_triggers += 1
        else:
            v3 = 0.0

        excess_shed = max(0.0, cum_shed_ratio - 0.20)
        if excess_shed > 0: self._state.v4_triggers += 1
        # Order of magnitude smaller than Hospital (-1000) to avoid hesitation
        v4_budget = -10.0 * excess_shed
        v4_friction = -0.5 * (shed_this_step / max(total_demand, 1.0))
        v4 = v4_budget + v4_friction

        core = ((1.0 * h_ratio + 0.6 * i_ratio + 0.3 * r_ratio) * 100.0
                + 10.0 * renew_frac
                - 50.0 * wear_inc
                - 0.005 * 0) # Removed import cost from here to avoid double dipping, actual cost in state

        total = core + v2 + v3 + v4
        if any_swan: total *= 3.0

        return total, False

    def _build_obs(self, hour, reward, done, solar=0.0, wind=0.0, hosp_demand=HOSP_BASE_KW, hosp_ratio=1.0, ind_demand=IND_BASE_KW, ind_ratio=1.0, res_demand=RES_BASE_KW, res_ratio=1.0, p_import=0.0, cum_shed_ratio=0.0) -> GridObservation:
        t = hour
        freq = self._grid_frequency
        step = self._state.step_count
        
        s_fc, w_fc = self._get_forecast(horizon=6)

        return GridObservation(
            time_sin=round(math.sin(2 * math.pi * t / 24), 6),
            time_cos=round(math.cos(2 * math.pi * t / 24), 6),
            solar_norm=round(min(1.0, solar / MAX_SOLAR_KW), 4),
            wind_norm=round(min(1.0, wind / MAX_WIND_KW), 4),
            battery_soc=round(self._battery_soc, 4),
            bess_health=round(self._bess_health, 4),
            hosp_demand_norm=round(hosp_demand / 500.0, 4),
            hosp_served_ratio=round(max(0.0, min(1.0, hosp_ratio)), 4),
            ind_demand_norm=round(ind_demand / 500.0, 4),
            ind_served_ratio=round(max(0.0, min(1.0, ind_ratio)), 4),
            res_demand_norm=round(res_demand / 500.0, 4),
            res_served_ratio=round(max(0.0, min(1.0, res_ratio)), 4),
            frequency_norm=round((freq - FREQ_MIN) / (FREQ_MAX - FREQ_MIN), 4),
            grid_import_norm=round(min(1.0, p_import / 500.0), 4),
            solar_swan_active=1.0 if self._solar_swan_active else 0.0,
            wind_swan_active=1.0 if self._wind_swan_active else 0.0,
            step_norm=round(min(1.0, step / 24), 4),
            cumulative_shed_ratio_norm=round(min(1.0, cum_shed_ratio / 0.20), 4),
            forecast_solar_1h=round(s_fc[0] / MAX_SOLAR_KW, 4),
            forecast_solar_3h=round(s_fc[2] / MAX_SOLAR_KW, 4),
            forecast_wind_1h=round(w_fc[0] / MAX_WIND_KW, 4),
            forecast_wind_3h=round(w_fc[2] / MAX_WIND_KW, 4),
            reward=round(reward, 4),
            done=done,
            metadata={
                "episode_id": self._state.episode_id,
                "step": step,
                "grid_frequency_hz": round(freq, 4),
                "solar_kw": round(solar, 2),
                "wind_kw": round(wind, 2),
                "hosp_demand_kw": round(hosp_demand, 2),
                "ind_demand_kw": round(ind_demand, 2),
                "res_demand_kw": round(res_demand, 2),
                "p_import_kw": round(p_import, 2),
                "cum_shed_ratio": round(cum_shed_ratio, 4),
                "solar_swan": self._solar_swan_active,
                "wind_swan": self._wind_swan_active,
                "hospital_consecutive_fail": self._state.consecutive_hospital_failures,
                "blackout_count": self._state.blackout_count,
                "total_cost": round(self._state.total_cost, 4),
                "scenario": self._scenario
            }
        )

    @property
    def state(self) -> GridState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="energy_grid_env",
            description="v2.1: 24h 500kW microgrid, scenarios, forecasts, 22-dim obs.",
            version="2.1.0",
            author="OpenEnv Community",
        )

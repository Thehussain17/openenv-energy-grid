"""
Energy Grid Environment Client — v2 (Multi-Discrete Action / Normalized Obs).

Example (sync):
    env = EnergyGridEnv(base_url="http://localhost:7860").sync()
    with env:
        result = env.reset(seed=42)
        obs = result.observation
        action = GridAction(bess=3, hospital=0, industrial=1, residential=1)
        result = env.step(action)
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import GridAction, GridObservation, GridState
except ImportError:
    from models import GridAction, GridObservation, GridState


class EnergyGridEnv(EnvClient[GridAction, GridObservation, GridState]):
    """Client for the v2 Energy Grid Environment."""

    def _step_payload(self, action: GridAction) -> Dict:
        return {
            "bess":        action.bess,
            "hospital":    action.hospital,
            "industrial":  action.industrial,
            "residential": action.residential,
            "metadata":    action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[GridObservation]:
        obs_data = payload.get("observation", {})

        observation = GridObservation(
            time_sin=float(obs_data.get("time_sin", 0.0)),
            time_cos=float(obs_data.get("time_cos", 1.0)),
            solar_norm=float(obs_data.get("solar_norm", 0.0)),
            wind_norm=float(obs_data.get("wind_norm", 0.0)),
            battery_soc=float(obs_data.get("battery_soc", 0.5)),
            bess_health=float(obs_data.get("bess_health", 1.0)),
            hosp_demand_norm=float(obs_data.get("hosp_demand_norm", 0.0)),
            hosp_served_ratio=float(obs_data.get("hosp_served_ratio", 1.0)),
            ind_demand_norm=float(obs_data.get("ind_demand_norm", 0.0)),
            ind_served_ratio=float(obs_data.get("ind_served_ratio", 1.0)),
            res_demand_norm=float(obs_data.get("res_demand_norm", 0.0)),
            res_served_ratio=float(obs_data.get("res_served_ratio", 1.0)),
            frequency_norm=float(obs_data.get("frequency_norm", 0.5)),
            grid_import_norm=float(obs_data.get("grid_import_norm", 0.0)),
            solar_swan_active=float(obs_data.get("solar_swan_active", 0.0)),
            wind_swan_active=float(obs_data.get("wind_swan_active", 0.0)),
            step_norm=float(obs_data.get("step_norm", 0.0)),
            cumulative_shed_ratio_norm=float(obs_data.get("cumulative_shed_ratio_norm", 0.0)),
            done=bool(payload.get("done", False)),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> GridState:
        return GridState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            cumulative_reward=float(payload.get("cumulative_reward", 0.0)),
            total_cost=float(payload.get("total_cost", 0.0)),
            hospital_failure_steps=int(payload.get("hospital_failure_steps", 0)),
            consecutive_hospital_failures=int(payload.get("consecutive_hospital_failures", 0)),
            hospital_terminal_triggered=bool(payload.get("hospital_terminal_triggered", False)),
            total_energy_shed_kwh=float(payload.get("total_energy_shed_kwh", 0.0)),
            total_energy_demanded_kwh=float(payload.get("total_energy_demanded_kwh", 0.0)),
            renewable_energy_used=float(payload.get("renewable_energy_used", 0.0)),
            frequency_violation_steps=int(payload.get("frequency_violation_steps", 0)),
            bess_health=float(payload.get("bess_health", 1.0)),
            solar_swan_triggered=bool(payload.get("solar_swan_triggered", False)),
            wind_swan_triggered=bool(payload.get("wind_swan_triggered", False)),
            wind_swan_steps_remaining=int(payload.get("wind_swan_steps_remaining", 0)),
            blackout_count=int(payload.get("blackout_count", 0)),
            total_steps=int(payload.get("total_steps", 0)),
        )

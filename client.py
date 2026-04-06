"""
Energy Grid Environment Client.

Uses the OpenEnv WebSocket-based EnvClient with a synchronous wrapper
so callers (including inference.py) can use it without async/await.

Example (sync):
    env = EnergyGridEnv(base_url="http://localhost:7860").sync()
    with env:
        result = env.reset()
        obs = result.observation
        result = env.step(GridAction(decision="idle", magnitude=0.0))
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import GridAction, GridObservation, GridState
except ImportError:
    from models import GridAction, GridObservation, GridState


class EnergyGridEnv(EnvClient[GridAction, GridObservation, GridState]):
    """
    Client for the Energy Grid Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Use the `.sync()` method for synchronous access (e.g., in inference.py).

    Example (sync wrapper):
        env = EnergyGridEnv(base_url="http://localhost:7860").sync()
        with env:
            result = env.reset(seed=42)
            for _ in range(24):
                action = GridAction(decision="idle", magnitude=0.0)
                result = env.step(action)
                if result.done:
                    break

    Example (async):
        async with EnergyGridEnv(base_url="http://localhost:7860") as env:
            result = await env.reset(seed=42)
            result = await env.step(GridAction(decision="buy_external", magnitude=0.3))
    """

    def _step_payload(self, action: GridAction) -> Dict:
        """Serialize GridAction to the JSON dict expected by the server."""
        return {
            "decision": action.decision,
            "magnitude": float(action.magnitude),
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[GridObservation]:
        """Deserialize server response into StepResult[GridObservation]."""
        obs_data = payload.get("observation", {})

        observation = GridObservation(
            solar_output=float(obs_data.get("solar_output", 0.0)),
            wind_output=float(obs_data.get("wind_output", 0.0)),
            demand=float(obs_data.get("demand", 0.0)),
            battery_soc=float(obs_data.get("battery_soc", 0.5)),
            grid_frequency=float(obs_data.get("grid_frequency", 50.0)),
            time_of_day=int(obs_data.get("time_of_day", 0)),
            electricity_price=float(obs_data.get("electricity_price", 5.0)),
            renewable_fraction=float(obs_data.get("renewable_fraction", 0.0)),
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
        """Deserialize server state response into GridState."""
        return GridState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            cumulative_reward=float(payload.get("cumulative_reward", 0.0)),
            blackout_count=int(payload.get("blackout_count", 0)),
            total_cost=float(payload.get("total_cost", 0.0)),
            renewable_energy_used=float(payload.get("renewable_energy_used", 0.0)),
            frequency_violation_steps=int(payload.get("frequency_violation_steps", 0)),
            total_steps=int(payload.get("total_steps", 0)),
            demand_spike_hours=payload.get("demand_spike_hours"),
        )

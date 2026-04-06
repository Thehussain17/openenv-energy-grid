"""Quick smoke test: reset → step × 3 → state."""
from client import EnergyGridEnv
from models import GridAction

env = EnergyGridEnv(base_url="http://localhost:7860").sync()
with env:
    # ── reset ──────────────────────────────────────────────────────────
    result = env.reset(seed=42)
    obs = result.observation
    print(f"RESET  hour={obs.time_of_day}  demand={obs.demand:.1f}kW  soc={obs.battery_soc:.3f}")

    # ── 3 steps ────────────────────────────────────────────────────────
    actions = [
        GridAction(decision="idle",             magnitude=0.0),
        GridAction(decision="buy_external",     magnitude=0.5),
        GridAction(decision="battery_discharge",magnitude=0.7),
    ]
    for i, action in enumerate(actions, 1):
        result = env.step(action)
        obs    = result.observation
        print(
            f"STEP {i}  action={action.decision:<20}  "
            f"reward={result.reward:.4f}  done={result.done}"
        )

    # ── state ───────────────────────────────────────────────────────────
    state = env.state()
    print(
        f"STATE  episode_id={state.episode_id}  steps={state.step_count}"
        f"  cost={state.total_cost:.4f}  blackouts={state.blackout_count}"
    )

print("\nAll OK ✓")

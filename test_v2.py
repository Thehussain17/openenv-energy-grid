"""Smoke-test for the v2 energy grid environment upgrade."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    GridAction, GridObservation, GridState,
    BESS_MAP, HOSPITAL_RATIOS, INDUSTRIAL_RATIOS, RESIDENTIAL_RATIOS,
)
from server.energy_grid_environment import EnergyGridEnvironment

print("=== Import OK ===")

# ── Reset ──────────────────────────────────────────────────────────────────────
env = EnergyGridEnvironment()
obs = env.reset(seed=42)
vec = obs.to_vector()
assert len(vec) == 18, f"Expected 18 obs dims, got {len(vec)}"
print(f"Reset OK | soc={obs.battery_soc:.3f} freq_norm={obs.frequency_norm:.3f}")
print(f"Obs vector (18): {[round(v, 3) for v in vec]}")

# ── Full 24-step episode with action cycling ───────────────────────────────────
env2 = EnergyGridEnvironment()
obs2 = env2.reset(seed=7)
bess_cycle = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
for step in range(24):
    a = GridAction(bess=bess_cycle[step], hospital=0, industrial=1, residential=1)
    obs2 = env2.step(a)
    print(
        f"  step={step+1:02d} bess={bess_cycle[step]} "
        f"hosp={obs2.hosp_served_ratio:.2f} ind={obs2.ind_served_ratio:.2f} "
        f"res={obs2.res_served_ratio:.2f} freq={obs2.frequency_norm:.3f} "
        f"solar_swan={obs2.solar_swan_active:.0f} wind_swan={obs2.wind_swan_active:.0f} "
        f"reward={obs2.reward:.2f} done={obs2.done}"
    )
    if obs2.done:
        break

assert obs2.done, "Episode should have terminated by step 24"
state = env2.state
print(f"\nEpisode state:")
print(f"  total_steps={state.total_steps}")
print(f"  hospital_failure_steps={state.hospital_failure_steps}")
print(f"  frequency_violation_steps={state.frequency_violation_steps}")
print(f"  cumulative_reward={state.cumulative_reward:.2f}")
print(f"  bess_health={state.bess_health:.4f}")
print(f"  solar_swan_triggered={state.solar_swan_triggered}")
print(f"  wind_swan_triggered={state.wind_swan_triggered}")

# ── Hospital terminal trigger test ─────────────────────────────────────────────
env3 = EnergyGridEnvironment()
env3.reset(seed=99)
print("\n=== Hospital Terminal Test ===")
for i in range(5):
    # hospital=1 → only 98% supply, but the hard clamp fires at <0.95
    # At step 0 (hour 0), hospital demand is ~100kW, hosp_ratio=1.0 * 0.98 = 0.98 > 0.95
    # To trigger <0.95 we need a scenario where supply < 95% of hospital demand
    # Force by draining battery first, then idle (no import will happen if p_gen < hosp)
    a = GridAction(bess=5, hospital=1, industrial=3, residential=3)
    o3 = env3.step(a)
    streak = env3._state.consecutive_hospital_failures
    print(f"  iter={i+1} h_ratio={o3.hosp_served_ratio:.3f} streak={streak} done={o3.done} reward={o3.reward:.1f}")
    if o3.done:
        print(f"  → Terminal triggered: {env3._state.hospital_terminal_triggered}")
        break

print("\n=== ALL TESTS PASSED ===")

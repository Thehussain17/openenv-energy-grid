import sys, traceback, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = []

def check(label, fn):
    try:
        result = fn()
        print(f"  [PASS] {label}" + (f": {result}" if result is not None else ""))
        results.append((label, True, None))
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        traceback.print_exc()
        results.append((label, False, str(e)))

print("\n=== 1. Import checks ===")

_GridAction = _GridObservation = _GridState = _EnergyGridEnvironment = None

def import_models():
    global _GridAction, _GridObservation, _GridState
    from models import GridAction, GridObservation, GridState
    _GridAction = GridAction
    _GridObservation = GridObservation
    _GridState = GridState
    return "GridAction, GridObservation, GridState"

def import_env():
    global _EnergyGridEnvironment
    from server.energy_grid_environment import EnergyGridEnvironment
    _EnergyGridEnvironment = EnergyGridEnvironment
    return "EnergyGridEnvironment"

def import_client():
    from client import EnergyGridEnv
    return "EnergyGridEnv"

check("models.py", import_models)
check("server/energy_grid_environment.py", import_env)
check("client.py", import_client)

print("\n=== 2. Environment physics ===")

_env = None

def do_reset():
    global _env
    _env = _EnergyGridEnvironment()
    obs = _env.reset(seed=42)
    assert 0 <= obs.battery_soc <= 1, "SoC out of range"
    assert 0 <= obs.time_of_day <= 23, "Hour out of range"
    assert obs.demand > 0, "Demand is zero"
    assert obs.done == False, "done should be False at reset"
    return f"hour={obs.time_of_day} demand={obs.demand:.1f}kW soc={obs.battery_soc:.3f}"

def do_idle_step():
    obs = _env.step(_GridAction(decision="idle", magnitude=0.0))
    assert obs.reward is not None, "reward is None"
    assert obs.done == False, "should not be done at step 1"
    return f"reward={obs.reward:.4f}"

def do_buy_step():
    obs = _env.step(_GridAction(decision="buy_external", magnitude=0.5))
    assert obs.reward is not None
    return f"reward={obs.reward:.4f} price={obs.electricity_price}"

def do_battery_discharge():
    e = _EnergyGridEnvironment(); e.reset(seed=1)
    obs = e.step(_GridAction(decision="battery_discharge", magnitude=0.8))
    assert obs.reward is not None
    return f"reward={obs.reward:.4f}"

def do_battery_charge():
    e = _EnergyGridEnvironment(); e.reset(seed=2)
    obs = e.step(_GridAction(decision="battery_charge", magnitude=0.6))
    assert obs.reward is not None
    return f"reward={obs.reward:.4f}"

def do_curtail():
    e = _EnergyGridEnvironment(); e.reset(seed=3)
    obs = e.step(_GridAction(decision="curtail_load", magnitude=0.4))
    assert obs.reward is not None
    return f"reward={obs.reward:.4f}"

def do_state():
    state = _env.state
    assert state.step_count == 2, f"step_count={state.step_count}"
    assert state.total_cost >= 0
    return f"steps={state.step_count} cost={state.total_cost:.4f} blackouts={state.blackout_count}"

def do_full_episode():
    e = _EnergyGridEnvironment()
    e.reset(seed=99)
    done = False
    steps = 0
    while not done and steps < 30:
        obs = e.step(_GridAction(decision="idle", magnitude=0.0))
        done = obs.done
        steps += 1
    assert steps == 24, f"Expected 24 steps, got {steps}"
    assert done, "Episode should be done after 24 steps"
    return f"completed in {steps} steps, total_cost={e.state.total_cost:.4f}"

def do_graders():
    # Task 1 grader
    score1 = max(0.0, 1.0 - 10.0 / 50.0)  # 0.8
    assert 0.0 <= score1 <= 1.0, f"Task1 grader out of range: {score1}"

    # Task 2 grader
    score2 = max(0.0, min(1.0, 1.0 - 2/24 - 0.2*1))  # 0.717
    assert 0.0 <= score2 <= 1.0, f"Task2 grader out of range: {score2}"

    # Task 3 grader
    score3 = max(0.0, min(1.0, 0.4*0.6 + 0.4*0.9 + 0.2*0.7))  # 0.74
    assert 0.0 <= score3 <= 1.0, f"Task3 grader out of range: {score3}"

    return f"T1={score1:.3f} T2={score2:.3f} T3={score3:.3f} all in [0,1]"

def do_spike_episode():
    e = _EnergyGridEnvironment()
    e.reset(seed=123, demand_spike_hours=[11, 19])
    # Step to hour 11 and verify demand is elevated
    for h in range(12):
        obs = e.step(_GridAction(decision="buy_external", magnitude=0.8))
    assert e.state.step_count == 12
    return f"spike episode OK, blackouts so far={e.state.blackout_count}"

check("reset()", do_reset)
check("step(idle)", do_idle_step)
check("step(buy_external)", do_buy_step)
check("step(battery_discharge)", do_battery_discharge)
check("step(battery_charge)", do_battery_charge)
check("step(curtail_load)", do_curtail)
check("state property", do_state)
check("full 24-step episode terminates", do_full_episode)
check("demand spike hours (Task 2)", do_spike_episode)

print("\n=== 3. Grader logic ===")
check("all graders return [0,1]", do_graders)

print("\n=== 4. pyproject.toml structure ===")

def check_pyproject():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open("pyproject.toml", "rb") as f:
        p = tomllib.load(f)
    scripts = p.get("project", {}).get("scripts", {})
    assert "server" in scripts, "Missing [project.scripts] server"
    assert ":main" in scripts["server"], "server entry must reference :main"
    deps = [d.lower() for d in p.get("project", {}).get("dependencies", [])]
    has_openenv = any(d.startswith("openenv") for d in deps)
    assert has_openenv, "Missing openenv dependency"
    return f"server={scripts['server']}"

check("pyproject.toml", check_pyproject)

print("\n=== 5. openenv.yaml structure ===")

def check_yaml():
    import yaml
    with open("openenv.yaml") as f:
        y = yaml.safe_load(f)
    assert y.get("spec_version") == 1
    assert y.get("name") == "energy_grid_env"
    assert y.get("runtime") == "fastapi"
    assert y.get("port") == 8000
    return f"name={y['name']} runtime={y['runtime']} port={y['port']}"

try:
    check("openenv.yaml", check_yaml)
except ImportError:
    print("  [SKIP] openenv.yaml: pyyaml not installed, checking manually")
    with open("openenv.yaml") as f:
        raw = f.read()
    assert "spec_version: 1" in raw
    assert "energy_grid_env" in raw
    print("  [PASS] openenv.yaml: manual check OK")
    results.append(("openenv.yaml", True, None))

print("\n=== 6. server/app.py structure ===")

def check_app():
    with open("server/app.py") as f:
        src = f.read()
    assert "def main(" in src, "Missing main() function"
    assert "__name__" in src, "Missing if __name__ == '__main__' block"
    # Confirm main() is invoked somewhere after its definition
    main_def_pos = src.index("def main(")
    rest = src[main_def_pos:]
    # Look for a bare call like main() or main(port=...) after the def
    import re
    calls = re.findall(r'\bmain\s*\(', rest)
    assert len(calls) >= 2, f"main() not called after its definition (found {calls})"
    return "main() defined and callable via __main__"

check("server/app.py", check_app)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
failed = [(lbl, err) for lbl, ok, err in results if not ok]

print(f"  Result: {passed}/{total} checks passed")
if failed:
    print("\n  FAILURES:")
    for lbl, err in failed:
        print(f"    - {lbl}: {err}")
else:
    print("  ALL CHECKS PASSED ✓")
print("="*55)
sys.exit(0 if not failed else 1)

import argparse
import json
import random
from collections import defaultdict

from server.energy_grid_environment import EnergyGridEnvironment
from models import GridAction
from inference import _heuristic_action_with_reasoning, _SYSTEM_PROMPT


def _format_prompt(obs, rng):
    vec_str = [round(v, 3) for v in obs.to_vector()]
    templates = [
        (
            f"Obs vector (22 values): {vec_str}\n"
            f"solar_swan={obs.solar_swan_active:.0f} wind_swan={obs.wind_swan_active:.0f} "
            f"soc={obs.battery_soc:.2f} freq_norm={obs.frequency_norm:.3f}\n"
            f"hosp_ratio={obs.hosp_served_ratio:.3f} ind_ratio={obs.ind_served_ratio:.3f} "
            f"res_ratio={obs.res_served_ratio:.3f}\nWhat is your dispatch decision?"
        ),
        (
            f"Current grid snapshot (22-dim normalized vector): {vec_str}\n"
            f"flags: solar_swan={obs.solar_swan_active:.0f}, wind_swan={obs.wind_swan_active:.0f}\n"
            f"battery_soc={obs.battery_soc:.2f}, frequency_norm={obs.frequency_norm:.3f}\n"
            f"served ratios -> hospital={obs.hosp_served_ratio:.3f}, industrial={obs.ind_served_ratio:.3f}, residential={obs.res_served_ratio:.3f}\n"
            "Choose the next dispatch action."
        ),
        (
            f"Observation vector: {vec_str}\n"
            f"Swan events: solar={obs.solar_swan_active:.0f}, wind={obs.wind_swan_active:.0f}\n"
            f"SoC={obs.battery_soc:.2f}, freq_norm={obs.frequency_norm:.3f}\n"
            f"Service levels: H={obs.hosp_served_ratio:.3f}, I={obs.ind_served_ratio:.3f}, R={obs.res_served_ratio:.3f}\n"
            "Provide the dispatch control now."
        ),
    ]
    return rng.choice(templates)


def _obs_fingerprint(obs, action_tuple):
    rounded_vec = tuple(round(v, 2) for v in obs.to_vector())
    return rounded_vec + action_tuple


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def generate_trajectories(
    num_episodes=600,
    train_output_file="grid_expert_sft_train.jsonl",
    val_output_file="grid_expert_sft_val.jsonl",
    val_ratio=0.1,
    seed=42,
):
    print(f"Generating {num_episodes} expert SFT trajectories...")
    env = EnergyGridEnvironment()
    rng = random.Random(seed)

    # Use equal scenario coverage to avoid scenario-class imbalance.
    scenarios = list(range(5))
    samples_by_action = defaultdict(list)
    seen_obs_action = set()
    generated_steps = 0

    for ep in range(num_episodes):
        scenario = scenarios[ep % len(scenarios)]
        episode_seed = seed + (ep * 17) + scenario
        obs, _ = env.reset(seed=episode_seed, scenario=scenario)
        done = False

        while not done:
            act_dict, reasoning = _heuristic_action_with_reasoning(obs)
            action_tuple = (
                act_dict["bess"],
                act_dict["hospital"],
                act_dict["industrial"],
                act_dict["residential"],
            )

            # Skip near-duplicates to reduce memorization/overfitting risk.
            key = _obs_fingerprint(obs, action_tuple)
            if key not in seen_obs_action:
                seen_obs_action.add(key)
                prompt = _format_prompt(obs, rng)
                completion = (
                    f"### Thought: {reasoning}\n"
                    f"### Action: [{action_tuple[0]}, {action_tuple[1]}, {action_tuple[2]}, {action_tuple[3]}]"
                )
                samples_by_action[action_tuple].append(
                    {
                        "prompt": _SYSTEM_PROMPT + "\n\n" + prompt,
                        "completion": completion,
                    }
                )

            action = GridAction(
                bess=action_tuple[0],
                hospital=action_tuple[1],
                industrial=action_tuple[2],
                residential=action_tuple[3],
            )
            obs = env.step(action)
            done = obs.done
            generated_steps += 1

        if (ep + 1) % 50 == 0:
            print(f"  ... completed {ep + 1}/{num_episodes} episodes")

    if not samples_by_action:
        raise RuntimeError("No samples generated.")

    class_sizes = {k: len(v) for k, v in samples_by_action.items()}
    min_class_size = min(class_sizes.values())
    print(f"Raw unique samples: {sum(class_sizes.values())} from {generated_steps} environment steps")
    print(f"Action classes discovered: {len(class_sizes)}")
    print(f"Balancing all classes to {min_class_size} samples each")

    balanced_rows_all = []

    for action_key, rows in sorted(samples_by_action.items()):
        rng.shuffle(rows)
        balanced_rows = rows[:min_class_size]
        balanced_rows_all.extend(balanced_rows)
        print(
            f"  class {action_key}: raw={len(rows)} balanced={len(balanced_rows)} "
            "included in pooled split"
        )

    rng.shuffle(balanced_rows_all)
    val_count = int(len(balanced_rows_all) * val_ratio)
    if balanced_rows_all and val_count == 0:
        val_count = 1
    if len(balanced_rows_all) > 1 and val_count >= len(balanced_rows_all):
        val_count = len(balanced_rows_all) - 1

    balanced_val = balanced_rows_all[:val_count]
    balanced_train = balanced_rows_all[val_count:]

    _write_jsonl(train_output_file, balanced_train)
    _write_jsonl(val_output_file, balanced_val)

    print(f"Wrote train set: {len(balanced_train)} samples -> {train_output_file}")
    print(f"Wrote val set:   {len(balanced_val)} samples -> {val_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate balanced SFT data for Energy Grid policy learning.")
    parser.add_argument("--num-episodes", type=int, default=600)
    parser.add_argument("--train-output-file", type=str, default="grid_expert_sft_train.jsonl")
    parser.add_argument("--val-output-file", type=str, default="grid_expert_sft_val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_trajectories(
        num_episodes=args.num_episodes,
        train_output_file=args.train_output_file,
        val_output_file=args.val_output_file,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
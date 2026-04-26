import argparse
import json
import math
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
    max_oversample_ratio=4,
):
    """
    Collect expert trajectories and produce a class-balanced SFT dataset.

    Balancing strategy: oversample rare classes (with replacement up to
    `max_oversample_ratio × majority-class size`) rather than shrinking
    the majority. This keeps the dataset large while still reducing the
    bias toward the dominant action.

    Parameters
    ----------
    num_episodes         : int   Number of environment episodes to roll out.
    train_output_file    : str   Path for the train JSONL.
    val_output_file      : str   Path for the validation JSONL.
    val_ratio            : float Fraction of balanced dataset held out for val.
    seed                 : int   RNG seed for reproducibility.
    max_oversample_ratio : int   Rare classes are upsampled to at most
                                 (majority_size // max_oversample_ratio).
                                 Keeps oversampling sensible even if one
                                 class has only 1 raw sample.
    """
    print(f"Generating {num_episodes} expert SFT trajectories...")
    env = EnergyGridEnvironment()
    rng = random.Random(seed)

    scenarios = list(range(5))
    samples_by_action = defaultdict(list)
    seen_obs_action = set()
    generated_steps = 0

    for ep in range(num_episodes):
        scenario = scenarios[ep % len(scenarios)]
        episode_seed = seed + (ep * 17) + scenario
        obs = env.reset(seed=episode_seed, scenario=scenario)
        done = False

        while not done:
            act_dict, reasoning = _heuristic_action_with_reasoning(obs)
            action_tuple = (
                act_dict["bess"],
                act_dict["hospital"],
                act_dict["industrial"],
                act_dict["residential"],
            )

            key = _obs_fingerprint(obs, action_tuple)
            if key not in seen_obs_action:
                seen_obs_action.add(key)
                prompt = _format_prompt(obs, rng)
                completion = (
                    f"### Thought: {reasoning}\n"
                    f"### Action: [{action_tuple[0]}, {action_tuple[1]}, "
                    f"{action_tuple[2]}, {action_tuple[3]}]"
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
    majority_size = max(class_sizes.values())
    # Target: upsample minority classes toward majority, but cap growth so a
    # class with 1 raw sample doesn't dominate through heavy repetition.
    target_size = max(1, majority_size // max_oversample_ratio)

    raw_total = sum(class_sizes.values())
    print(f"Raw unique samples  : {raw_total} from {generated_steps} env steps")
    print(f"Action classes      : {len(class_sizes)}")
    print(f"Majority class size : {majority_size}  |  oversample target: {target_size}")

    balanced_rows_all = []

    for action_key, rows in sorted(samples_by_action.items()):
        rng.shuffle(rows)
        n_raw = len(rows)
        if n_raw >= target_size:
            # Majority (or close to it): keep as-is, no shrinking.
            balanced = rows[:]
        else:
            # Minority: oversample with replacement up to target_size.
            repeats = math.ceil(target_size / n_raw)
            pool = (rows * repeats)[:target_size]
            rng.shuffle(pool)
            balanced = pool

        balanced_rows_all.extend(balanced)
        print(
            f"  class {action_key}: raw={n_raw:4d}  "
            f"after_balance={len(balanced):4d}"
        )

    rng.shuffle(balanced_rows_all)

    val_count = int(len(balanced_rows_all) * val_ratio)
    val_count = max(1, val_count)
    val_count = min(val_count, len(balanced_rows_all) - 1)

    balanced_val   = balanced_rows_all[:val_count]
    balanced_train = balanced_rows_all[val_count:]

    _write_jsonl(train_output_file, balanced_train)
    _write_jsonl(val_output_file, balanced_val)

    print(f"\nWrote train set : {len(balanced_train):5d} samples -> {train_output_file}")
    print(f"Wrote val set   : {len(balanced_val):5d} samples -> {val_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate balanced SFT data for Energy Grid policy learning.")
    parser.add_argument("--num-episodes", type=int, default=600)
    parser.add_argument("--train-output-file", type=str, default="grid_expert_sft_train.jsonl")
    parser.add_argument("--val-output-file", type=str, default="grid_expert_sft_val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-oversample-ratio", type=int, default=4,
        help="Rare classes are oversampled up to majority_size // this value. "
             "Lower = more aggressive upsampling. Default 4."
    )
    args = parser.parse_args()

    generate_trajectories(
        num_episodes=args.num_episodes,
        train_output_file=args.train_output_file,
        val_output_file=args.val_output_file,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_oversample_ratio=args.max_oversample_ratio,
    )
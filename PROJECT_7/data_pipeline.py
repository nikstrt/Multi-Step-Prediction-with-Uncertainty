# =====================================================================
# Stochastic Data Collection & Saving
# =====================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import pandas as pd
import gymnasium as gym

# ── FIX: Global reproducibility seed ──────────────────────────────────
SEED = 42
np.random.seed(SEED)


def run_stochastic_data_collection(
    num_trajectories: int = 500,
    output_csv: str = "cartpole_stochastic_transitions.csv",
    metadata_json: str = "collection_metadata.json",
):
    """
    Collects stochastic CartPole transition data using a mostly-random policy
    with occasional PD-controller corrections.

    System model:  x_{t+1} = f(x_t, a_t) + noise,  noise ~ N(0, noise_std^2)

    Args:
        num_trajectories: Number of episodes to collect.
        output_csv:       Path to save the transitions CSV.
        metadata_json:    Path to save collection hyperparameters & stats.
    """
    env = gym.make("CartPole-v1")

    # ── Collection hyperparameters ────────────────────────────────────
    noise_std   = np.array([0.01, 0.02, 0.005, 0.02], dtype=np.float32)
    Kp, Kd      = 1.5, 0.3          # PD gains (used for the 5% non-random steps)
    epsilon     = 0.95              # Fraction of uniformly-random actions
    init_low    = np.array([-1.8, -0.5, -0.105, -0.5], dtype=np.float32)
    init_high   = np.array([ 1.8,  0.5,  0.105,  0.5], dtype=np.float32)

    all_steps           = []
    terminated_count    = 0
    truncated_count     = 0
    episode_lengths     = []

    print(f"[+] Launching data factory  |  seed={SEED}  |  "
          f"{int(epsilon*100)}% random  |  {num_trajectories} episodes")

    for episode_idx in range(num_trajectories):
        obs, _ = env.reset()

        # Force wide-span custom start
        custom_start = np.random.uniform(low=init_low, high=init_high).astype(np.float32)
        env.unwrapped.state = custom_start
        obs = custom_start.copy()

        step_count = 0
        done       = False

        while not done:
            pole_angle, pole_velocity = obs[2], obs[3]

            # ε-greedy policy: mostly random, occasionally PD
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                pd_force = (pole_angle * Kp) + (pole_velocity * Kd)
                action   = 1 if pd_force > 0 else 0

            next_obs, _, terminated, truncated, _ = env.step(action)

            # Inject zero-mean Gaussian process noise
            noise        = np.random.normal(loc=0.0, scale=noise_std).astype(np.float32)
            next_obs_n   = (next_obs + noise).astype(np.float32)

            all_steps.append({
                "trajectory_id":        episode_idx,
                "step_id":              step_count,
                "cart_position":        float(obs[0]),
                "cart_velocity":        float(obs[1]),
                "pole_angle":           float(obs[2]),
                "pole_angular_velocity":float(obs[3]),
                "action":               int(action),
                "next_cart_position":        float(next_obs_n[0]),
                "next_cart_velocity":        float(next_obs_n[1]),
                "next_pole_angle":           float(next_obs_n[2]),
                "next_pole_angular_velocity":float(next_obs_n[3]),
                "terminated": int(terminated),
                "truncated":  int(truncated),
            })

            obs        = next_obs_n
            step_count += 1
            done       = terminated or truncated

        # ── Per-episode statistics ────────────────────────────
        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        episode_lengths.append(step_count)

    env.close()

    df = pd.DataFrame(all_steps)
    df.to_csv(output_csv, index=False)

    # ── Save all hyperparameters & stats to JSON ────────────────
    metadata = {
        "seed":              SEED,
        "num_trajectories":  num_trajectories,
        "epsilon":           epsilon,
        "Kp":                Kp,
        "Kd":                Kd,
        "noise_std":         noise_std.tolist(),
        "init_low":          init_low.tolist(),
        "init_high":         init_high.tolist(),
        "total_transitions": len(df),
        "terminated_episodes": terminated_count,
        "truncated_episodes":  truncated_count,
        "avg_episode_length":  float(np.mean(episode_lengths)),
        "min_episode_length":  int(np.min(episode_lengths)),
        "max_episode_length":  int(np.max(episode_lengths)),
    }
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary printout ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total transitions  : {len(df):,}")
    print(f"  Terminated episodes: {terminated_count}")
    print(f"  Truncated episodes : {truncated_count}")
    print(f"  Avg episode length : {np.mean(episode_lengths):.1f} steps")
    print(f"  Min episode length : {np.min(episode_lengths)}")
    print(f"  Max episode length : {np.max(episode_lengths)}")
    print(f"  Saved CSV          : '{output_csv}'")
    print(f"  Saved metadata     : '{metadata_json}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_stochastic_data_collection()
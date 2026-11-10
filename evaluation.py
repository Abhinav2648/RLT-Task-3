"""
evaluation.py
-------------
Evaluation utilities for comparing the trained Q-Learning agent
against a random-policy baseline.

Both policies are evaluated on the same maze with the same step limit.
Only the action-selection strategy differs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from maze_env import MazeEnv, N_ACTIONS
from q_agent  import QLearningAgent


# ── Single-policy evaluation ──────────────────────────────────────────────────

def evaluate_agent(
    agent:    QLearningAgent,
    n_trials: int = 100,
    seed:     int = 0,
) -> Dict:
    """
    Evaluate the trained (greedy) policy over n_trials episodes.

    Parameters
    ----------
    agent    : QLearningAgent  Trained agent.
    n_trials : int             Number of evaluation episodes.
    seed     : int             RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        success_rate   : float  Fraction of episodes where goal was reached.
        avg_steps      : float  Mean steps per successful episode (0 if none).
        avg_reward     : float  Mean total reward across all episodes.
        rewards        : list   Per-episode total rewards.
        steps          : list   Per-episode steps taken.
        successes      : list   Per-episode boolean success flags.
    """
    env = agent.env
    rng = np.random.default_rng(seed)

    rewards   : List[float] = []
    steps     : List[int]   = []
    successes : List[bool]  = []

    for _ in range(n_trials):
        state   = env.reset()
        total_r = 0.0
        success = False

        for _ in range(agent.max_steps):
            action = int(np.argmax(agent.q_table[state]))   # greedy
            next_state, reward, done, info = env.step(action)
            total_r += reward
            state    = next_state
            if done:
                if info.get("reason") == "goal_reached":
                    success = True
                break

        rewards.append(total_r)
        steps.append(env.steps_taken)
        successes.append(success)

    successful_steps = [s for s, ok in zip(steps, successes) if ok]
    return {
        "success_rate": float(np.mean(successes)),
        "avg_steps":    float(np.mean(successful_steps)) if successful_steps else 0.0,
        "avg_reward":   float(np.mean(rewards)),
        "rewards":      rewards,
        "steps":        steps,
        "successes":    successes,
    }


def evaluate_random(
    env:      MazeEnv,
    max_steps:int = 200,
    n_trials: int = 100,
    seed:     int = 1,
) -> Dict:
    """
    Evaluate a purely random policy (uniform random action each step).

    Returns the same dict shape as evaluate_agent().
    """
    rng = np.random.default_rng(seed)

    rewards   : List[float] = []
    steps     : List[int]   = []
    successes : List[bool]  = []

    for _ in range(n_trials):
        env.reset()
        total_r = 0.0
        success = False

        for _ in range(max_steps):
            action = int(rng.integers(N_ACTIONS))
            _, reward, done, info = env.step(action)
            total_r += reward
            if done:
                if info.get("reason") == "goal_reached":
                    success = True
                break

        rewards.append(total_r)
        steps.append(env.steps_taken)
        successes.append(success)

    successful_steps = [s for s, ok in zip(steps, successes) if ok]
    return {
        "success_rate": float(np.mean(successes)),
        "avg_steps":    float(np.mean(successful_steps)) if successful_steps else 0.0,
        "avg_reward":   float(np.mean(rewards)),
        "rewards":      rewards,
        "steps":        steps,
        "successes":    successes,
    }


# ── Moving-average helper ─────────────────────────────────────────────────────

def moving_average(values: List[float], window: int = 50) -> np.ndarray:
    """
    Compute an unweighted moving average with the given window size.
    For the first (window-1) entries the window is shrunk to avoid
    leading NaN values.
    """
    arr    = np.array(values, dtype=float)
    result = np.empty_like(arr)
    for i in range(len(arr)):
        start      = max(0, i - window + 1)
        result[i]  = arr[start : i + 1].mean()
    return result


# ── Full comparison ───────────────────────────────────────────────────────────

def run_full_evaluation(
    agent:    QLearningAgent,
    n_trials: int = 200,
) -> Dict:
    """
    Run both the trained-agent and random-baseline evaluations
    and return a combined results dict.
    """
    trained = evaluate_agent(agent, n_trials=n_trials, seed=0)
    random  = evaluate_random(agent.env, max_steps=agent.max_steps,
                               n_trials=n_trials, seed=1)
    return {"trained": trained, "random": random}


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from maze_env import MazeEnv
    from q_agent  import QLearningAgent

    env   = MazeEnv()
    agent = QLearningAgent(env, n_episodes=2000)
    agent.train()

    results = run_full_evaluation(agent, n_trials=200)

    t = results["trained"]
    r = results["random"]

    print("\nTrained Agent")
    print(f"  Success rate : {t['success_rate']*100:.1f}%")
    print(f"  Avg steps    : {t['avg_steps']:.1f}")
    print(f"  Avg reward   : {t['avg_reward']:.2f}")

    print("\nRandom Policy")
    print(f"  Success rate : {r['success_rate']*100:.1f}%")
    print(f"  Avg steps    : {r['avg_steps']:.1f}")
    print(f"  Avg reward   : {r['avg_reward']:.2f}")

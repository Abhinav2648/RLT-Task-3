"""
q_agent.py
----------
Tabular Q-Learning agent for the MazeEnv environment.

Algorithm
---------
    Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]

For terminal states the target is simply `r` (no bootstrapping).

Exploration
-----------
    Epsilon-greedy: with probability ε choose a random action,
    otherwise choose the greedy action argmax_a Q(s,a).
    ε decays multiplicatively after each episode until it reaches ε_min.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple

from maze_env import MazeEnv, N_ACTIONS


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    env          : MazeEnv   The environment to interact with.
    alpha        : float     Learning rate (0 < α ≤ 1).
    gamma        : float     Discount factor (0 ≤ γ ≤ 1).
    epsilon      : float     Initial exploration probability.
    epsilon_decay: float     Multiplicative decay applied after each episode.
    epsilon_min  : float     Floor value for epsilon.
    n_episodes   : int       Number of training episodes.
    max_steps    : int       Maximum steps per episode (overrides env if given).
    seed         : int       Random seed for reproducibility.
    """

    def __init__(
        self,
        env:           MazeEnv,
        alpha:         float = 0.1,
        gamma:         float = 0.99,
        epsilon:       float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min:   float = 0.01,
        n_episodes:    int   = 2000,
        max_steps:     int   = 200,
        seed:          int   = 42,
    ):
        self.env           = env
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.n_episodes    = n_episodes
        self.max_steps     = max_steps
        self.seed          = seed

        # Seeded RNG for reproducibility
        self.rng = np.random.default_rng(seed)

        # Q-table: shape (n_states, n_actions), initialised to zero
        self.q_table = np.zeros((env.n_states, N_ACTIONS), dtype=np.float64)

        # Training history (populated by train())
        self.episode_rewards:     List[float] = []
        self.episode_steps:       List[int]   = []
        self.episode_successes:   List[bool]  = []
        self.epsilon_history:     List[float] = []
        self.is_trained: bool = False

    # ── Core methods ──────────────────────────────────────────────────────────

    def select_action(self, state: int, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state  : int   Current state index.
        greedy : bool  If True, always pick the greedy action (used at eval time).
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self.q_table[state]))

    def train(self, progress_callback=None) -> Dict:
        """
        Run the Q-Learning training loop.

        Parameters
        ----------
        progress_callback : callable, optional
            Called with (episode, total_episodes, epsilon) after each episode.
            Useful for updating a progress bar in the UI.

        Returns
        -------
        dict  Training history with keys:
              rewards, steps, successes, epsilon_history.
        """
        # Reset training history
        self.episode_rewards   = []
        self.episode_steps     = []
        self.episode_successes = []
        self.epsilon_history   = []

        # Reset Q-table so retraining starts fresh
        self.q_table = np.zeros((self.env.n_states, N_ACTIONS), dtype=np.float64)

        # Reset epsilon to initial value
        current_epsilon = 1.0

        for episode in range(self.n_episodes):
            state    = self.env.reset()
            total_r  = 0.0
            success  = False

            for _ in range(self.max_steps):
                # ε-greedy action selection
                if self.rng.random() < current_epsilon:
                    action = int(self.rng.integers(N_ACTIONS))
                else:
                    action = int(np.argmax(self.q_table[state]))

                next_state, reward, done, info = self.env.step(action)
                total_r += reward

                # Q-Learning update (Bellman equation)
                if done and info.get("reason") == "goal_reached":
                    # Terminal state: target = reward only (no bootstrapping)
                    td_target = reward
                    success   = True
                else:
                    td_target = reward + self.gamma * np.max(self.q_table[next_state])

                td_error            = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error

                state = next_state
                if done:
                    break

            # Decay epsilon
            current_epsilon = max(self.epsilon_min, current_epsilon * self.epsilon_decay)

            # Record history
            self.episode_rewards.append(total_r)
            self.episode_steps.append(self.env.steps_taken)
            self.episode_successes.append(success)
            self.epsilon_history.append(current_epsilon)

            if progress_callback and (episode % max(1, self.n_episodes // 100) == 0):
                progress_callback(episode + 1, self.n_episodes, current_epsilon)

        self.epsilon = current_epsilon
        self.is_trained = True
        return {
            "rewards":         self.episode_rewards,
            "steps":           self.episode_steps,
            "successes":       self.episode_successes,
            "epsilon_history": self.epsilon_history,
        }

    def get_learned_path(self) -> List[Tuple[int, int]]:
        """
        Greedily follow the learned policy from start to goal.

        Returns the list of (row, col) positions visited, or an empty list
        if the agent fails to reach the goal within max_steps.
        """
        self.env.reset()
        path = [self.env.agent_pos]

        for _ in range(self.max_steps):
            state  = self.env.current_state
            action = int(np.argmax(self.q_table[state]))
            _, _, done, info = self.env.step(action)
            path.append(self.env.agent_pos)
            if done:
                if info.get("reason") == "goal_reached":
                    return path
                break  # timed out or stuck

        return []  # failed to reach goal

    def reset_qtable(self):
        """Reset the Q-table and training history for a fresh run."""
        self.q_table       = np.zeros((self.env.n_states, N_ACTIONS), dtype=np.float64)
        self.episode_rewards   = []
        self.episode_steps     = []
        self.episode_successes = []
        self.epsilon_history   = []
        self.is_trained        = False


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    env   = MazeEnv()
    agent = QLearningAgent(env, n_episodes=2000)

    print("Training Q-Learning agent …")
    t0   = time.time()
    hist = agent.train()
    dt   = time.time() - t0

    n_success = sum(hist["successes"])
    print(f"Training done in {dt:.2f}s")
    print(f"Episodes trained  : {agent.n_episodes}")
    print(f"Successful episodes: {n_success} / {agent.n_episodes}"
          f"  ({100*n_success/agent.n_episodes:.1f}%)")
    print(f"Final epsilon     : {agent.epsilon:.4f}")

    path = agent.get_learned_path()
    if path:
        print(f"Learned path ({len(path)} steps): {path}")
    else:
        print("Agent failed to reach goal on greedy run.")

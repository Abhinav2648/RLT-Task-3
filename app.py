"""
app.py
------
Streamlit dashboard for "Intelligent Maze Navigation Using Reinforcement Learning".

Sections
--------
A. Project Overview
B. Maze Environment  (interactive grid + controls)
C. Training Controls (hyperparameters + train button)
D. Learning Performance (training graphs)
E. Evaluation Results (trained vs random comparison)
F. Technical Explanation
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from maze_env   import MazeEnv, DEFAULT_MAZE, N_ACTIONS, ACTION_NAMES
from q_agent    import QLearningAgent
from evaluation import (
    evaluate_agent, evaluate_random, run_full_evaluation, moving_average
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Intelligent Maze Navigation – RL",
    page_icon  = "🤖",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global colour palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "wall":   "#2c3e50",
    "free":   "#ecf0f1",
    "start":  "#27ae60",
    "goal":   "#e74c3c",
    "agent":  "#3498db",
    "path":   "#f39c12",
    "bg":     "#ffffff",
    "accent": "#8e44ad",
}

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    if "env" not in st.session_state:
        st.session_state.env        = MazeEnv()
    if "agent" not in st.session_state:
        st.session_state.agent      = None
    if "trained" not in st.session_state:
        st.session_state.trained    = False
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
    if "learned_path" not in st.session_state:
        st.session_state.learned_path = []
    if "show_path" not in st.session_state:
        st.session_state.show_path  = False
    if "train_history" not in st.session_state:
        st.session_state.train_history = None
    if "auto_trained" not in st.session_state:
        st.session_state.auto_trained = False

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-train helper (runs once on first page load with default params)
# ─────────────────────────────────────────────────────────────────────────────
def _auto_train():
    """Silently train with default hyperparameters on first app load."""
    env   = st.session_state.env
    agent = QLearningAgent(
        env          = env,
        alpha        = 0.1,
        gamma        = 0.99,
        epsilon_decay = 0.995,
        n_episodes   = 3000,
        max_steps    = 200,
        seed         = 42,
    )
    hist = agent.train()
    st.session_state.agent         = agent
    st.session_state.train_history = hist
    st.session_state.trained       = True
    st.session_state.auto_trained  = True
    # Extract and store learned path too
    path = agent.get_learned_path()
    st.session_state.learned_path  = path
    st.session_state.show_path     = bool(path)
    # Run evaluation
    st.session_state.eval_results  = run_full_evaluation(agent, n_trials=200)

# ─────────────────────────────────────────────────────────────────────────────
# Maze rendering helper
# ─────────────────────────────────────────────────────────────────────────────
def render_maze_figure(
    env:        MazeEnv,
    path:       list  = None,
    show_qvals: bool  = False,
    q_table:    np.ndarray = None,
) -> plt.Figure:
    """Return a Matplotlib Figure of the maze grid."""
    nrows, ncols = env.n_rows, env.n_cols
    fig, ax = plt.subplots(figsize=(ncols * 0.85, nrows * 0.85))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    path_set = set(path) if path else set()

    for r in range(nrows):
        for c in range(ncols):
            cell_val = env.maze_layout[r, c]
            pos      = (r, c)

            if cell_val == 1:                          # wall
                color = COLORS["wall"]
            elif pos == env.goal_pos:
                color = COLORS["goal"]
            elif pos == env.start_pos:
                color = COLORS["start"]
            elif pos in path_set:
                color = COLORS["path"]
            else:
                color = COLORS["free"]

            rect = mpatches.FancyBboxPatch(
                (c + 0.05, nrows - r - 1 + 0.05),
                0.90, 0.90,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="#bdc3c7",
                linewidth=0.8,
            )
            ax.add_patch(rect)

            # Cell labels
            cx, cy = c + 0.5, nrows - r - 0.5
            if pos == env.agent_pos:
                ax.text(cx, cy, "🤖", ha="center", va="center", fontsize=14)
            elif pos == env.goal_pos:
                ax.text(cx, cy, "🏆", ha="center", va="center", fontsize=14)
            elif pos == env.start_pos:
                ax.text(cx, cy, "🚀", ha="center", va="center", fontsize=9)
            elif cell_val == 1:
                ax.text(cx, cy, "■", ha="center", va="center",
                        fontsize=10, color="#ecf0f1")
            elif pos in path_set and pos not in (env.start_pos, env.goal_pos):
                ax.text(cx, cy, "·", ha="center", va="center",
                        fontsize=20, color="#e67e22")

            # Optional: max Q-value overlay
            if show_qvals and q_table is not None and cell_val != 1:
                s    = env._pos_to_state(pos)
                qmax = np.max(q_table[s])
                ax.text(cx, cy - 0.25, f"{qmax:.1f}",
                        ha="center", va="center", fontsize=5.5, color="#555")

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["wall"],  label="Wall"),
        mpatches.Patch(facecolor=COLORS["start"], label="Start (🚀)"),
        mpatches.Patch(facecolor=COLORS["goal"],  label="Goal (🏆)"),
        mpatches.Patch(facecolor=COLORS["agent"], label="Agent (🤖)"),
        mpatches.Patch(facecolor=COLORS["path"],  label="Learned path"),
        mpatches.Patch(facecolor=COLORS["free"],  label="Free cell"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training graph helpers
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_training_curves(history: dict, window: int = 50) -> plt.Figure:
    """Four-panel training-curve figure."""
    rewards   = history["rewards"]
    steps     = history["steps"]
    successes = history["successes"]
    eps_hist  = history["epsilon_history"]

    episodes  = np.arange(1, len(rewards) + 1)
    ma_rewards = moving_average(rewards,   window)
    ma_steps   = moving_average(steps,     window)

    # Rolling success rate
    success_arr = np.array(successes, dtype=float)
    ma_success  = moving_average(success_arr, window) * 100

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor("#f8f9fa")

    # ── Panel 1: Episode reward + moving average ──
    ax = axes[0, 0]
    ax.plot(episodes, rewards,    alpha=0.25, color="#3498db", linewidth=0.6,
            label="Raw reward")
    ax.plot(episodes, ma_rewards, color="#e74c3c",  linewidth=1.8,
            label=f"MA ({window})")
    ax.axhline(0, color="#95a5a6", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=8)
    _style_ax(ax, "Episode Reward", "Episode", "Total Reward")

    # ── Panel 2: Steps per episode ──
    ax = axes[0, 1]
    ax.plot(episodes, steps,    alpha=0.25, color="#27ae60", linewidth=0.6,
            label="Steps")
    ax.plot(episodes, ma_steps, color="#8e44ad",  linewidth=1.8,
            label=f"MA ({window})")
    ax.legend(fontsize=8)
    _style_ax(ax, "Steps per Episode", "Episode", "Steps")

    # ── Panel 3: Rolling success rate ──
    ax = axes[1, 0]
    ax.plot(episodes, ma_success, color="#f39c12", linewidth=1.8)
    ax.set_ylim(-5, 105)
    ax.axhline(100, color="#95a5a6", linestyle=":", linewidth=0.8)
    _style_ax(ax, f"Rolling Success Rate (window={window})", "Episode", "Success Rate (%)")

    # ── Panel 4: Epsilon decay ──
    ax = axes[1, 1]
    ax.plot(episodes, eps_hist, color="#e67e22", linewidth=1.8)
    ax.set_ylim(-0.05, 1.05)
    _style_ax(ax, "Exploration Rate (ε) Decay", "Episode", "ε (Epsilon)")

    plt.tight_layout(pad=2.0)
    return fig


def plot_comparison(eval_results: dict) -> plt.Figure:
    """Bar chart comparing trained agent vs random policy."""
    trained = eval_results["trained"]
    random  = eval_results["random"]

    metrics = ["Success Rate (%)", "Avg Steps (success)", "Avg Reward"]
    trained_vals = [
        trained["success_rate"] * 100,
        trained["avg_steps"],
        trained["avg_reward"],
    ]
    random_vals = [
        random["success_rate"] * 100,
        random["avg_steps"],
        random["avg_reward"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.patch.set_facecolor("#f8f9fa")

    bar_colors = {"Trained Agent": "#3498db", "Random Policy": "#e74c3c"}
    x = np.arange(1)

    for i, (metric, tv, rv) in enumerate(zip(metrics, trained_vals, random_vals)):
        ax = axes[i]
        b1 = ax.bar(x - 0.2, tv, 0.35, label="Trained Agent",
                    color=bar_colors["Trained Agent"], alpha=0.85)
        b2 = ax.bar(x + 0.2, rv, 0.35, label="Random Policy",
                    color=bar_colors["Random Policy"], alpha=0.85)
        ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=9)
        ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=9)
        ax.set_xticks([])
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION A: Header + Project Overview ─────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤖 Intelligent Maze Navigation Using Reinforcement Learning")
st.markdown(
    "<p style='color:#666; font-size:14px;'>"
    "B.Tech AIML · Fundamentals of Reinforcement Learning · Assignment Demo"
    "</p>",
    unsafe_allow_html=True,
)

# ── Auto-train on very first page load ───────────────────────────────────────
if not st.session_state.auto_trained:
    with st.status("🚀 Auto-training agent with default settings…", expanded=True) as status_box:
        st.write("Setting up maze environment…")
        st.write("Running Q-Learning for 3 000 episodes…")
        _auto_train()
        status_box.update(
            label="✅ Agent trained and ready! (3 000 episodes)",
            state="complete",
            expanded=False,
        )
    st.rerun()   # refresh so sidebar status and all sections render correctly

# ── Persistent status banner ─────────────────────────────────────────────────
if st.session_state.trained:
    hist_snap = st.session_state.train_history
    n_succ    = sum(hist_snap["successes"]) if hist_snap else 0
    n_total   = len(hist_snap["rewards"])   if hist_snap else 0
    sr        = 100 * n_succ / max(n_total, 1)
    st.success(
        f"✅ Agent is trained | {n_total} episodes | "
        f"Training success rate: **{sr:.1f}%** | "
        f"Click **🔍 Show Learned Path** in Section B to see the route."
    )
else:
    st.warning(
        "⚠️ Agent not yet trained. "
        "Use the **🚀 Train Agent** button in **Section B** below to start training."
    )

with st.expander("📖 A. Project Overview", expanded=True):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
### Problem Statement
A robot is placed at the **start** of a grid maze.  It must find its way to the
**goal** by navigating around walls, learning entirely from trial-and-error
experience — *no map, no prior knowledge*.

### What is Reinforcement Learning?
Reinforcement Learning (RL) is a branch of machine learning where an **agent**
learns to make decisions by interacting with an **environment**.

| Component | Description |
|-----------|-------------|
| **Agent** | The decision-maker (our robot) |
| **Environment** | The maze grid |
| **State (s)** | Current position in the maze |
| **Action (a)** | Move UP / DOWN / LEFT / RIGHT |
| **Reward (r)** | Feedback signal after each action |
| **Policy (π)** | Strategy: which action to take in each state |

The agent's objective is to learn a **policy** that maximises the
*cumulative discounted reward* over time.
""")

    with col2:
        st.markdown("""
### Reward Design

| Event | Reward | Purpose |
|-------|--------|---------|
| 🏆 Reach the goal | **+100** | Strong incentive to solve the maze |
| ➡️ Legal move | **−1** | Encourage finding *short* paths |
| 🧱 Hit a wall / boundary | **−5** | Discourage illegal moves |

### Objective
Train a Q-Learning agent so that after sufficient experience it:

- Reaches the goal **reliably** from the start.
- Follows a **short, efficient** path.
- Performs **significantly better** than a random-action policy.
""")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION C: Sidebar – Training Controls ───────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ C. Training Controls")

n_episodes    = st.sidebar.slider("Training Episodes",    500,  10000, 3000, 500)
alpha         = st.sidebar.slider("Learning Rate (α)",    0.01,  1.0,  0.1,  0.01)
gamma         = st.sidebar.slider("Discount Factor (γ)",  0.80,  0.999, 0.99, 0.001, format="%.3f")
epsilon_decay = st.sidebar.slider("Epsilon Decay",        0.980, 0.999, 0.995, 0.001, format="%.3f")
max_steps     = st.sidebar.slider("Max Steps / Episode",  50,    500,   200,   10)

st.sidebar.markdown("---")
reward_goal = st.sidebar.number_input("Reward – Goal",       value=100.0,  step=10.0)
reward_step = st.sidebar.number_input("Reward – Step",       value=-1.0,   step=1.0)
reward_wall = st.sidebar.number_input("Reward – Wall hit",   value=-5.0,   step=1.0)
eval_trials = st.sidebar.slider("Evaluation Trials",         50, 500, 200, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("**Current status:**")
if st.session_state.trained:
    hist_s = st.session_state.train_history
    n_s    = sum(hist_s["successes"]) if hist_s else 0
    n_t    = len(hist_s["rewards"])   if hist_s else 0
    st.sidebar.success(f"✅ Agent trained\n\n{n_s}/{n_t} successes ({100*n_s/max(n_t,1):.1f}%)")
    if st.session_state.eval_results:
        er = st.session_state.eval_results["trained"]
        st.sidebar.metric("Eval success rate", f"{er['success_rate']*100:.1f}%")
        st.sidebar.metric("Avg steps (success)", f"{er['avg_steps']:.1f}")
else:
    st.sidebar.warning("⚠️ Agent not yet trained")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION B: Maze Environment ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🗺️ B. Maze Environment")

env = st.session_state.env

# Buttons row
bcol1, bcol2, bcol3, bcol4 = st.columns(4)

train_clicked = bcol1.button("🚀 Train Agent", type="primary", use_container_width=True)
path_clicked  = bcol2.button("🔍 Show Learned Path", use_container_width=True,
                              disabled=not st.session_state.trained)
reset_clicked = bcol3.button("🔄 Reset Environment", use_container_width=True)
retrain_clicked = bcol4.button("🗑️ Reset & Retrain", use_container_width=True)

show_qvals = st.checkbox("Overlay max Q-values on grid", value=False)

# ── Reset environment ────────────────────────────────────────────────────────
if reset_clicked:
    env.reset()
    st.session_state.show_path   = False
    st.session_state.learned_path = []

if retrain_clicked:
    env.reset()
    st.session_state.agent        = None
    st.session_state.trained      = False
    st.session_state.eval_results = None
    st.session_state.learned_path = []
    st.session_state.show_path    = False
    st.session_state.train_history = None
    st.rerun()

# ── Train agent ──────────────────────────────────────────────────────────────
if train_clicked:
    # Re-create env with current reward settings
    new_env = MazeEnv(
        reward_goal = reward_goal,
        reward_step = reward_step,
        reward_wall = reward_wall,
        max_steps   = max_steps,
    )
    st.session_state.env = new_env
    env = new_env

    agent = QLearningAgent(
        env          = env,
        alpha        = alpha,
        gamma        = gamma,
        epsilon_decay = epsilon_decay,
        n_episodes   = n_episodes,
        max_steps    = max_steps,
        seed         = 42,
    )
    st.session_state.agent   = agent
    st.session_state.trained = False

    progress_bar = st.progress(0, text="Training … 0%")
    status_text  = st.empty()

    def _cb(ep, total, eps):
        pct = ep / total
        progress_bar.progress(pct, text=f"Training … {ep}/{total} episodes  ε={eps:.4f}")
        status_text.info(f"Episode {ep}/{total} — ε = {eps:.4f}")

    t0   = time.time()
    hist = agent.train(progress_callback=_cb)
    dt   = time.time() - t0

    st.session_state.train_history = hist
    st.session_state.trained       = True
    st.session_state.show_path     = False
    st.session_state.learned_path  = []

    progress_bar.progress(1.0, text="Training complete ✅")
    n_success = sum(hist["successes"])
    status_text.success(
        f"Training done in {dt:.1f}s — "
        f"{n_success}/{n_episodes} successful episodes "
        f"({100*n_success/n_episodes:.1f}%)"
    )

    # Run evaluation immediately
    with st.spinner("Evaluating agent …"):
        st.session_state.eval_results = run_full_evaluation(
            agent, n_trials=eval_trials
        )

# ── Show learned path ────────────────────────────────────────────────────────
if path_clicked and st.session_state.trained:
    agent = st.session_state.agent
    path  = agent.get_learned_path()
    if path:
        st.session_state.learned_path = path
        st.session_state.show_path    = True
        st.success(f"Learned path found — {len(path)-1} steps.")
    else:
        st.error("Agent failed to reach the goal on the greedy run. Try more episodes.")
        st.session_state.show_path    = False
        st.session_state.learned_path = []

# ── Render maze ───────────────────────────────────────────────────────────────
agent     = st.session_state.agent
path_disp = st.session_state.learned_path if st.session_state.show_path else []
q_disp    = agent.q_table if (show_qvals and agent) else None

maze_fig = render_maze_figure(env, path=path_disp, show_qvals=show_qvals, q_table=q_disp)

mcol1, mcol2 = st.columns([3, 2])
with mcol1:
    st.pyplot(maze_fig, use_container_width=False)
    plt.close(maze_fig)

with mcol2:
    st.markdown("### 🧭 Maze Key")
    st.markdown("""
| Symbol | Meaning |
|--------|---------|
| 🚀 | Start position |
| 🏆 | Goal |
| 🤖 | Agent |
| ■ dark | Wall / obstacle |
| 🟡 orange | Learned path |
| ⬜ light grey | Free cell |

**Maze size:** `6 × 6` grid  
**Start:** top-left cell  
**Goal:** bottom-right cell  
**Actions:** UP · DOWN · LEFT · RIGHT
""")
    st.markdown("### 🔢 Maze Statistics")
    n_walls = int(np.sum(env.maze_layout == 1))
    n_free  = int(np.sum(env.maze_layout != 1))
    st.metric("Total cells",   env.n_rows * env.n_cols)
    st.metric("Wall cells",    n_walls)
    st.metric("Navigable cells", n_free)
    if st.session_state.show_path and st.session_state.learned_path:
        st.metric("Path length (steps)", len(st.session_state.learned_path) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION D: Learning Performance ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📈 D. Learning Performance")

if st.session_state.train_history is None:
    st.info("Train the agent first (click **🚀 Train Agent**) to see performance graphs.")
else:
    hist = st.session_state.train_history

    window = st.slider("Moving-average window", 10, 200, 50, 10)
    fig_curves = plot_training_curves(hist, window=window)
    st.pyplot(fig_curves, use_container_width=True)
    plt.close(fig_curves)

    # Summary stats from training
    n_eps     = len(hist["rewards"])
    n_succ    = sum(hist["successes"])
    last_q    = int(n_eps * 0.1)          # last 10 % of episodes
    final_sr  = np.mean(hist["successes"][-last_q:]) * 100

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    dcol1.metric("Total episodes",        n_eps)
    dcol2.metric("Successful episodes",   n_succ)
    dcol3.metric(f"Success rate (last {last_q})", f"{final_sr:.1f}%")
    dcol4.metric("Final ε",
                 f"{hist['epsilon_history'][-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION E: Evaluation Results ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 E. Evaluation Results")

if st.session_state.eval_results is None:
    st.info("Evaluation runs automatically after training. Train the agent first.")
else:
    results = st.session_state.eval_results
    t_res   = results["trained"]
    r_res   = results["random"]

    # ── Comparison bar chart ──
    fig_cmp = plot_comparison(results)
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

    # ── Metric cards ──
    st.markdown("#### Trained Agent vs Random Policy")
    ecol1, ecol2 = st.columns(2)

    with ecol1:
        st.markdown("**🤖 Trained Agent**")
        ec1a, ec1b, ec1c = st.columns(3)
        ec1a.metric("Success Rate",  f"{t_res['success_rate']*100:.1f}%")
        ec1b.metric("Avg Steps ✓",   f"{t_res['avg_steps']:.1f}")
        ec1c.metric("Avg Reward",    f"{t_res['avg_reward']:.1f}")

    with ecol2:
        st.markdown("**🎲 Random Policy**")
        ec2a, ec2b, ec2c = st.columns(3)
        ec2a.metric("Success Rate",  f"{r_res['success_rate']*100:.1f}%")
        ec2b.metric("Avg Steps ✓",   f"{r_res['avg_steps']:.1f}")
        ec2c.metric("Avg Reward",    f"{r_res['avg_reward']:.1f}")

    # ── Improvement deltas ──
    sr_delta   = (t_res["success_rate"] - r_res["success_rate"]) * 100
    rew_delta  = t_res["avg_reward"] - r_res["avg_reward"]
    st.success(
        f"✅ Trained agent improves success rate by **{sr_delta:+.1f}%** "
        f"and average reward by **{rew_delta:+.1f}** over the random policy."
    )

    # ── Detailed table ──
    st.markdown("#### Detailed Metrics Table")
    df = pd.DataFrame({
        "Metric": [
            "Success Rate (%)",
            "Avg Steps (successful episodes only)",
            "Avg Total Reward (all episodes)",
            "Evaluation Trials",
        ],
        "Trained Agent": [
            f"{t_res['success_rate']*100:.2f}",
            f"{t_res['avg_steps']:.2f}" if t_res["avg_steps"] > 0 else "N/A",
            f"{t_res['avg_reward']:.2f}",
            len(t_res["rewards"]),
        ],
        "Random Policy": [
            f"{r_res['success_rate']*100:.2f}",
            f"{r_res['avg_steps']:.2f}" if r_res["avg_steps"] > 0 else "N/A",
            f"{r_res['avg_reward']:.2f}",
            len(r_res["rewards"]),
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Learned path ──
    st.markdown("#### 🗺️ Learned Path")
    agent_obj = st.session_state.agent
    lpath = agent_obj.get_learned_path() if agent_obj else []
    if lpath:
        path_str = " → ".join([str(p) for p in lpath])
        st.code(path_str, language=None)
        st.success(f"Goal reached in **{len(lpath)-1} steps** on the greedy run.")
    else:
        st.warning("Agent did not reach the goal on the greedy evaluation run.")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION F: Technical Explanation ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🔬 F. Technical Explanation")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Q-Learning",
    "Exploration vs Exploitation",
    "Bellman Update",
    "Why Does It Work?",
    "Limitations",
])

with tab1:
    st.markdown("""
## Q-Learning

Q-Learning is a **model-free**, **off-policy** RL algorithm introduced by
Christopher Watkins (1989).

- **Model-free**: The agent does not need to know the transition probabilities
  or reward function in advance. It learns purely from experience.
- **Off-policy**: The algorithm can learn the optimal policy even while the
  agent follows a different (exploratory) policy.

The agent maintains a **Q-table** — a 2-D array of shape
`(n_states × n_actions)` — where `Q(s, a)` represents the agent's estimate of
the **expected cumulative discounted reward** from state `s` taking action `a`
and following the optimal policy thereafter.

At convergence, the optimal policy is simply:

$$\\pi^*(s) = \\arg\\max_a \\, Q^*(s, a)$$

**In this project:**
- States  = 36 grid cells (6 × 6 maze)
- Actions = 4 (UP, DOWN, LEFT, RIGHT)
- Q-table = 36 × 4 = 144 values
""")

with tab2:
    st.markdown("""
## Exploration vs Exploitation

A fundamental dilemma in RL:

| Strategy | Description | Risk |
|----------|-------------|------|
| **Exploit** | Always pick the action with the highest Q-value | May miss better strategies (local optima) |
| **Explore** | Try random actions | May waste time on poor actions |

### Epsilon-Greedy Policy

$$a = \\begin{cases} \\text{random action} & \\text{with probability } \\varepsilon \\\\ \\arg\\max_a Q(s,a) & \\text{with probability } 1 - \\varepsilon \\end{cases}$$

- At the **start** ε = 1.0 → agent explores freely.
- After each episode: ε ← max(ε_min, ε × decay).
- At the **end** ε ≈ ε_min → agent exploits the learned policy.

This *annealing schedule* ensures the agent explores enough early on,
then commits to what it has learned.
""")

with tab3:
    st.markdown("""
## The Q-Learning (Bellman) Update

After each transition `(s, a, r, s', done)`:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s,a) \\right]$$

| Symbol | Meaning |
|--------|---------|
| α (alpha) | Learning rate — how fast new info overwrites old estimates |
| γ (gamma) | Discount factor — how much future rewards are valued |
| r | Immediate reward |
| max Q(s',a') | Best future value estimate (bootstrapping) |
| **TD error** | `r + γ max Q(s',a') - Q(s,a)` — the prediction error |

**Terminal state handling:**  
When the goal is reached (`done = True`), there is no next state, so the
target reduces to just `r`:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha \\left[ r - Q(s,a) \\right]$$

This avoids incorrectly bootstrapping from a non-existent future state.
""")

with tab4:
    st.markdown("""
## Why Does the Policy Improve?

1. **Early episodes**: The agent wanders randomly. Occasionally it stumbles upon
   the goal and receives +100. That reward propagates back through the Q-table.

2. **Q-values propagate**: Each update nudges `Q(s, a)` slightly toward the
   true value. Over thousands of episodes, values from goal-adjacent states
   propagate backwards through the maze.

3. **Exploitation increases**: As ε decreases, the agent increasingly follows
   the improving Q-table, reaching the goal more reliably and in fewer steps.

4. **Convergence guarantee**: Under standard conditions (every state-action pair
   visited infinitely often, decaying learning rate), Q-Learning is proven to
   converge to the optimal Q-function `Q*`.
""")

with tab5:
    st.markdown("""
## Limitations of Tabular Q-Learning

| Limitation | Explanation |
|------------|-------------|
| **Scalability** | The Q-table size grows as `states × actions`. For large or continuous spaces (e.g., Atari, robotics) this is infeasible. Deep Q-Networks (DQN) replace the table with a neural network. |
| **Slow convergence** | Tabular Q-Learning may need millions of episodes for complex environments. |
| **No generalisation** | The agent cannot generalise knowledge between similar states. Each state is treated independently. |
| **Discrete actions only** | Q-Learning naturally handles discrete action spaces. Continuous control requires actor-critic methods (e.g., PPO, SAC). |
| **Partial observability** | Assumes the agent sees its full state. Real robots often have partial or noisy observations (POMDPs). |

### Extensions
- **DQN** (Deep Q-Network): neural-network function approximation.
- **Double DQN**: reduces overestimation bias.
- **Dueling DQN**: separate value and advantage streams.
- **Policy Gradient / PPO**: directly optimise the policy.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:12px;'>"
    "Intelligent Maze Navigation Using Reinforcement Learning · "
    "B.Tech AIML · Fundamentals of Reinforcement Learning · "
    "Built with Python, NumPy, Matplotlib & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)

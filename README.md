# Intelligent Maze Navigation Using Reinforcement Learning

> B.Tech AIML · Fundamentals of Reinforcement Learning Assignment

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** in your browser.

---

## Project Structure

```
RLT Task 3/
├── maze_env.py       # Grid maze environment (reset, step, render, BFS path check)
├── q_agent.py        # Tabular Q-Learning agent (train, evaluate, get_learned_path)
├── evaluation.py     # Evaluation utilities (trained vs random policy comparison)
├── app.py            # Streamlit interactive dashboard (all 6 sections)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## How to Use the Dashboard

1. **Open the sidebar** (left panel) to configure hyperparameters:
   - Training episodes, learning rate α, discount factor γ, epsilon decay
   - Reward values and evaluation trial count

2. **Click "🚀 Train Agent"** — the agent trains in real-time with a progress bar.

3. **Click "🔍 Show Learned Path"** — the learned route is overlaid on the maze.

4. **Scroll to Section D** to view four training-curve graphs:
   - Episode reward + moving average
   - Steps per episode + moving average
   - Rolling success rate
   - Epsilon decay

5. **Scroll to Section E** to see:
   - Trained agent vs random policy bar charts
   - Metric table (success rate, avg steps, avg reward)
   - The greedy learned path

6. **Section F** (tabs at the bottom) explains Q-Learning theory.

---

## Algorithm

**Q-Learning (Watkins, 1989)**

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]
```

| Symbol | Meaning |
|--------|---------|
| α      | Learning rate |
| γ      | Discount factor |
| r      | Immediate reward |
| TD error | `r + γ·max Q(s',a') - Q(s,a)` |

For terminal states: target = `r` (no bootstrapping).

---

## Maze Layout (Default 6×6)

```
S . . # . .
. # . # . #
. # . . . .
. . # # . #
# . . . . .
. . # # . G
```

`S` = Start · `G` = Goal · `#` = Wall · `.` = Free cell

A valid path exists from S to G (verified via BFS at environment creation).

---

## Reward Structure

| Event | Reward | Rationale |
|-------|--------|-----------|
| Reach goal | +100 | Strong incentive to solve the maze |
| Legal move | −1 | Encourages shorter paths |
| Wall / boundary hit | −5 | Discourages illegal moves |

---

## Results (default settings, 3000 episodes)

| Metric | Trained Agent | Random Policy |
|--------|--------------|---------------|
| Success rate | ~100% | ~50% |
| Avg steps (success) | ~10 | ~109 |
| Avg total reward | ~91 | ~−400 |

---

## Dependencies

- Python 3.8+
- `streamlit >= 1.32`
- `numpy >= 1.24`
- `matplotlib >= 3.7`
- `pandas >= 2.0`

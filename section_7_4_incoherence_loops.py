
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

plt.rcParams.update({
    "font.size": 13,            # Base font size
    "axes.titlesize": 15,       # Title font
    "axes.labelsize": 13,       # x/y label size
    "xtick.labelsize": 11,      # Tick label size
    "ytick.labelsize": 11,
    "legend.fontsize": 12,      # Legend text size
})

# Simulation Parameters
np.random.seed(42)
n_agents = 100
n_rounds = 1000
coherence_threshold = 0.4
dropout_threshold = 0.2
k = 5
epsilon = 0.05
modes = ["Contraction", "Postponement", "Oscillation", "Addition", "Submission", "Subsumption"]
mode_indices = {m: i for i, m in enumerate(modes)}

# Define coherence function
def coherence_score(mode, t):
    if mode == "Contraction":
        return max(0, 1 - 0.02 * t)
    elif mode == "Postponement":
        return 0.3 if t < 25 else 0.6
    elif mode == "Oscillation":
        return 0.5 + 0.4 * np.sin(0.3 * t)
    elif mode == "Addition":
        return min(1, 0.1 * t)
    elif mode == "Submission":
        return min(1, 0.05 * t + 0.3)
    elif mode == "Subsumption":
        return 0.6 - 0.1 * np.exp(-0.05 * t)
    return 0.5

# Initialize agents
agents = []
for i in range(n_agents):
    agents.append({
        'id': i,
        'mode': np.random.choice(modes),
        'memory': defaultdict(list),
        'coherence': [],
        'history': [],
        'switch_count': 0,
        'dropout': False
    })

# Run simulation
for t in range(n_rounds):
    np.random.shuffle(agents)
    pairs = [(agents[i], agents[i + 1]) for i in range(0, n_agents - 1, 2)]

    for A, B in pairs:
        if A['dropout'] or B['dropout']:
            continue

        for agent in [A, B]:
            cscore = coherence_score(agent['mode'], t)
            agent['coherence'].append(cscore)
            agent['memory'][agent['mode']].append(cscore)
            agent['history'].append(agent['mode'])

            if len(agent['coherence']) > k and np.mean(agent['coherence'][-k:]) < coherence_threshold:
                agent['switch_count'] += 1
                if np.random.rand() < epsilon:
                    agent['mode'] = np.random.choice([m for m in modes if m != agent['mode']])
                else:
                    avg_scores = {m: np.mean(scores) for m, scores in agent['memory'].items() if len(scores) >= 3}
                    if avg_scores:
                        agent['mode'] = max(avg_scores, key=avg_scores.get)

            if not agent['dropout'] and len(agent['coherence']) >= k:
                if np.mean(agent['coherence'][-k:]) < dropout_threshold:
                    agent['dropout'] = True

# Plot only the zoomed-in view
fig, ax = plt.subplots(figsize=(10, 3.5))
for agent in agents:
    indices = [mode_indices[m] for m in agent['history'][:100]]
    ax.plot(indices, linewidth=0.6, alpha=0.6)

ax.set_title("Early Structural Instability (0–100 Rounds)", fontsize=14)
ax.set_xlabel("Rounds", fontsize=12)
ax.set_ylabel("Mode Index", fontsize=12)
ax.set_yticks(list(mode_indices.values()))
ax.set_yticklabels(list(mode_indices.keys()), fontsize=10)
ax.tick_params(axis='x', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("incoherence_loops_traj.png", dpi=300)
plt.close()

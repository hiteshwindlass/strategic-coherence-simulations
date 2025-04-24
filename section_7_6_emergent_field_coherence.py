
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from scipy.stats import entropy

plt.rcParams.update({
    "font.size": 13,            # Base font size
    "axes.titlesize": 15,       # Title font
    "axes.labelsize": 13,       # x/y label size
    "xtick.labelsize": 11,      # Tick label size
    "ytick.labelsize": 11,
    "legend.fontsize": 12,      # Legend text size
})

# Parameters
n_agents = 100
n_rounds = 500
k_neighbors = 4
rewiring_prob = 0.1
coherence_threshold = 0.4
dropout_threshold = 0.2
k = 5
epsilon = 0.05
neighbor_penalty = 0.30

modes = ["Contraction", "Postponement", "Oscillation", "Addition", "Submission", "Subsumption"]

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

G = nx.watts_strogatz_graph(n=n_agents, k=k_neighbors, p=rewiring_prob, seed=42)

agents = []
for i in range(n_agents):
    agents.append({
        'id': i,
        'mode': np.random.choice(modes),
        'memory': defaultdict(list),
        'coherence': [],
        'history': [],
        'switch_count': 0,
        'dropout': False,
        'dropout_time': None
    })

avg_coherence, coherence_std, mode_entropy, phase_synchrony = [], [], [], []

for t in range(n_rounds):
    all_scores = []
    mode_counts = Counter()
    for i, agent in enumerate(agents):
        if agent['dropout']: continue

        cscore = coherence_score(agent['mode'], t)
        for neighbor in G.neighbors(i):
            if agents[neighbor]['dropout']:
                cscore -= neighbor_penalty / len(list(G.neighbors(i)))
        cscore = max(cscore, 0)

        agent['coherence'].append(cscore)
        agent['memory'][agent['mode']].append(cscore)
        agent['history'].append(agent['mode'])
        all_scores.append(cscore)
        mode_counts[agent['mode']] += 1

        if len(agent['coherence']) > k and np.mean(agent['coherence'][-k:]) < coherence_threshold:
            agent['switch_count'] += 1
            if np.random.rand() < epsilon:
                agent['mode'] = np.random.choice([m for m in modes if m != agent['mode']])
            else:
                avg_scores = {m: np.mean(scores) for m, scores in agent['memory'].items() if len(scores) >= 3}
                if avg_scores:
                    agent['mode'] = max(avg_scores, key=avg_scores.get)

        if len(agent['coherence']) >= k and np.mean(agent['coherence'][-k:]) < dropout_threshold:
            agent['dropout'] = True
            agent['dropout_time'] = t

    if all_scores:
        avg_coherence.append(np.mean(all_scores))
        coherence_std.append(np.std(all_scores))
        total = sum(mode_counts.values())
        probs = [v / total for v in mode_counts.values()]
        mode_entropy.append(entropy(probs))

        slopes = [np.mean(np.diff(agent['coherence'][-5:])) if len(agent['coherence']) > 5 else 0 for agent in agents if not agent['dropout']]
        if len(slopes) > 1:
            phases = np.angle(np.exp(1j * np.array(slopes)))
            vector_strength = np.abs(np.mean(np.exp(1j * phases)))
            phase_synchrony.append(vector_strength)
        else:
            phase_synchrony.append(0)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(14, 12))

axs[0].plot(avg_coherence, label="Mean Coherence")
axs[0].fill_between(range(n_rounds),
                    np.array(avg_coherence) - np.array(coherence_std),
                    np.array(avg_coherence) + np.array(coherence_std),
                    alpha=0.2, label="Coherence Spread")
axs[0].set_title("A. Average Coherence and Spread")
axs[0].set_ylabel("Coherence")
axs[0].legend()

axs[1].plot(phase_synchrony, color='purple')
axs[1].set_title("B. Phase Synchrony Index (PSI) Across Agents")
axs[1].set_ylabel("Synchrony (0–1)")

axs[2].plot(mode_entropy, color='darkgreen')
axs[2].set_title("C. Structural Mode Entropy Over Time")
axs[2].set_xlabel("Rounds")
axs[2].set_ylabel("Entropy")

plt.tight_layout()
plt.savefig("field_coherence.png")
plt.close()

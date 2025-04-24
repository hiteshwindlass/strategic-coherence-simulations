
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

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

# Generate small-world network
G = nx.watts_strogatz_graph(n=n_agents, k=k_neighbors, p=rewiring_prob, seed=42)

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
        'dropout': False,
        'dropout_time': None
    })

# Run simulation
for t in range(n_rounds):
    for i, agent in enumerate(agents):
        if agent['dropout']:
            continue

        cscore = coherence_score(agent['mode'], t)
        neighbors = list(G.neighbors(i))
        for neighbor in neighbors:
            if agents[neighbor]['dropout']:
                cscore -= neighbor_penalty / len(neighbors)

        cscore = max(cscore, 0)
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

        if len(agent['coherence']) >= k and np.mean(agent['coherence'][-k:]) < dropout_threshold:
            agent['dropout'] = True
            agent['dropout_time'] = t

# Extract dropout times
dropout_times = [(a['id'], a['dropout_time']) for a in agents if a['dropout']]
dropout_times.sort(key=lambda x: x[1])
ordered_agents = [x[0] for x in dropout_times]
dropout_rounds = [x[1] for x in dropout_times]

# Build cascade edges
edges = []
for i, agent_id in enumerate(ordered_agents):
    t = agents[agent_id]['dropout_time']
    for neighbor_id in G.neighbors(agent_id):
        neighbor_time = agents[neighbor_id]['dropout_time']
        if neighbor_time is not None and neighbor_time < t:
            edges.append(((agent_id, t), (neighbor_id, neighbor_time)))

# Plot cascade
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 10))
y_vals = range(len(ordered_agents))
x_vals = dropout_rounds

plt.scatter(x_vals, y_vals, color='crimson', s=50, marker='o', zorder=3)
for ((aid1, t1), (aid2, t2)) in edges:
    y1 = ordered_agents.index(aid1)
    y2 = ordered_agents.index(aid2)
    plt.annotate('', xy=(t1, y1), xytext=(t2, y2),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, alpha=0.6),
                 zorder=1)

plt.title("Section 4.6 – Cascade Plot (Small-World Network, 500 Rounds)")
plt.xlabel("Simulation Round")
plt.ylabel("Agent (ordered by dropout time)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("collapse_cascade.png")
plt.close()

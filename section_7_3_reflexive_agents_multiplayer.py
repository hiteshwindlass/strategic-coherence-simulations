
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter



# --- PANEL A: PARAMETER SWEEP ---

theta_vals = [0.3, 0.4, 0.5]
k_vals = [3, 5, 7]
epsilon_vals = [0.01, 0.05, 0.1]
interaction_types = {
    'soft': 0.1,
    'strong': 0.3,
    'symmetric': 0.2
}

def evolve(mode, t):
    if mode == "Contraction":
        return np.exp(-0.05 * t)
    elif mode == "Postponement":
        return np.log1p(t) / np.log1p(500)
    elif mode == "Oscillation":
        return 0.5 * np.sin(0.4 * t) + 0.5
    elif mode == "Addition":
        return min(1.0, 0.01 * (t ** 1.5))
    elif mode == "Submission":
        return 1 - np.exp(-0.1 * t)
    elif mode == "Subsumption":
        return 1 / (1 + np.exp(-0.1 * (t - 50)))
    return 0.5

def run_dropout_sim(theta, k, epsilon, interaction_bonus, n_agents=50, T=100):
    modes = ["Contraction", "Postponement", "Oscillation", "Addition", "Submission", "Subsumption"]
    dropouts = 0
    for i in range(n_agents):
        mode = np.random.choice(modes)
        coherence = []
        for t in range(T):
            base = evolve(mode, t)
            partner = np.random.choice(modes)
            interaction = interaction_bonus if mode == partner else -interaction_bonus
            total = max(0, min(1.5, base + interaction))
            coherence.append(total)
            if t >= k and np.mean(coherence[-k:]) < theta:
                if np.mean(coherence[-k:]) < 0.1:
                    dropouts += 1
                break
    return dropouts / n_agents

dropout_by_theta = [run_dropout_sim(theta=θ, k=5, epsilon=0.05, interaction_bonus=0.2) for θ in theta_vals]
dropout_by_k = [run_dropout_sim(theta=0.3, k=kv, epsilon=0.05, interaction_bonus=0.2) for kv in k_vals]
dropout_by_eps = [run_dropout_sim(theta=0.3, k=5, epsilon=ϵ, interaction_bonus=0.2) for ϵ in epsilon_vals]
dropout_by_inter = [run_dropout_sim(theta=0.3, k=5, epsilon=0.05, interaction_bonus=val)
                    for val in interaction_types.values()]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes[0, 0].bar([str(v) for v in theta_vals], dropout_by_theta, color="dodgerblue")
axes[0, 0].set_title("Dropout by θ")

axes[0, 1].bar([str(v) for v in k_vals], dropout_by_k, color="coral")
axes[0, 1].set_title("Dropout by k")

axes[1, 0].bar([str(v) for v in epsilon_vals], dropout_by_eps, color="seagreen")
axes[1, 0].set_title("Dropout by ε")

axes[1, 1].bar(list(interaction_types.keys()), dropout_by_inter, color="orchid")
axes[1, 1].set_title("Dropout by Interaction Type")

fig.suptitle("A. Dropout Across All Parameters", fontsize=14)
plt.tight_layout()
plt.savefig("panel_A_dropout_sweep.png")
plt.close()

# --- PANELS B, C, D: LONG SIMULATION ---

np.random.seed(42)
N = 50
T = 500
k = 5
theta = 0.3
epsilon = 0.05

modes = ["Contraction", "Postponement", "Oscillation", "Addition", "Submission", "Subsumption"]

def compatibility(m1, m2):
    if m1 == m2:
        return 0.2
    elif (m1, m2) in [("Addition", "Submission"), ("Submission", "Addition"), ("Subsumption", "Submission"),
                      ("Subsumption", "Addition")]:
        return 0.1
    else:
        return -0.1

def update(mode, prev, t):
    if mode == "Contraction":
        return prev * 0.9
    elif mode == "Postponement":
        return prev if t < 25 else prev + 0.1
    elif mode == "Oscillation":
        return 1 + 0.5 * np.sin(0.4 * t)
    elif mode == "Addition":
        return prev + 0.2
    elif mode == "Submission":
        return prev * 1.1
    elif mode == "Subsumption":
        return (prev + 2) / 2
    return prev

agents = []
dropout_timeline = []
for i in range(N):
    agents.append({
        'id': i,
        'mode': np.random.choice(modes),
        'coherence': [],
        'memory': defaultdict(list),
        'history': [],
        'dropout': False,
        'dropout_round': T
    })

for t in range(T):
    for agent in agents:
        if agent['dropout']:
            agent['coherence'].append(0)
            continue

        mode = agent['mode']
        prev_val = agent['coherence'][-1] if agent['coherence'] else 1.0
        self_coh = update(mode, prev_val, t)

        partner = np.random.choice([a for a in agents if a['id'] != agent['id']])
        interaction = compatibility(mode, partner['mode'])

        coh = max(0, min(2, self_coh + interaction))
        agent['coherence'].append(coh)
        agent['memory'][mode].append(coh)
        agent['history'].append((mode, coh))

        if t >= k:
            recent_avg = np.mean(agent['coherence'][-k:])
            if recent_avg < theta:
                if np.random.rand() < epsilon:
                    agent['mode'] = np.random.choice([m for m in modes if m != mode])
                else:
                    viable_modes = {m: np.mean(v) for m, v in agent['memory'].items() if len(v) > 3}
                    if viable_modes:
                        agent['mode'] = max(viable_modes, key=viable_modes.get)

            if recent_avg < 0.1:
                agent['dropout'] = True
                agent['dropout_round'] = t
                dropout_timeline.append(t)

# --- Panel B: Dropout Frequency in First 100 Rounds ---
dropout_counts = Counter(dropout_timeline)
rounds = range(T)
dropout_per_round = [dropout_counts.get(r, 0) for r in rounds]

plt.figure(figsize=(8, 4))
plt.bar(range(100), dropout_per_round[:100], color='black', width=1.0)
plt.title("B. Dropout Timeline (First 100 Rounds)")
plt.xlabel("Rounds")
plt.ylabel("Agents Dropped")
plt.tight_layout()
plt.savefig("panel_B_dropout_first_100_rounds.png")
plt.close()

# --- Panel BB: Surviving Modes ---
#survivor_modes = [a['mode'] for a in agents if not a['dropout']]
#mode_counts = Counter(survivor_modes)

#plt.figure(figsize=(6, 4))
#plt.bar(mode_counts.keys(), mode_counts.values(), color="seagreen")
#plt.title("C. Surviving Agents by Final Mode")
#plt.ylabel("Agent Count")
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.savefig("panel_C_survivor_modes.png")
#plt.close()

# --- Panel C: Misalignment Pressure ---
misalignments = []
for agent in agents:
    if agent['dropout']:
        misalignments.append(0)
        continue
    penalties = []
    for t in range(T):
        if t >= len(agent['history']):
            continue
        m_i = agent['history'][t][0]
        p = np.random.choice([a for a in agents if a['id'] != agent['id']])
        m_j = p['mode']
        penalty = compatibility(m_i, m_j)
        penalties.append(penalty)
    misalignments.append(np.mean(penalties))

sorted_ids = np.argsort(misalignments)
sorted_vals = np.array(misalignments)[sorted_ids]
plt.figure(figsize=(8, 4))
plt.bar(range(len(sorted_vals)), sorted_vals, color="salmon")
plt.title("C. Mode Misalignment Pressure")
plt.xlabel("Agent ID (sorted)")
plt.ylabel("Avg. Interaction Penalty")
plt.tight_layout()
plt.savefig("panel_C_misalignment_pressure.png")
plt.close()

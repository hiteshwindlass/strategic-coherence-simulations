
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 17,            # Base font size
    "axes.titlesize": 19,       # Title font
    "axes.labelsize": 17,       # x/y label size
    "xtick.labelsize": 15,      # Tick label size
    "ytick.labelsize": 15,
    "legend.fontsize": 16,      # Legend text size
})

np.random.seed(42)  # for reproducibility

# Define the number of agents and steps
num_agents = 10
num_steps = 100

# Define resolution modes
modes = ["Contraction", "Postponement", "Oscillation", "Addition", "Submission", "Subsumption"]

# Assign each agent a mode (intentionally diversified)
assigned_modes = [
    "Submission", "Subsumption", "Contraction", "Addition", "Addition",
    "Addition", "Postponement", "Addition", "Subsumption", "Oscillation"
]

# Define how coherence evolves for each mode (with light noise)
def generate_coherence(mode, steps):
    base = np.zeros(steps)
    for t in range(steps):
        if mode == "Contraction":
            value = np.exp(-0.05 * t)
        elif mode == "Postponement":
            value = np.log1p(t) / np.log1p(steps)
        elif mode == "Oscillation":
            value = 0.5 * np.sin(0.4 * t) + 0.5
        elif mode == "Addition":
            value = min(1.0, 0.01 * (t ** 1.5))
        elif mode == "Submission":
            value = 1 - np.exp(-0.1 * t)
        elif mode == "Subsumption":
            value = 1 / (1 + np.exp(-0.1 * (t - 50)))
        noise = np.random.normal(0, 0.02)
        base[t] = np.clip(value + noise, 0, 1)
    return base

# Generate coherence trajectories
trajectories = [generate_coherence(mode, num_steps) for mode in assigned_modes]

# Plotting
fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
fig.suptitle("Coherence Trajectories of 10 Fixed-Mode Agents", fontsize=22)
for i, ax in enumerate(axes.flat):
    ax.plot(trajectories[i], color="blue")
    ax.set_title(f"Agent {i+1} ({assigned_modes[i]})", fontsize=16)
    ax.set_xlim(0, num_steps)
    ax.set_ylim(0, 1.05)
    if i >= 5:
        ax.set_xlabel("Rounds")
    if i % 5 == 0:
        ax.set_ylabel("Coherence")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("mixed_mode_agent_trajectories-panel.png")
plt.show()


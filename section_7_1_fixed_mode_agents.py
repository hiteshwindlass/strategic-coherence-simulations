import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,            # Base font size
    "axes.titlesize": 15,       # Title font
    "axes.labelsize": 13,       # x/y label size
    "xtick.labelsize": 11,      # Tick label size
    "ytick.labelsize": 11,
    "legend.fontsize": 12,      # Legend text size
})

# Simulation parameters
n_steps = 100
modes = ["Contraction", "Postponement", "Oscillation", "Addition", "Submission", "Subsumption"]
trajectories = {}

# Define coherence functions
for mode in modes:
    x = []
    for t in range(n_steps):
        if mode == "Contraction":
            value = np.exp(-0.05 * t)
        elif mode == "Postponement":
            value = np.log1p(t) / np.log1p(n_steps)
        elif mode == "Oscillation":
            value = 0.5 * np.sin(0.4 * t) + 0.5  # normalized sine wave [0,1]
        elif mode == "Addition":
            value = min(1.0, 0.01 * (t ** 1.5))  # sublinear growth
        elif mode == "Submission":
            value = 1 - np.exp(-0.1 * t)  # fast rise, saturating
        elif mode == "Subsumption":
            value = 1 / (1 + np.exp(-0.1 * (t - 50)))  # sigmoid centered
        x.append(value)
    trajectories[mode] = x

# Plot with distinct markers
plt.figure(figsize=(12, 7))
markers = {
    "Contraction": 'o',
    "Postponement": 's',
    "Oscillation": '^',
    "Addition": 'v',
    "Submission": 'D',
    "Subsumption": 'x'
}

for mode, values in trajectories.items():
    plt.plot(values, label=mode, marker=markers[mode], markevery=5)

plt.xlabel("Rounds")
plt.ylabel("Normalized Coherence Score")
plt.title("Coherence Evolution under Fixed Resolution Modes")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save output image
plt.savefig("fixed_mode_corrected.png")
plt.show()

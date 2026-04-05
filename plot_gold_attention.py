import json
import numpy as np
import matplotlib.pyplot as plt
import os

with open("results/q2/gold_attention_results.json") as f:
    results = json.load(f)

# Aggregate scores per gold_position
position_scores = {}
for res in results:
    pos = res["gold_position"]
    score = res["gold_score"]
    position_scores.setdefault(pos, []).append(score)

positions = sorted(position_scores.keys())
avg_scores = np.array([np.mean(position_scores[p]) for p in positions])
std_scores = np.array([np.std(position_scores[p]) for p in positions])
counts = np.array([len(position_scores[p]) for p in positions])
se_scores = std_scores / np.sqrt(counts)  # standard error for confidence band

os.makedirs("plot2", exist_ok=True)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(14, 6))

# Bar chart of average attention scores
bar_colors = plt.cm.RdYlGn(avg_scores / avg_scores.max())  # color by score magnitude
bars = ax1.bar(positions, avg_scores, color=bar_colors, edgecolor="gray",
               linewidth=0.4, alpha=0.85, zorder=2)

# Confidence band via error bars (±1 SE)
ax1.errorbar(positions, avg_scores, yerr=se_scores, fmt="none",
             ecolor="black", elinewidth=0.6, capsize=1.5, zorder=3)

# Smoothed trend line (moving average, window=7)
window = 7
kernel = np.ones(window) / window
smoothed = np.convolve(avg_scores, kernel, mode="same")
ax1.plot(positions, smoothed, color="darkred", linewidth=2, label="Trend (MA-7)", zorder=4)

# Secondary axis: sample count per position
ax2 = ax1.twinx()
ax2.plot(positions, counts, color="steelblue", linewidth=1, alpha=0.5,
         linestyle="--", label="Sample count")
ax2.set_ylabel("Number of Queries at Position", fontsize=11, color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")

# Labels and formatting
ax1.set_xlabel("Gold Tool Position (index in shuffled list)", fontsize=12)
ax1.set_ylabel("Average Attention Score (Query → Gold Tool)", fontsize=12)
ax1.set_title("Lost in the Middle? — Average Query→Gold-Tool Attention by Position",
              fontsize=14, fontweight="bold", pad=12)

# Show every 5th tick for readability
ax1.set_xticks([p for p in positions if p % 5 == 0])
ax1.tick_params(axis="x", labelsize=8)
ax1.set_xlim(-1, max(positions) + 1)
ax1.grid(axis="y", alpha=0.3, zorder=0)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

plt.tight_layout()
save_path = "plot2/gold_attention_plot.png"
plt.savefig(save_path, dpi=200)
plt.close()
print(f"Plot saved to {save_path}")

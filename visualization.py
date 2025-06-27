import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final_log_fixed.csv")
df = df[df['player_number'].str.startswith("Player")]

# Pivot table: players vs frames
timeline_data = df.pivot_table(index='player_number', columns='frame', aggfunc='size', fill_value=0)

plt.figure(figsize=(14, 6))
sns.heatmap(timeline_data, cmap='Blues', cbar_kws={'label': 'Presence'})
plt.title("⏱️ Player Presence Over Time (Frames)")
plt.xlabel("Frame Number")
plt.ylabel("Player Number")
plt.tight_layout()
plt.show()

player_duration = df.groupby('player_number')['frame'].nunique().sort_values(ascending=False)

plt.figure(figsize=(12, 5))
player_duration.plot(kind='bar', color='mediumseagreen')
plt.title("Player Tracking Consistency (Number of Frames Detected)")
plt.ylabel("Frames")
plt.xlabel("Player Number")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

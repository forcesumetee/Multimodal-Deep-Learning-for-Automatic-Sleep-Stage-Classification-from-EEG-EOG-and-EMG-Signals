#!/usr/bin/env python3
"""
Visualize Sleep-EDF Dataset
Creates visualizations of the multimodal signals and sleep stages
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
data_dir = Path('/home/ubuntu/sleep_classification/data')
psg_file = data_dir / 'SC4001E0-PSG.edf'
hypno_file = data_dir / 'SC4001EC-Hypnogram.edf'
fig_dir = Path('/home/ubuntu/sleep_classification/figures')

print("Creating visualizations...")

# Load data
raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
annot = mne.read_annotations(hypno_file)

# 1. Sleep stage distribution pie chart
print("1. Creating sleep stage distribution chart...")
stage_counts = {}
for desc in annot.description:
    stage_counts[desc] = stage_counts.get(desc, 0) + 1

stage_mapping = {
    'Sleep stage W': 'Wake',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2', 
    'Sleep stage 3': 'N3',
    'Sleep stage 4': 'N4',
    'Sleep stage R': 'REM',
    'Sleep stage ?': 'Unknown'
}

# Prepare data for plotting
labels = []
sizes = []
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', '#ffb3e6']

for stage, count in sorted(stage_counts.items()):
    if 'Movement' not in stage:  # Exclude movement time
        labels.append(stage_mapping.get(stage, stage))
        sizes.append(count)

fig, ax = plt.subplots(figsize=(10, 7))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(labels)],
                                    autopct='%1.1f%%', startangle=90)
ax.set_title('Sleep Stage Distribution in Sleep-EDF Dataset', fontsize=14, fontweight='bold')

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

plt.tight_layout()
plt.savefig(fig_dir / 'sleep_stage_distribution.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'sleep_stage_distribution.png'}")
plt.close()

# 2. Multimodal signal visualization (30-second epoch)
print("2. Creating multimodal signal visualization...")

# Extract a 30-second segment (typical epoch length)
start_time = 3600  # Start at 1 hour into recording
duration = 30  # 30 seconds
start_sample = int(start_time * raw.info['sfreq'])
end_sample = int((start_time + duration) * raw.info['sfreq'])

# Get EEG, EOG, EMG channels
eeg_fpz_cz = raw.get_data(picks='EEG Fpz-Cz')[0, start_sample:end_sample]
eeg_pz_oz = raw.get_data(picks='EEG Pz-Oz')[0, start_sample:end_sample]
eog = raw.get_data(picks='EOG horizontal')[0, start_sample:end_sample]
emg = raw.get_data(picks='EMG submental')[0, start_sample:end_sample]

time_axis = np.linspace(0, duration, len(eeg_fpz_cz))

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# EEG Fpz-Cz
axes[0].plot(time_axis, eeg_fpz_cz * 1e6, color='#2E86AB', linewidth=0.8)
axes[0].set_ylabel('EEG Fpz-Cz\n(µV)', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Multimodal Polysomnography Signals (30-second epoch)', 
                  fontsize=14, fontweight='bold', pad=15)

# EEG Pz-Oz
axes[1].plot(time_axis, eeg_pz_oz * 1e6, color='#A23B72', linewidth=0.8)
axes[1].set_ylabel('EEG Pz-Oz\n(µV)', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# EOG
axes[2].plot(time_axis, eog * 1e6, color='#F18F01', linewidth=0.8)
axes[2].set_ylabel('EOG\n(µV)', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)

# EMG
axes[3].plot(time_axis, emg * 1e6, color='#C73E1D', linewidth=0.8)
axes[3].set_ylabel('EMG\n(µV)', fontsize=11, fontweight='bold')
axes[3].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'multimodal_signals.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'multimodal_signals.png'}")
plt.close()

# 3. Hypnogram visualization
print("3. Creating hypnogram visualization...")

# Convert annotations to hypnogram array
stage_to_num = {
    'Sleep stage W': 0,
    'Sleep stage R': 1,
    'Sleep stage 1': 2,
    'Sleep stage 2': 3,
    'Sleep stage 3': 4,
    'Sleep stage 4': 5,
    'Sleep stage ?': -1
}

hypnogram = []
time_points = []

for idx, (onset, duration, desc) in enumerate(zip(annot.onset, annot.duration, annot.description)):
    if desc in stage_to_num:
        hypnogram.append(stage_to_num[desc])
        time_points.append(onset / 3600)  # Convert to hours

fig, ax = plt.subplots(figsize=(14, 6))

# Plot hypnogram as step function
for i in range(len(hypnogram) - 1):
    if hypnogram[i] >= 0:  # Skip unknown stages
        ax.hlines(hypnogram[i], time_points[i], time_points[i+1], 
                 colors='#2E86AB', linewidth=2)
        ax.vlines(time_points[i+1], hypnogram[i], hypnogram[i+1], 
                 colors='#2E86AB', linewidth=2)

ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
ax.set_title('Hypnogram: Sleep Stage Progression Over Night', 
             fontsize=14, fontweight='bold')

# Set y-axis labels
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels(['Wake', 'REM', 'N1', 'N2', 'N3', 'N4'])
ax.invert_yaxis()  # Invert so Wake is at top
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, max(time_points))

plt.tight_layout()
plt.savefig(fig_dir / 'hypnogram.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'hypnogram.png'}")
plt.close()

print("\nAll visualizations created successfully!")


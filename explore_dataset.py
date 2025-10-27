#!/usr/bin/env python3
"""
Explore Sleep-EDF Dataset
This script loads and analyzes the Sleep-EDF dataset structure
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
data_dir = Path('/home/ubuntu/sleep_classification/data')
psg_file = data_dir / 'SC4001E0-PSG.edf'
hypno_file = data_dir / 'SC4001EC-Hypnogram.edf'

print("=" * 80)
print("Sleep-EDF Dataset Exploration")
print("=" * 80)

# Load PSG data
print("\n1. Loading PSG (Polysomnography) data...")
raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)

print(f"   Duration: {raw.times[-1] / 3600:.2f} hours")
print(f"   Sampling frequency: {raw.info['sfreq']} Hz")
print(f"   Number of channels: {len(raw.ch_names)}")
print(f"   Channel names: {raw.ch_names}")

# Get channel information
print("\n2. Channel Details:")
for idx, ch_name in enumerate(raw.ch_names):
    ch_type = raw.get_channel_types()[idx]
    print(f"   - {ch_name}: {ch_type}")

# Load hypnogram (sleep stage annotations)
print("\n3. Loading Hypnogram (Sleep Stage Annotations)...")
annot = mne.read_annotations(hypno_file)

print(f"   Total annotations: {len(annot)}")
print(f"   Annotation types: {set(annot.description)}")

# Count sleep stages
stage_counts = {}
for desc in annot.description:
    stage_counts[desc] = stage_counts.get(desc, 0) + 1

print("\n4. Sleep Stage Distribution:")
stage_mapping = {
    'Sleep stage W': 'Wake',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2', 
    'Sleep stage 3': 'N3',
    'Sleep stage 4': 'N4',
    'Sleep stage R': 'REM',
    'Sleep stage ?': 'Unknown',
    'Movement time': 'Movement'
}

total_epochs = sum(stage_counts.values())
for stage, count in sorted(stage_counts.items()):
    stage_name = stage_mapping.get(stage, stage)
    percentage = (count / total_epochs) * 100
    print(f"   {stage_name:12s}: {count:4d} epochs ({percentage:5.1f}%)")

print(f"\n   Total epochs: {total_epochs}")

# Extract signal statistics
print("\n5. Signal Statistics:")
data, times = raw[:, :]
for idx, ch_name in enumerate(raw.ch_names):
    ch_data = data[idx]
    print(f"   {ch_name}:")
    print(f"      Mean: {np.mean(ch_data):.2f} µV")
    print(f"      Std:  {np.std(ch_data):.2f} µV")
    print(f"      Min:  {np.min(ch_data):.2f} µV")
    print(f"      Max:  {np.max(ch_data):.2f} µV")

# Identify EEG, EOG, EMG channels
print("\n6. Modality Classification:")
eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch or 'Fpz' in ch or 'Pz' in ch]
eog_channels = [ch for ch in raw.ch_names if 'EOG' in ch]
emg_channels = [ch for ch in raw.ch_names if 'EMG' in ch]

print(f"   EEG channels: {eeg_channels}")
print(f"   EOG channels: {eog_channels}")
print(f"   EMG channels: {emg_channels}")

# Save summary
print("\n7. Saving dataset summary...")
summary_file = Path('/home/ubuntu/sleep_classification/dataset_summary.txt')
with open(summary_file, 'w') as f:
    f.write("Sleep-EDF Dataset Summary\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"PSG File: {psg_file.name}\n")
    f.write(f"Hypnogram File: {hypno_file.name}\n\n")
    f.write(f"Duration: {raw.times[-1] / 3600:.2f} hours\n")
    f.write(f"Sampling Frequency: {raw.info['sfreq']} Hz\n")
    f.write(f"Number of Channels: {len(raw.ch_names)}\n\n")
    
    f.write("Channels:\n")
    for ch_name in raw.ch_names:
        f.write(f"  - {ch_name}\n")
    
    f.write("\nSleep Stage Distribution:\n")
    for stage, count in sorted(stage_counts.items()):
        stage_name = stage_mapping.get(stage, stage)
        percentage = (count / total_epochs) * 100
        f.write(f"  {stage_name:12s}: {count:4d} epochs ({percentage:5.1f}%)\n")
    
    f.write(f"\nTotal Epochs: {total_epochs}\n")
    f.write(f"\nEEG Channels: {', '.join(eeg_channels)}\n")
    f.write(f"EOG Channels: {', '.join(eog_channels)}\n")
    f.write(f"EMG Channels: {', '.join(emg_channels)}\n")

print(f"   Summary saved to: {summary_file}")

print("\n" + "=" * 80)
print("Dataset exploration complete!")
print("=" * 80)


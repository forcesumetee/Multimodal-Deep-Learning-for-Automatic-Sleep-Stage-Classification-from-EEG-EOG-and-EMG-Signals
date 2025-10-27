#!/usr/bin/env python3
"""
Standalone Evaluation Script
Loads data, trains if needed, and evaluates the model
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            cohen_kappa_score, f1_score, accuracy_score)
import tensorflow as tf
from tensorflow import keras
import pickle
import sys
sys.path.append('/home/ubuntu/sleep_classification')
from model import MultimodalSleepNet

# Configuration
EPOCH_LENGTH_SEC = 30
SAMPLING_FREQ = 100
EPOCH_LENGTH_SAMPLES = EPOCH_LENGTH_SEC * SAMPLING_FREQ
NUM_CLASSES = 5

STAGE_MAPPING = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

# Paths
data_dir = Path('/home/ubuntu/sleep_classification/data')
model_dir = Path('/home/ubuntu/sleep_classification/models')
results_dir = Path('/home/ubuntu/sleep_classification/results')
fig_dir = Path('/home/ubuntu/sleep_classification/figures')

print("=" * 80)
print("Multimodal Sleep Stage Classification - Complete Evaluation")
print("=" * 80)

# Load and preprocess data
print("\n1. Loading data...")
psg_file = data_dir / 'SC4001E0-PSG.edf'
hypno_file = data_dir / 'SC4001EC-Hypnogram.edf'

raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
annot = mne.read_annotations(hypno_file)

eeg_fpz_cz = raw.get_data(picks='EEG Fpz-Cz')[0]
eeg_pz_oz = raw.get_data(picks='EEG Pz-Oz')[0]
eog = raw.get_data(picks='EOG horizontal')[0]
emg = raw.get_data(picks='EMG submental')[0]

# Create epochs
print("\n2. Creating epochs...")
eeg_epochs, eog_epochs, emg_epochs, labels = [], [], [], []

for onset, duration, description in zip(annot.onset, annot.duration, annot.description):
    if description in STAGE_MAPPING:
        start_sample = int(onset * SAMPLING_FREQ)
        end_sample = start_sample + EPOCH_LENGTH_SAMPLES
        
        if end_sample <= len(eeg_fpz_cz):
            eeg_epoch = np.stack([
                eeg_fpz_cz[start_sample:end_sample],
                eeg_pz_oz[start_sample:end_sample]
            ], axis=-1)
            
            eog_epoch = eog[start_sample:end_sample].reshape(-1, 1)
            emg_epoch = emg[start_sample:end_sample].reshape(-1, 1)
            
            eeg_epochs.append(eeg_epoch)
            eog_epochs.append(eog_epoch)
            emg_epochs.append(emg_epoch)
            labels.append(STAGE_MAPPING[description])

eeg_epochs = np.array(eeg_epochs)
eog_epochs = np.array(eog_epochs)
emg_epochs = np.array(emg_epochs)
labels = np.array(labels)

print(f"   Total epochs: {len(labels)}")

# Normalize
print("\n3. Normalizing data...")
with open(model_dir / 'scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

eeg_flat = eeg_epochs.reshape(-1, 2)
eog_flat = eog_epochs.reshape(-1, 1)
emg_flat = emg_epochs.reshape(-1, 1)

eeg_flat_scaled = scalers['eeg_scaler'].transform(eeg_flat)
eog_flat_scaled = scalers['eog_scaler'].transform(eog_flat)
emg_flat_scaled = scalers['emg_scaler'].transform(emg_flat)

eeg_epochs_scaled = eeg_flat_scaled.reshape(eeg_epochs.shape)
eog_epochs_scaled = eog_flat_scaled.reshape(eog_epochs.shape)
emg_epochs_scaled = emg_flat_scaled.reshape(emg_epochs.shape)

# Split data (same split as training)
np.random.seed(42)
tf.random.set_seed(42)

labels_categorical = keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

X_eeg_temp, X_eeg_test, X_eog_temp, X_eog_test, X_emg_temp, X_emg_test, y_temp, y_test = train_test_split(
    eeg_epochs_scaled, eog_epochs_scaled, emg_epochs_scaled, labels_categorical,
    test_size=0.2, random_state=42, stratify=labels
)

print(f"   Test set size: {len(y_test)}")

# Load model
print("\n4. Loading trained model...")
model = keras.models.load_model(
    model_dir / 'best_model.keras',
    custom_objects={'MultimodalSleepNet': MultimodalSleepNet}
)

# Make predictions
print("\n5. Making predictions...")
y_pred_proba = model.predict([X_eeg_test, X_eog_test, X_emg_test], verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate metrics
print("\n6. Performance Metrics:")
print("=" * 80)

accuracy = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")

print("\n" + "=" * 80)
print("Per-Class Performance:")
print("=" * 80)

report = classification_report(y_true, y_pred, target_names=STAGE_NAMES, 
                              zero_division=0, output_dict=True)

for stage in STAGE_NAMES:
    metrics = report[stage]
    print(f"{stage:8s}: Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
          f"Support={int(metrics['support'])}")

# Confusion Matrix
print("\n7. Creating visualizations...")
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Sleep Stage', fontsize=12, fontweight='bold')
ax.set_ylabel('True Sleep Stage', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Multimodal Sleep Stage Classification', 
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(fig_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'confusion_matrix.png'}")
plt.close()

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
ax.set_xlabel('Predicted Sleep Stage', fontsize=12, fontweight='bold')
ax.set_ylabel('True Sleep Stage', fontsize=12, fontweight='bold')
ax.set_title('Normalized Confusion Matrix', 
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(fig_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'confusion_matrix_normalized.png'}")
plt.close()

# Per-class performance
stages = STAGE_NAMES
precision_scores = [report[stage]['precision'] for stage in stages]
recall_scores = [report[stage]['recall'] for stage in stages]
f1_scores = [report[stage]['f1-score'] for stage in stages]

x = np.arange(len(stages))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, precision_scores, width, label='Precision', color='#2E86AB')
bars2 = ax.bar(x, recall_scores, width, label='Recall', color='#A23B72')
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#F18F01')

ax.set_xlabel('Sleep Stage', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(fig_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'per_class_performance.png'}")
plt.close()

# Save report
print("\n8. Saving evaluation report...")
with open(results_dir / 'evaluation_report.txt', 'w') as f:
    f.write("Multimodal Sleep Stage Classification - Evaluation Report\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Test Samples: {len(y_test)}\n\n")
    
    f.write("Overall Performance:\n")
    f.write(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"  Cohen's Kappa: {kappa:.4f}\n")
    f.write(f"  F1-Score (Macro): {f1_macro:.4f}\n")
    f.write(f"  F1-Score (Weighted): {f1_weighted:.4f}\n\n")
    
    f.write("Per-Class Performance:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Stage':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
    f.write("-" * 80 + "\n")
    
    for stage in STAGE_NAMES:
        metrics = report[stage]
        f.write(f"{stage:<10} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} "
               f"{metrics['f1-score']:<12.3f} {int(metrics['support']):<10}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("\nDetailed Classification Report:\n")
    f.write(classification_report(y_true, y_pred, target_names=STAGE_NAMES, zero_division=0))
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("\nConfusion Matrix:\n")
    f.write(f"{'':>10}")
    for stage in STAGE_NAMES:
        f.write(f"{stage:>10}")
    f.write("\n")
    
    for i, stage in enumerate(STAGE_NAMES):
        f.write(f"{stage:>10}")
        for j in range(len(STAGE_NAMES)):
            f.write(f"{cm[i, j]:>10}")
        f.write("\n")

print(f"   Report saved to: {results_dir / 'evaluation_report.txt'}")

print("\n" + "=" * 80)
print("Evaluation Complete!")
print("=" * 80)


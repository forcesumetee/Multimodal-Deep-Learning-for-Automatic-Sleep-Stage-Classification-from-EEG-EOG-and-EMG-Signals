#!/usr/bin/env python3
"""
Analyze Training Results and Create Visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path('/home/ubuntu/sleep_classification/results')
fig_dir = Path('/home/ubuntu/sleep_classification/figures')

print("=" * 80)
print("Training Results Analysis")
print("=" * 80)

# Load training history
print("\n1. Loading training history...")
history = pd.read_csv(results_dir / 'training_history.csv')

print(f"   Total epochs trained: {len(history)}")
print(f"   Best validation accuracy: {history['val_accuracy'].max():.4f} at epoch {history['val_accuracy'].idxmax() + 1}")

# Create training curves
print("\n2. Creating training curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(history['epoch'], history['accuracy'], label='Training', linewidth=2, color='#2E86AB')
axes[0, 0].plot(history['epoch'], history['val_accuracy'], label='Validation', linewidth=2, color='#A23B72')
axes[0, 0].set_xlabel('Epoch', fontweight='bold')
axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
axes[0, 0].set_title('Model Accuracy', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history['epoch'], history['loss'], label='Training', linewidth=2, color='#2E86AB')
axes[0, 1].plot(history['epoch'], history['val_loss'], label='Validation', linewidth=2, color='#A23B72')
axes[0, 1].set_xlabel('Epoch', fontweight='bold')
axes[0, 1].set_ylabel('Loss', fontweight='bold')
axes[0, 1].set_title('Model Loss', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history['epoch'], history['precision'], label='Training', linewidth=2, color='#2E86AB')
axes[1, 0].plot(history['epoch'], history['val_precision'], label='Validation', linewidth=2, color='#A23B72')
axes[1, 0].set_xlabel('Epoch', fontweight='bold')
axes[1, 0].set_ylabel('Precision', fontweight='bold')
axes[1, 0].set_title('Model Precision', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history['epoch'], history['recall'], label='Training', linewidth=2, color='#2E86AB')
axes[1, 1].plot(history['epoch'], history['val_recall'], label='Validation', linewidth=2, color='#A23B72')
axes[1, 1].set_xlabel('Epoch', fontweight='bold')
axes[1, 1].set_ylabel('Recall', fontweight='bold')
axes[1, 1].set_title('Model Recall', fontweight='bold', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Multimodal Sleep Stage Classification - Training History', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(fig_dir / 'training_history.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'training_history.png'}")
plt.close()

# Create summary report
print("\n3. Creating summary report...")

best_epoch = history['val_accuracy'].idxmax()
best_metrics = history.iloc[best_epoch]

summary = f"""
Multimodal Sleep Stage Classification - Training Summary
{"=" * 80}

Dataset: Sleep-EDF Database Expanded
Model Architecture: Multi-stream CNN with Attention-based Fusion
Modalities: EEG (2 channels), EOG (1 channel), EMG (1 channel)

Training Configuration:
- Total epochs: {len(history)}
- Best epoch: {best_epoch + 1}
- Early stopping: Yes (patience=15)
- Learning rate reduction: Yes (factor=0.5, patience=5)

Best Validation Performance (Epoch {best_epoch + 1}):
- Accuracy: {best_metrics['val_accuracy']:.4f} ({best_metrics['val_accuracy']*100:.2f}%)
- Loss: {best_metrics['val_loss']:.4f}
- Precision: {best_metrics['val_precision']:.4f}
- Recall: {best_metrics['val_recall']:.4f}

Final Training Performance:
- Accuracy: {history['accuracy'].iloc[-1]:.4f} ({history['accuracy'].iloc[-1]*100:.2f}%)
- Loss: {history['loss'].iloc[-1]:.4f}
- Precision: {history['precision'].iloc[-1]:.4f}
- Recall: {history['recall'].iloc[-1]:.4f}

Model Characteristics:
- Total parameters: 476,171
- Trainable parameters: 473,355
- Non-trainable parameters: 2,816

Architecture Highlights:
- Separate CNN streams for each modality (EEG, EOG, EMG)
- Attention mechanism for adaptive modality fusion
- Batch normalization and dropout for regularization
- Global average pooling for feature aggregation

{"=" * 80}
"""

with open(results_dir / 'training_summary.txt', 'w') as f:
    f.write(summary)

print(summary)
print(f"\n   Summary saved to: {results_dir / 'training_summary.txt'}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)


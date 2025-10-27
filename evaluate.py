#!/usr/bin/env python3
"""
Evaluation Script for Multimodal Sleep Stage Classification
Generates detailed performance metrics and confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix, 
                            cohen_kappa_score, f1_score, accuracy_score)
import tensorflow as tf
from tensorflow import keras

# Configuration
STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
results_dir = Path('/home/ubuntu/sleep_classification/results')
model_dir = Path('/home/ubuntu/sleep_classification/models')
fig_dir = Path('/home/ubuntu/sleep_classification/figures')

print("=" * 80)
print("Multimodal Sleep Stage Classification - Evaluation")
print("=" * 80)

# Load test data
print("\n1. Loading test data...")
test_data = np.load(results_dir / 'test_data.npz')
X_eeg_test = test_data['X_eeg_test']
X_eog_test = test_data['X_eog_test']
X_emg_test = test_data['X_emg_test']
y_test = test_data['y_test']

print(f"   Test samples: {len(y_test)}")

# Load model
print("\n2. Loading trained model...")
model = keras.models.load_model(model_dir / 'best_model.keras')

# Make predictions
print("\n3. Making predictions...")
y_pred_proba = model.predict([X_eeg_test, X_eog_test, X_emg_test], verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate metrics
print("\n4. Computing performance metrics...")

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"   Overall Accuracy: {accuracy:.4f}")

# Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred)
print(f"   Cohen's Kappa: {kappa:.4f}")

# F1 scores
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
print(f"   F1-Score (Macro): {f1_macro:.4f}")
print(f"   F1-Score (Weighted): {f1_weighted:.4f}")

# Per-class metrics
print("\n5. Per-class Performance:")
report = classification_report(y_true, y_pred, target_names=STAGE_NAMES, 
                              zero_division=0, output_dict=True)

for stage in STAGE_NAMES:
    metrics = report[stage]
    print(f"   {stage:8s}: Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
          f"Support={int(metrics['support'])}")

# Confusion matrix
print("\n6. Creating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Sleep Stage', fontsize=12, fontweight='bold')
ax.set_ylabel('True Sleep Stage', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Sleep Stage Classification', 
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(fig_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'confusion_matrix.png'}")
plt.close()

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
            cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
ax.set_xlabel('Predicted Sleep Stage', fontsize=12, fontweight='bold')
ax.set_ylabel('True Sleep Stage', fontsize=12, fontweight='bold')
ax.set_title('Normalized Confusion Matrix - Sleep Stage Classification', 
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(fig_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {fig_dir / 'confusion_matrix_normalized.png'}")
plt.close()

# Per-class performance bar chart
print("\n7. Creating performance visualization...")
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

# Save detailed report
print("\n8. Saving evaluation report...")
with open(results_dir / 'evaluation_report.txt', 'w') as f:
    f.write("Multimodal Sleep Stage Classification - Evaluation Report\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Test Samples: {len(y_test)}\n\n")
    
    f.write("Overall Performance:\n")
    f.write(f"  Accuracy: {accuracy:.4f}\n")
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
print("Evaluation complete!")
print("=" * 80)

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Overall Accuracy: {accuracy:.2%}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Macro F1-Score: {f1_macro:.4f}")
print("=" * 80)


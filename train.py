#!/usr/bin/env python3
"""
Training Script for Multimodal Sleep Stage Classification
"""

import mne
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from model import build_model
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("Multimodal Sleep Stage Classification - Training")
print("=" * 80)

# Configuration
EPOCH_LENGTH_SEC = 30
SAMPLING_FREQ = 100
EPOCH_LENGTH_SAMPLES = EPOCH_LENGTH_SEC * SAMPLING_FREQ
NUM_CLASSES = 5

# Stage mapping
STAGE_MAPPING = {
    'Sleep stage W': 0,   # Wake
    'Sleep stage 1': 1,   # N1
    'Sleep stage 2': 2,   # N2
    'Sleep stage 3': 3,   # N3 (combine N3 and N4)
    'Sleep stage 4': 3,   # N4 -> N3
    'Sleep stage R': 4,   # REM
}

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

# Paths
data_dir = Path('/home/ubuntu/sleep_classification/data')
model_dir = Path('/home/ubuntu/sleep_classification/models')
results_dir = Path('/home/ubuntu/sleep_classification/results')

print("\n1. Loading and preprocessing data...")

# Load PSG data
psg_file = data_dir / 'SC4001E0-PSG.edf'
hypno_file = data_dir / 'SC4001EC-Hypnogram.edf'

raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
annot = mne.read_annotations(hypno_file)

# Extract channels
eeg_fpz_cz = raw.get_data(picks='EEG Fpz-Cz')[0]
eeg_pz_oz = raw.get_data(picks='EEG Pz-Oz')[0]
eog = raw.get_data(picks='EOG horizontal')[0]
emg = raw.get_data(picks='EMG submental')[0]

print(f"   Total recording duration: {len(eeg_fpz_cz) / SAMPLING_FREQ / 3600:.2f} hours")

# Create epochs based on annotations
print("\n2. Creating epochs from annotations...")

eeg_epochs = []
eog_epochs = []
emg_epochs = []
labels = []

for onset, duration, description in zip(annot.onset, annot.duration, annot.description):
    if description in STAGE_MAPPING:
        # Convert onset time to sample index
        start_sample = int(onset * SAMPLING_FREQ)
        end_sample = start_sample + EPOCH_LENGTH_SAMPLES
        
        # Check if epoch is within recording bounds
        if end_sample <= len(eeg_fpz_cz):
            # Extract epoch for each modality
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

# Convert to numpy arrays
eeg_epochs = np.array(eeg_epochs)
eog_epochs = np.array(eog_epochs)
emg_epochs = np.array(emg_epochs)
labels = np.array(labels)

print(f"   Total epochs created: {len(labels)}")
print(f"   EEG shape: {eeg_epochs.shape}")
print(f"   EOG shape: {eog_epochs.shape}")
print(f"   EMG shape: {emg_epochs.shape}")

# Print class distribution
print("\n3. Class distribution:")
for i, stage_name in enumerate(STAGE_NAMES):
    count = np.sum(labels == i)
    percentage = (count / len(labels)) * 100
    print(f"   {stage_name:8s}: {count:4d} epochs ({percentage:5.1f}%)")

# Normalize data
print("\n4. Normalizing data...")

# Reshape for normalization
eeg_flat = eeg_epochs.reshape(-1, 2)
eog_flat = eog_epochs.reshape(-1, 1)
emg_flat = emg_epochs.reshape(-1, 1)

# Fit scalers
eeg_scaler = StandardScaler()
eog_scaler = StandardScaler()
emg_scaler = StandardScaler()

eeg_flat_scaled = eeg_scaler.fit_transform(eeg_flat)
eog_flat_scaled = eog_scaler.fit_transform(eog_flat)
emg_flat_scaled = emg_scaler.fit_transform(emg_flat)

# Reshape back
eeg_epochs_scaled = eeg_flat_scaled.reshape(eeg_epochs.shape)
eog_epochs_scaled = eog_flat_scaled.reshape(eog_epochs.shape)
emg_epochs_scaled = emg_flat_scaled.reshape(emg_epochs.shape)

# Save scalers
with open(model_dir / 'scalers.pkl', 'wb') as f:
    pickle.dump({
        'eeg_scaler': eeg_scaler,
        'eog_scaler': eog_scaler,
        'emg_scaler': emg_scaler
    }, f)

print("   Scalers saved.")

# Convert labels to categorical
labels_categorical = keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

# Split data
print("\n5. Splitting data into train/validation/test sets...")

# First split: separate test set (20%)
X_eeg_temp, X_eeg_test, X_eog_temp, X_eog_test, X_emg_temp, X_emg_test, y_temp, y_test = train_test_split(
    eeg_epochs_scaled, eog_epochs_scaled, emg_epochs_scaled, labels_categorical,
    test_size=0.2, random_state=42, stratify=labels
)

# Second split: separate train and validation (80% train, 20% val of remaining)
X_eeg_train, X_eeg_val, X_eog_train, X_eog_val, X_emg_train, X_emg_val, y_train, y_val = train_test_split(
    X_eeg_temp, X_eog_temp, X_emg_temp, y_temp,
    test_size=0.25, random_state=42, stratify=np.argmax(y_temp, axis=1)
)

print(f"   Training set: {len(y_train)} epochs")
print(f"   Validation set: {len(y_val)} epochs")
print(f"   Test set: {len(y_test)} epochs")

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("\n6. Class weights for imbalanced data:")
for i, stage_name in enumerate(STAGE_NAMES):
    print(f"   {stage_name:8s}: {class_weight_dict[i]:.3f}")

# Build model
print("\n7. Building model...")
model = build_model(num_classes=NUM_CLASSES, epoch_length=EPOCH_LENGTH_SAMPLES)

print("\nModel Summary:")
model.summary()

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(model_dir / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        str(results_dir / 'training_history.csv')
    )
]

# Train model
print("\n8. Training model...")
print("   This may take a while...\n")

history = model.fit(
    [X_eeg_train, X_eog_train, X_emg_train],
    y_train,
    validation_data=([X_eeg_val, X_eog_val, X_emg_val], y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save(model_dir / 'final_model.keras')
print(f"\n   Final model saved to: {model_dir / 'final_model.keras'}")

# Evaluate on test set
print("\n9. Evaluating on test set...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(
    [X_eeg_test, X_eog_test, X_emg_test],
    y_test,
    verbose=0
)

print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_acc:.4f}")
print(f"   Test Precision: {test_precision:.4f}")
print(f"   Test Recall: {test_recall:.4f}")

# Calculate F1 score
if (test_precision + test_recall) > 0:
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
else:
    test_f1 = 0.0
print(f"   Test F1-Score: {test_f1:.4f}")

# Save test data for later evaluation
np.savez(
    results_dir / 'test_data.npz',
    X_eeg_test=X_eeg_test,
    X_eog_test=X_eog_test,
    X_emg_test=X_emg_test,
    y_test=y_test
)

print(f"\n   Test data saved to: {results_dir / 'test_data.npz'}")

# Save training summary
with open(results_dir / 'training_summary.txt', 'w') as f:
    f.write("Multimodal Sleep Stage Classification - Training Summary\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset: Sleep-EDF\n")
    f.write(f"Total epochs: {len(labels)}\n")
    f.write(f"Training epochs: {len(y_train)}\n")
    f.write(f"Validation epochs: {len(y_val)}\n")
    f.write(f"Test epochs: {len(y_test)}\n\n")
    f.write(f"Model parameters: {model.count_params():,}\n\n")
    f.write(f"Test Results:\n")
    f.write(f"  Accuracy: {test_acc:.4f}\n")
    f.write(f"  Precision: {test_precision:.4f}\n")
    f.write(f"  Recall: {test_recall:.4f}\n")
    f.write(f"  F1-Score: {test_f1:.4f}\n")

print("\n" + "=" * 80)
print("Training complete!")
print("=" * 80)


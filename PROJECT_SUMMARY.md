# Multimodal AI for Sleep Stage Classification - Project Summary

## Overview
This project implements a multimodal deep learning model for automatic sleep stage classification using EEG, EOG, and EMG signals from the Sleep-EDF database.

## Project Structure

```
sleep_classification/
├── data/                          # Dataset files
│   ├── SC4001E0-PSG.edf          # PSG recording
│   └── SC4001EC-Hypnogram.edf    # Sleep stage annotations
├── models/                        # Trained models
│   ├── best_model.keras          # Best model checkpoint
│   ├── final_model.keras         # Final trained model
│   └── scalers.pkl               # Data normalization scalers
├── figures/                       # Visualizations
│   ├── sleep_stage_distribution.png
│   ├── multimodal_signals.png
│   ├── hypnogram.png
│   ├── model_architecture.png
│   └── training_history.png
├── results/                       # Training results
│   ├── training_history.csv
│   ├── training_summary.txt
│   └── training_log.txt
├── model.py                       # Model architecture definition
├── train.py                       # Training script
├── explore_dataset.py             # Dataset exploration
├── visualize_data.py              # Data visualization
├── analyze_results.py             # Results analysis
└── conference_paper.md            # Final research paper

```

## Key Components

### 1. Model Architecture (MultimodalSleepNet)
- **Modality-Specific Streams**: Separate CNN streams for EEG, EOG, and EMG
- **Attention Mechanism**: Adaptive fusion of multimodal features
- **Classification Head**: Final dense layers for 5-class sleep stage prediction
- **Total Parameters**: 476,171 (473,355 trainable)

### 2. Dataset
- **Source**: Sleep-EDF Database Expanded (PhysioNet)
- **Signals**: EEG (2 channels), EOG (1 channel), EMG (1 channel)
- **Sampling Rate**: 100 Hz
- **Epoch Length**: 30 seconds (3000 samples)
- **Sleep Stages**: Wake, N1, N2, N3, REM

### 3. Training Results
- **Best Validation Accuracy**: 74.19% (Epoch 9)
- **Training Accuracy**: 97.80% (Epoch 30)
- **Total Epochs**: 30 (early stopping)
- **Optimization**: Adam optimizer with learning rate reduction

## Files Description

### Python Scripts
- `model.py`: Defines the MultimodalSleepNet architecture
- `train.py`: Handles data preprocessing and model training
- `explore_dataset.py`: Analyzes dataset structure and statistics
- `visualize_data.py`: Creates visualizations of signals and sleep stages
- `analyze_results.py`: Generates training performance plots

### Documentation
- `conference_paper.md`: Complete research paper in Markdown format
- `PROJECT_SUMMARY.md`: This file
- `dataset_summary.txt`: Dataset statistics and information

## How to Use

### 1. Explore the Dataset
```bash
python3 explore_dataset.py
python3 visualize_data.py
```

### 2. Train the Model
```bash
python3 train.py
```

### 3. Analyze Results
```bash
python3 analyze_results.py
```

## Key Findings

1. **Multimodal Fusion**: The attention-based fusion mechanism allows the model to adaptively weight different signal modalities.

2. **Overfitting Challenge**: Significant overfitting observed due to limited training data (single subject). Future work should use the full Sleep-EDFx dataset.

3. **Clinical Relevance**: The model demonstrates the feasibility of automated sleep staging using multimodal physiological signals.

## Future Directions

1. Train on the complete Sleep-EDFx dataset (197 subjects)
2. Implement advanced regularization techniques
3. Add explainability analysis for attention weights
4. Explore Transformer-based architectures
5. Conduct cross-dataset validation

## References

- Sleep-EDF Database: https://www.physionet.org/content/sleep-edfx/1.0.0/
- PhysioNet: https://www.physionet.org/

## Author
Sumetee Jirapattarasakul

## Date
October 27, 2025

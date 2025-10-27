#!/usr/bin/env python3
"""
Multimodal Sleep Stage Classification Model
Architecture: Multi-stream CNN with attention-based fusion
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

class MultimodalSleepNet(Model):
    """
    Multimodal Deep Learning Model for Sleep Stage Classification
    
    Architecture:
    - Separate CNN streams for each modality (EEG, EOG, EMG)
    - Attention mechanism for modality fusion
    - Classification head for 5-stage sleep classification
    
    Input: 30-second epochs of EEG (2 channels), EOG (1 channel), EMG (1 channel)
    Output: Sleep stage (Wake, N1, N2, N3, REM)
    """
    
    def __init__(self, num_classes=5, epoch_length=3000):
        """
        Args:
            num_classes: Number of sleep stages (default: 5 for W, N1, N2, N3, REM)
            epoch_length: Length of each epoch in samples (default: 3000 for 30s at 100Hz)
        """
        super(MultimodalSleepNet, self).__init__()
        
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        
        # EEG Stream (2 channels: Fpz-Cz, Pz-Oz)
        self.eeg_conv1 = layers.Conv1D(64, kernel_size=50, strides=6, padding='same', activation='relu')
        self.eeg_bn1 = layers.BatchNormalization()
        self.eeg_pool1 = layers.MaxPooling1D(pool_size=8, strides=8)
        self.eeg_dropout1 = layers.Dropout(0.3)
        
        self.eeg_conv2 = layers.Conv1D(128, kernel_size=8, padding='same', activation='relu')
        self.eeg_bn2 = layers.BatchNormalization()
        self.eeg_pool2 = layers.MaxPooling1D(pool_size=4, strides=4)
        self.eeg_dropout2 = layers.Dropout(0.3)
        
        self.eeg_conv3 = layers.Conv1D(256, kernel_size=4, padding='same', activation='relu')
        self.eeg_bn3 = layers.BatchNormalization()
        self.eeg_gap = layers.GlobalAveragePooling1D()
        
        # EOG Stream (1 channel)
        self.eog_conv1 = layers.Conv1D(32, kernel_size=50, strides=6, padding='same', activation='relu')
        self.eog_bn1 = layers.BatchNormalization()
        self.eog_pool1 = layers.MaxPooling1D(pool_size=8, strides=8)
        self.eog_dropout1 = layers.Dropout(0.3)
        
        self.eog_conv2 = layers.Conv1D(64, kernel_size=8, padding='same', activation='relu')
        self.eog_bn2 = layers.BatchNormalization()
        self.eog_pool2 = layers.MaxPooling1D(pool_size=4, strides=4)
        self.eog_dropout2 = layers.Dropout(0.3)
        
        self.eog_conv3 = layers.Conv1D(256, kernel_size=4, padding='same', activation='relu')
        self.eog_bn3 = layers.BatchNormalization()
        self.eog_gap = layers.GlobalAveragePooling1D()
        
        # EMG Stream (1 channel)
        self.emg_conv1 = layers.Conv1D(32, kernel_size=50, strides=6, padding='same', activation='relu')
        self.emg_bn1 = layers.BatchNormalization()
        self.emg_pool1 = layers.MaxPooling1D(pool_size=8, strides=8)
        self.emg_dropout1 = layers.Dropout(0.3)
        
        self.emg_conv2 = layers.Conv1D(64, kernel_size=8, padding='same', activation='relu')
        self.emg_bn2 = layers.BatchNormalization()
        self.emg_pool2 = layers.MaxPooling1D(pool_size=4, strides=4)
        self.emg_dropout2 = layers.Dropout(0.3)
        
        self.emg_conv3 = layers.Conv1D(256, kernel_size=4, padding='same', activation='relu')
        self.emg_bn3 = layers.BatchNormalization()
        self.emg_gap = layers.GlobalAveragePooling1D()
        
        # Attention mechanism for modality fusion
        self.attention_dense = layers.Dense(1, activation='sigmoid', name='modality_attention')
        
        # Fusion and classification layers
        self.fusion_dense1 = layers.Dense(256, activation='relu')
        self.fusion_bn = layers.BatchNormalization()
        self.fusion_dropout = layers.Dropout(0.5)
        
        self.fusion_dense2 = layers.Dense(128, activation='relu')
        
        # Output layer
        self.output_layer = layers.Dense(num_classes, activation='softmax', name='sleep_stage')
        
    def call(self, inputs, training=False):
        """
        Forward pass
        
        Args:
            inputs: Dictionary with keys 'eeg', 'eog', 'emg'
                    eeg: (batch_size, epoch_length, 2)
                    eog: (batch_size, epoch_length, 1)
                    emg: (batch_size, epoch_length, 1)
            training: Boolean flag for training mode
            
        Returns:
            output: (batch_size, num_classes) - Sleep stage probabilities
        """
        eeg_input = inputs['eeg']
        eog_input = inputs['eog']
        emg_input = inputs['emg']
        
        # EEG stream
        eeg = self.eeg_conv1(eeg_input)
        eeg = self.eeg_bn1(eeg, training=training)
        eeg = self.eeg_pool1(eeg)
        eeg = self.eeg_dropout1(eeg, training=training)
        
        eeg = self.eeg_conv2(eeg)
        eeg = self.eeg_bn2(eeg, training=training)
        eeg = self.eeg_pool2(eeg)
        eeg = self.eeg_dropout2(eeg, training=training)
        
        eeg = self.eeg_conv3(eeg)
        eeg = self.eeg_bn3(eeg, training=training)
        eeg_features = self.eeg_gap(eeg)
        
        # EOG stream
        eog = self.eog_conv1(eog_input)
        eog = self.eog_bn1(eog, training=training)
        eog = self.eog_pool1(eog)
        eog = self.eog_dropout1(eog, training=training)
        
        eog = self.eog_conv2(eog)
        eog = self.eog_bn2(eog, training=training)
        eog = self.eog_pool2(eog)
        eog = self.eog_dropout2(eog, training=training)
        
        eog = self.eog_conv3(eog)
        eog = self.eog_bn3(eog, training=training)
        eog_features = self.eog_gap(eog)
        
        # EMG stream
        emg = self.emg_conv1(emg_input)
        emg = self.emg_bn1(emg, training=training)
        emg = self.emg_pool1(emg)
        emg = self.emg_dropout1(emg, training=training)
        
        emg = self.emg_conv2(emg)
        emg = self.emg_bn2(emg, training=training)
        emg = self.emg_pool2(emg)
        emg = self.emg_dropout2(emg, training=training)
        
        emg = self.emg_conv3(emg)
        emg = self.emg_bn3(emg, training=training)
        emg_features = self.emg_gap(emg)
        
        # Stack modality features
        stacked_features = tf.stack([eeg_features, eog_features, emg_features], axis=1)
        
        # Compute attention weights for each modality
        # stacked_features shape: (batch, 3, 256)
        attention_weights = self.attention_dense(stacked_features)  # (batch, 3, 1)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)  # Normalize across modalities
        
        # Apply attention weights and reduce
        weighted_features = tf.reduce_sum(
            stacked_features * attention_weights, 
            axis=1
        )  # Result shape: (batch, 256)
        
        # Fusion and classification
        fused = self.fusion_dense1(weighted_features)
        fused = self.fusion_bn(fused, training=training)
        fused = self.fusion_dropout(fused, training=training)
        
        fused = self.fusion_dense2(fused)
        
        # Output
        output = self.output_layer(fused)
        
        return output
    
    def get_attention_weights(self, inputs):
        """
        Get attention weights for each modality
        Useful for model interpretability
        """
        eeg_input = inputs['eeg']
        eog_input = inputs['eog']
        emg_input = inputs['emg']
        
        # Forward pass through streams
        eeg = self.eeg_gap(self.eeg_bn3(self.eeg_conv3(
            self.eeg_pool2(self.eeg_bn2(self.eeg_conv2(
                self.eeg_pool1(self.eeg_bn1(self.eeg_conv1(eeg_input)))))))))
        
        eog = self.eog_gap(self.eog_bn3(self.eog_conv3(
            self.eog_pool2(self.eog_bn2(self.eog_conv2(
                self.eog_pool1(self.eog_bn1(self.eog_conv1(eog_input)))))))))
        
        emg = self.emg_gap(self.emg_bn3(self.emg_conv3(
            self.emg_pool2(self.emg_bn2(self.emg_conv2(
                self.emg_pool1(self.emg_bn1(self.emg_conv1(emg_input)))))))))
        
        stacked_features = tf.stack([eeg, eog, emg], axis=1)
        attention_input = tf.reduce_mean(stacked_features, axis=-1, keepdims=True)
        attention_weights = self.attention_dense(attention_input)
        
        return attention_weights


def build_model(num_classes=5, epoch_length=3000):
    """
    Build and compile the multimodal sleep stage classification model
    
    Args:
        num_classes: Number of sleep stages
        epoch_length: Length of each epoch in samples
        
    Returns:
        Compiled Keras model
    """
    # Define inputs
    eeg_input = keras.Input(shape=(epoch_length, 2), name='eeg')
    eog_input = keras.Input(shape=(epoch_length, 1), name='eog')
    emg_input = keras.Input(shape=(epoch_length, 1), name='emg')
    
    # Create model
    model = MultimodalSleepNet(num_classes=num_classes, epoch_length=epoch_length)
    
    # Build model by calling it once
    inputs = {'eeg': eeg_input, 'eog': eog_input, 'emg': emg_input}
    outputs = model(inputs)
    
    # Create functional model
    functional_model = keras.Model(inputs=[eeg_input, eog_input, emg_input], 
                                   outputs=outputs,
                                   name='MultimodalSleepNet')
    
    # Compile model
    functional_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    return functional_model


if __name__ == '__main__':
    # Test model creation
    print("Building Multimodal Sleep Stage Classification Model...")
    model = build_model()
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nModel architecture created successfully!")
    print(f"Total parameters: {model.count_params():,}")


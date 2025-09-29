Drone Detection and Classification using 60 GHz Millimeter-Wave Radar
=====================================================================

Author: Maguette Gueye
Project: Simultaneous Monitoring and Identifying Drone Targets using mmWave Radar and CNN

1. Objective
------------
This research aims to detect and classify small drones (UAS) and birds using
millimeter-wave radar. A CNN is trained to analyze micro-Doppler spectrograms
generated from radar IQ signals. The system performs simultaneous monitoring
(tracking position) and identifying (classifying type).

2. Dataset
----------
- Millimeter Wave 60 GHz Radar Measurements: UAS and Birds
- Collected with FMCW radar at 60 GHz
- Includes raw IQ signals capturing micro-Doppler effects (propeller rotation, wing flapping)
- Data split: 80% training, 20% testing

3. Data Preprocessing
---------------------
- Normalization: magnitude of IQ signals scaled to [0,1]
- Micro-Doppler spectrogram generation using Short-Time Fourier Transform (STFT)
- Spectrograms resized to 128x128 pixels, converted to 3-channel images
- Labels:
    0 → Bird
    1 → Drone

Mathematical details:
- Magnitude normalization: X_norm = (|X| - min(|X|)) / (max(|X|) - min(|X|))
- STFT: S(t,f) = Σ x[n] * w[n-m] * exp(-j2πfn/N)
- Micro-Doppler captures periodic motions (propellers, wings) encoded in frequency domain

4. CNN Model Architecture
-------------------------
- 3 Convolutional layers with ReLU activation + MaxPooling
- Flatten → Fully Connected (128 units) → Output (2 classes)
- Loss: Cross-Entropy: L = - Σ y_i * log(p_i)
- Optimizer: Adam (lr=1e-3)
- Metrics: Accuracy, Precision, Recall, F1

5. Training & Evaluation
------------------------
- Batch size: 16, Epochs: 20
- Validation monitored per epoch
- Simulated results (~70% accuracy) show steady improvement:
    Epoch 1: ~55%
    Epoch 5: ~60%
    Epoch 10: ~65%
    Epoch 15: ~68%
    Epoch 20: ~70%
- Confusion matrix (simulated):
        Predicted
        Drone  Bird
Actual
Drone    350   150
Bird     130   370

- Sample spectrogram predictions show correct and incorrect classifications,
illustrating real-world challenges.

6. Visualizations
-----------------
- Loss curves: decreasing training/validation loss
- Accuracy curves: validation accuracy improving over epochs
- Confusion matrix: shows overall ~70% correct classification
- Sample predictions: demonstrate how micro-Doppler spectrograms are classified

7. Notes
--------
- Logs stored in logs/
- Models saved in outputs/models/
- Plots and figures saved in outputs/plots/
- This README and plots simulate outcomes and methodology, showing workflow
progress even before completing actual experiments.

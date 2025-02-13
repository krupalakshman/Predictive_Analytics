# Predictive Maintenance of Industrial Equipment Using LSTM

## Overview

This project implements a predictive maintenance model for aircraft engines using LSTM (Long Short-Term Memory) networks. The model is trained to predict engine failure within a certain number of cycles based on sensor data.

## Dataset

The dataset consists of:

- **PM\_train.txt**: Aircraft engine run-to-failure data (training set).
- **PM\_test.txt**: Aircraft engine operating data (testing set, without failure events recorded).
- **PM\_truth.txt**: Actual remaining cycles for each engine in the test dataset.

## Project Structure

```
.
├── data/
│   ├── PM_train.txt
│   ├── PM_test.txt
│   ├── PM_truth.txt
│
├── Output/
│   ├── binary_model.keras  # Trained model
│   ├── binary_submit_train.csv  # Predictions on train set
│   ├── binary_submit_test.csv  # Predictions on test set
│   ├── model_accuracy.png  # Accuracy graph
│   ├── model_loss.png  # Loss graph
│   ├── model_verify.png  # Prediction vs Actual visualization
│
├── model_training  # Training and evaluation script
├── manual_prediction  # Script for manual input prediction
├── gui_prediction  # GUI for manual predictions using Tkinter
└── README.md
```

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Required Python packages:
  ```sh
  pip install numpy pandas matplotlib scikit-learn keras tensorflow
  ```

## Model Training

The LSTM-based predictive maintenance model is built using the following key steps:

1. **Data Preprocessing**:
   - Data is read and cleaned from the training and testing datasets.
   - Remaining Useful Life (RUL) and labels are generated.
   - Features are normalized using Min-Max Scaling.
2. **Model Architecture**:
   - Two LSTM layers with dropout for regularization.
   - Final dense layer with a sigmoid activation function for binary classification.
3. **Training**:
   - The model is trained for 100 epochs with early stopping.
   - The best-performing model is saved.
4. **Evaluation**:
   - Accuracy and loss graphs are plotted.
   - Precision, recall, and confusion matrix are computed.



## Manual Input Prediction

You can predict the likelihood of engine failure using manual inputs:



- Takes manual input for engine cycle and settings.
- Generates sensor values randomly.
- Preprocesses and normalizes input.
- Predicts failure probability using the trained model.

## GUI for Predictions

A GUI-based interface is available for easier interaction.

- Achieved **98.3% accuracy** on binary classification.
- Model successfully predicts engine failures within a given window.

## Future Improvements

- Implement multi-class classification for different failure types.
- Improve feature engineering to enhance prediction accuracy.
- Deploy the model as a web-based application.

## License

This project is open-source under the MIT License.

---

Developed as part of a Predictive Maintenance Project using Deep Learning.


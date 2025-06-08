# Multimodal Classification Pipeline for ECG Data using ML and LSTM Models

This project presents a comprehensive machine learning and deep learning pipeline for classifying ECG signals using traditional models (Logistic Regression, Random Forest) and deep learning (LSTM). The workflow includes preprocessing, visualization, dimensionality reduction, and performance evaluation using multiple metrics.

## 📚 Project Overview

### Dataset: MIT-BIH ECG dataset (`mitbih_train.csv`)
- Tasks:
  - Data preprocessing, normalization, and cleaning
  - Feature reduction via PCA
  - Model training and evaluation:
    - Logistic Regression (with GridSearchCV)
    - Random Forest Classifier
    - LSTM Neural Network with EarlyStopping
  - Evaluation metrics: Accuracy, ROC AUC, AUPR, Top-5 Accuracy
  - Visualizations: Correlation heatmaps, class distributions, ROC curves

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

## 📊 Models and Metrics

| Model              | Accuracy | ROC AUC (Macro) | AUPR (Macro) | Top-5 mAP |
|-------------------|----------|------------------|---------------|-------------|
| Logistic Regression | 95.3%   | 0.945           | 0.744        | 1.0         |
| Random Forest      | 97.6%   | 0.994           | 0.933        | 1.0         |
| LSTM               | 96.9%   | 0.977           | 0.867        | 1.0         |

## 📈 Visualizations

- Feature correlation heatmap
- Class distribution plots
- Feature histograms and box plots
- ROC curves per class for all models


## 📁 Folder Structure

```
ecg-classification-pipeline/
├── ML.ipynb                  # Jupyter notebook
├── ML.pdf                   # Exported notebook as PDF
├── README.md
└── requirements.txt         # Python dependencies
```


## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

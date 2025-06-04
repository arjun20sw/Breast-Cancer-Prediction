# Breast-Cancer-Prediction
# 🧠 Breast Cancer Prediction using Machine Learning

This project predicts whether a breast cancer tumor is **malignant** or **benign** using machine learning techniques. It uses the popular Breast Cancer Wisconsin dataset and aims to assist in early detection and diagnosis.

## 📂 Dataset

- **Source**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features**: 30 numerical features (e.g., radius, texture, smoothness)
- **Target**: 
  - 0: Malignant  
  - 1: Benign
- **Instances**: 569 samples

## 🎯 Objectives

- Preprocess and clean the dataset
- Visualize feature relationships
- Train multiple machine learning models
- Evaluate performance using metrics like accuracy, precision, recall, and F1-score
- Save and reuse the trained model

## 🛠️ Technologies Used

- Python
- NumPy & Pandas
- Matplotlib & Seaborn (Data Visualization)
- Scikit-learn (Modeling)
- Jupyter Notebook / VS Code

## 🧪 Models Tried

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

> The best performing model is: **[Your best model name]** with accuracy of **XX%** on test data.

## 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Score

## 📁 Project Structure

```bash
breast-cancer-prediction/
├── data/                  # Dataset files (if applicable)
├── notebooks/             # Jupyter notebooks
├── models/                # Trained models (pickle or joblib files)
├── src/                   # Python scripts (training, preprocessing)
├── outputs/               # Plots and reports
├── requirements.txt       # Python dependencies
├── main.py                # Script to train or test model
└── README.md              # Project documentation

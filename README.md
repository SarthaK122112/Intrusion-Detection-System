# Intrusion Detection System (IDS) for IoT Networks

This project implements a Machine Learning-based Intrusion Detection System to classify IoT network traffic as **normal** or **malicious**, using data from a Philips IoT device environment.

## 🧠 Project Overview

This IDS uses multiple machine learning models to analyze and classify traffic based on protocol types and behavior patterns. The dataset is preprocessed with label encoding, feature engineering, and scaling before being passed into models like:

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- Neural Network (MLP)

The best-performing model is selected based on accuracy.

## 📁 Files

- `541c5d84-242c-45e6-8d3c-3697cbbb409f.py`: Main Python script for the IDS pipeline.
- `philips.csv`: The dataset containing network traffic data.

## ⚙️ Features Extracted

- Protocol encoding
- Packet length and time deltas
- Source/Destination prefix encoding
- Protocol-specific flags (MDNS, DHCP, NTP)

## 🧪 Model Evaluation

Each model is trained and tested on a 70-30 train-test split. Metrics such as accuracy, confusion matrix, and classification report are used for performance comparison.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/intrusion-detection-system.git
   cd intrusion-detection-system

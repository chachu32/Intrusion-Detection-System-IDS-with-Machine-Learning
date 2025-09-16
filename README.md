# Intrusion Detection System (IDS) with Machine Learning

## 📌 **Objective**
Design and implement an Intrusion Detection System (IDS) that uses machine learning algorithms to detect malicious network activities based on traffic patterns. The system analyzes network traffic to identify anomalies and potential attacks in real-time.

## 🛠 **Tools Used**
- **Python** – Programming language for implementing machine learning models.
- **Scikit-learn** – Used for feature extraction, preprocessing, and building models.
- **TensorFlow** – Used for creating deep learning models to improve detection accuracy.
- **Wireshark** – Used to capture and analyze network packets for training and testing data.

## ✅ **Skills Learned**
- Machine learning techniques applied to cybersecurity.
- Network traffic analysis and preprocessing.
- Anomaly detection using classification algorithms.
- Working with real network data using packet capture tools.
- Model evaluation using accuracy, precision, recall, and F1-score metrics.

## 📂 **Folder Structure**
IDS-ML/
├── data/
│ ├── captured_traffic.csv # Processed network traffic data
│ └── test_data.csv # Test dataset
├── models/
│ ├── decision_tree.pkl # Saved machine learning model
│ └── deep_learning_model.h5 # Saved TensorFlow model
├── notebooks/
│ └── exploration.ipynb # Exploratory data analysis
├── scripts/
│ ├── train_model.py # Script to train models
│ ├── detect.py # Script to perform intrusion detection
│ └── preprocess.py # Data preprocessing pipeline
├── requirements.txt # List of dependencies
└── README.md

## 🚀 **How to Run**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Capture traffic using Wireshark or load preprocessed traffic data.

3. Run preprocessing: python scripts/preprocess.py
4. Train the models: python scripts/train_model.py

5.Perform detection on test data: python scripts/detect.py

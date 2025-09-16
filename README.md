# Intrusion Detection System (IDS) with Machine Learning

## Objective
Design and implement an Intrusion Detection System (IDS) that uses machine learning algorithms to detect malicious network activity based on traffic patterns. The system analyzes network traffic to identify anomalies and potential attacks in real-time.

## Tools Used
- **Python** – Programming language for implementing machine learning models.
- **Scikit-learn** – For feature extraction, preprocessing, and building models.
- **TensorFlow** – For creating deep learning models to improve detection accuracy.
- **Wireshark** – To capture and analyze network packets for training and testing data.

## Skills Learned
- Applying machine learning techniques to cybersecurity.
- Network traffic analysis and preprocessing.
- Anomaly detection using classification algorithms.
- Working with real network data captured via Wireshark.
- Model evaluation using accuracy, precision, recall, and F1-score metrics.

## Folder Structure

```
IDS-ML/
├── data/
│   ├── captured_traffic.csv      # Processed network traffic data
│   └── test_data.csv             # Test dataset
├── models/
│   ├── decision_tree.pkl         # Saved ML model
│   └── deep_learning_model.h5    # Saved TensorFlow model
├── notebooks/
│   └── exploration.ipynb         # Exploratory data analysis
├── scripts/
│   ├── train_model.py            # Script to train models
│   ├── detect.py                 # Script to perform intrusion detection
│   └── preprocess.py             # Data preprocessing pipeline
├── requirements.txt              # List of dependencies
└── README.md
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Capture network traffic using Wireshark or load preprocessed traffic data.
3. Run preprocessing:
   ```bash
   python scripts/preprocess.py
   ```
4. Train the models:
   ```bash
   python scripts/train_model.py
   ```
5. Perform detection on test data:
   ```bash
   python scripts/detect.py
   ```

## Demo Explanation
> "The demo shows how network traffic is analyzed in real time using machine learning models. It first preprocesses traffic data captured by Wireshark, extracts relevant features, and then applies classification algorithms to detect anomalies like DoS attacks, scanning attempts, or malware communication. The results are evaluated using metrics like accuracy and false positive rate."

You can run the detection script live by feeding sample network data and showing how malicious traffic is flagged.

## Future Improvements
- Implement real-time packet capture using sockets or APIs.
- Integrate with alerting systems for automated responses.
- Explore advanced deep learning models like RNNs or autoencoders for anomaly detection.

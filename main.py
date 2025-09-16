#!/usr/bin/env python3
"""
Main entry point for the Intrusion Detection System (IDS) with Machine Learning
"""

import argparse
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from detector import IntrusionDetector
from utils import save_model, load_model, plot_results

def main():
    parser = argparse.ArgumentParser(description='Intrusion Detection System with ML')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--detect', action='store_true', help='Run intrusion detection')
    parser.add_argument('--input', type=str, help='Input file for detection')
    parser.add_argument('--output', type=str, default='results.csv', help='Output file for results')
    parser.add_argument('--model', type=str, choices=['rf', 'nn', 'both'], default='both', 
                       help='Model to use for detection (rf: Random Forest, nn: Neural Network)')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training models...")
        train_models()
    elif args.detect:
        if not args.input:
            print("Error: Please specify an input file with --input")
            sys.exit(1)
        print("Running intrusion detection...")
        detect_intrusions(args.input, args.output, args.model)
    else:
        parser.print_help()

def train_models():
    """Train the machine learning models"""
    # Load and preprocess data
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
    
    # Train models
    trainer = ModelTrainer()
    
    print("Training Random Forest model...")
    rf_model, rf_history = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    save_model(rf_model, 'models/random_forest_model.pkl')
    
    print("Training Neural Network model...")
    nn_model, nn_history = trainer.train_neural_network(X_train, y_train, X_test, y_test)
    nn_model.save('models/neural_network_model.h5')
    
    # Evaluate and plot results
    rf_metrics = trainer.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    nn_metrics = trainer.evaluate_model(nn_model, X_test, y_test, 'neural_network')
    
    plot_results(rf_history, nn_history, rf_metrics, nn_metrics)
    
    print("Training completed. Models saved in models/ directory.")

def detect_intrusions(input_file, output_file, model_type):
    """Detect intrusions in the given input file"""
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_data(input_file)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_processed, _, _, _ = preprocessor.preprocess_data(df, training=False)
    
    # Load models
    detector = IntrusionDetector()
    
    if model_type in ['rf', 'both']:
        rf_model = load_model('models/random_forest_model.pkl')
        rf_predictions = detector.detect(rf_model, X_processed, model_type='random_forest')
        df['rf_prediction'] = rf_predictions[0]
        df['rf_confidence'] = rf_predictions[1]
    
    if model_type in ['nn', 'both']:
        nn_model = detector.load_neural_network('models/neural_network_model.h5')
        nn_predictions = detector.detect(nn_model, X_processed, model_type='neural_network')
        df['nn_prediction'] = nn_predictions[0]
        df['nn_confidence'] = nn_predictions[1]
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
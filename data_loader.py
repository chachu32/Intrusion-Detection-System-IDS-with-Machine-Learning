import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    def load_data(self, file_path=None):
        """Load network traffic data from CSV file"""
        if file_path is None:
            file_path = os.path.join(self.data_dir, 'raw', 'sample_dataset.csv')
        
        if not os.path.exists(file_path):
            # Generate sample data if file doesn't exist
            print(f"File {file_path} not found. Generating sample data...")
            return self._generate_sample_data()
        
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self, n_samples=10000):
        """Generate sample network traffic data for demonstration"""
        print("Generating sample data...")
        
        # Features that might be present in network traffic data
        feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'label'
        ]
        
        # Initialize DataFrame
        df = pd.DataFrame(columns=feature_names)
        
        # Generate synthetic data
        df['duration'] = np.random.exponential(scale=1.0, size=n_samples)
        df['protocol_type'] = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
        df['service'] = np.random.choice(range(10), size=n_samples)
        df['flag'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        df['src_bytes'] = np.random.lognormal(mean=0, sigma=2.0, size=n_samples)
        df['dst_bytes'] = np.random.lognormal(mean=0, sigma=2.0, size=n_samples)
        df['land'] = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])
        df['wrong_fragment'] = np.random.poisson(lam=0.1, size=n_samples)
        df['urgent'] = np.random.poisson(lam=0.01, size=n_samples)
        df['hot'] = np.random.poisson(lam=0.5, size=n_samples)
        
        # Create labels: 0 for normal, 1 for malicious
        mal_prob = (
            0.1 * (df['duration'] > 10) +
            0.2 * (df['src_bytes'] > 10000) +
            0.3 * (df['wrong_fragment'] > 0) +
            0.1 * (df['hot'] > 5)
        )
        
        df['label'] = ((mal_prob + 0.1 * np.random.random(n_samples)) > 0.3).astype(int)
        
        # Save sample data for future use
        sample_path = os.path.join(self.data_dir, 'raw', 'sample_dataset.csv')
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        df.to_csv(sample_path, index=False)
        print(f"Sample data saved to {sample_path}")
        
        return df
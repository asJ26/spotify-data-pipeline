import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.model_training import SpotifyModelTrainer, ModelTrainingError

class TestSpotifyModelTrainer(unittest.TestCase):
    """Test cases for SpotifyModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.trainer = SpotifyModelTrainer()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        
        self.X = pd.DataFrame({
            'danceability': np.random.random(n_samples),
            'energy': np.random.random(n_samples),
            'key': np.random.randint(0, 12, n_samples),
            'loudness': np.random.uniform(-20, 0, n_samples),
            'mode': np.random.randint(0, 2, n_samples),
            'speechiness': np.random.random(n_samples),
            'acousticness': np.random.random(n_samples),
            'instrumentalness': np.random.random(n_samples),
            'liveness': np.random.random(n_samples),
            'valence': np.random.random(n_samples),
            'tempo': np.random.uniform(60, 180, n_samples),
            'duration_ms': np.random.uniform(100000, 300000, n_samples),
            'time_signature': np.random.randint(3, 5, n_samples)
        })
        
        # Create target variable (popularity) with some relationship to features
        self.y = (
            0.3 * self.X['danceability'] +
            0.3 * self.X['energy'] +
            0.2 * self.X['valence'] +
            0.2 * np.random.random(n_samples)
        ) * 100
        
        # Sample processed data in parquet format
        self.sample_processed_data = pd.DataFrame({
            'id': [f'track{i}' for i in range(n_samples)],
            'name': [f'Track {i}' for i in range(n_samples)],
            'popularity': self.y,
            **self.X
        })
    
    @patch('src.model_training.boto3.client')
    def test_load_data_from_s3(self, mock_boto3_client):
        """Test loading data from S3."""
        # Mock S3 response
        mock_s3 = MagicMock()
        mock_response = {
            'Body': MagicMock()
        }
        mock_s3.get_object.return_value = mock_response
        mock_boto3_client.return_value = mock_s3
        
        # Mock parquet reading
        with patch('pandas.read_parquet') as mock_read_parquet:
            mock_read_parquet.return_value = self.sample_processed_data
            
            # Load data
            data = self.trainer._load_data_from_s3('test_file.parquet')
            
            # Assertions
            self.assertEqual(len(data), 100)
            mock_s3.get_object.assert_called_once()
    
    def test_prepare_data(self):
        """Test data preparation for training."""
        # Prepare data
        X, y = self.trainer._prepare_data(self.sample_processed_data)
        
        # Assertions
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(y.shape[0], 100)
        self.assertNotIn('id', X)
        self.assertNotIn('name', X)
        self.assertNotIn('popularity', X)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create and train a simple model
        model = DecisionTreeRegressor(random_state=42)
        X_train = self.X.values[:80]
        X_test = self.X.values[80:]
        y_train = self.y[:80]
        y_test = self.y[80:]
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self.trainer._evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )
        
        # Assertions
        self.assertIn('train_mse', metrics)
        self.assertIn('test_mse', metrics)
        self.assertIn('train_rmse', metrics)
        self.assertIn('test_rmse', metrics)
        self.assertIn('train_mae', metrics)
        self.assertIn('test_mae', metrics)
        self.assertIn('train_r2', metrics)
        self.assertIn('test_r2', metrics)
        self.assertIn('cv_mean_r2', metrics)
        self.assertIn('cv_std_r2', metrics)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig):
        """Test feature importance plotting."""
        # Create and train a model
        model = RandomForestRegressor(random_state=42)
        model.fit(self.X, self.y)
        
        # Plot feature importance
        self.trainer._plot_feature_importance(
            model,
            self.X.columns.tolist(),
            'random_forest'
        )
        
        # Assertions
        mock_savefig.assert_called_once()
    
    @patch('src.model_training.SpotifyModelTrainer._load_data_from_s3')
    @patch('src.model_training.SpotifyModelTrainer._save_model_to_s3')
    @patch('src.model_training.SpotifyModelTrainer._save_metrics_to_s3')
    def test_train_models(
        self,
        mock_save_metrics,
        mock_save_model,
        mock_load_data
    ):
        """Test the complete model training pipeline."""
        # Mock loading data
        mock_load_data.return_value = self.sample_processed_data
        
        # Train models
        self.trainer.train_models()
        
        # Assertions
        mock_load_data.assert_called_once()
        self.assertGreater(mock_save_model.call_count, 0)
        mock_save_metrics.assert_called_once()
    
    @patch('src.model_training.boto3.client')
    def test_save_model_to_s3(self, mock_boto3_client):
        """Test saving model to S3."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        
        # Create a simple model
        model = DecisionTreeRegressor()
        model.fit(self.X, self.y)
        
        # Save model
        self.trainer._save_model_to_s3(model, 'test_model')
        
        # Assertions
        mock_s3.upload_file.assert_called_once()
    
    @patch('src.model_training.boto3.client')
    def test_save_metrics_to_s3(self, mock_boto3_client):
        """Test saving metrics to S3."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        
        # Sample metrics
        metrics = {
            'model1': {
                'train_mse': 100,
                'test_mse': 120,
                'r2_score': 0.85
            }
        }
        
        # Save metrics
        self.trainer._save_metrics_to_s3(metrics)
        
        # Assertions
        mock_s3.put_object.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in model training."""
        # Test with invalid data
        with self.assertRaises(Exception):
            self.trainer._prepare_data(pd.DataFrame())
        
        # Test with invalid model
        with self.assertRaises(Exception):
            self.trainer._evaluate_model(
                None,
                self.X.values,
                self.X.values,
                self.y,
                self.y
            )

if __name__ == '__main__':
    unittest.main()

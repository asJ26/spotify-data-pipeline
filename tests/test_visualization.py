import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from src.visualization import SpotifyDataVisualizer, VisualizationError

class TestSpotifyDataVisualizer(unittest.TestCase):
    """Test cases for SpotifyDataVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.visualizer = SpotifyDataVisualizer()
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'id': [f'track{i}' for i in range(n_samples)],
            'name': [f'Track {i}' for i in range(n_samples)],
            'popularity': np.random.uniform(0, 100, n_samples),
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
            'time_signature': np.random.randint(3, 5, n_samples),
            'release_date': pd.date_range(start='2020-01-01', periods=n_samples)
        })
        
        # Sample predictions data
        self.sample_predictions = pd.DataFrame({
            'track_id': [f'track{i}' for i in range(n_samples)],
            'predicted_popularity': np.random.uniform(0, 100, n_samples),
            'actual_popularity': np.random.uniform(0, 100, n_samples),
            'error': np.random.uniform(-10, 10, n_samples)
        })
    
    @patch('src.visualization.boto3.client')
    def test_load_data_from_s3_parquet(self, mock_boto3_client):
        """Test loading parquet data from S3."""
        # Mock S3 response
        mock_s3 = MagicMock()
        mock_response = {
            'Body': MagicMock()
        }
        mock_s3.get_object.return_value = mock_response
        mock_boto3_client.return_value = mock_s3
        
        # Mock parquet reading
        with patch('pandas.read_parquet') as mock_read_parquet:
            mock_read_parquet.return_value = self.sample_data
            
            # Load data
            data = self.visualizer._load_data_from_s3('test_file.parquet')
            
            # Assertions
            self.assertEqual(len(data), 100)
            mock_s3.get_object.assert_called_once()
    
    @patch('src.visualization.boto3.client')
    def test_load_data_from_s3_json(self, mock_boto3_client):
        """Test loading JSON data from S3."""
        # Mock S3 response
        mock_s3 = MagicMock()
        mock_response = {
            'Body': MagicMock(
                read=lambda: json.dumps(self.sample_predictions.to_dict()).encode()
            )
        }
        mock_s3.get_object.return_value = mock_response
        mock_boto3_client.return_value = mock_s3
        
        # Load data
        data = self.visualizer._load_data_from_s3('test_file.json')
        
        # Assertions
        self.assertIsInstance(data, pd.DataFrame)
        mock_s3.get_object.assert_called_once()
    
    @patch('src.visualization.boto3.client')
    @patch('matplotlib.pyplot.savefig')
    def test_save_plot_to_s3(self, mock_savefig, mock_boto3_client):
        """Test saving plot to S3."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        
        # Create a simple plot
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        
        # Save plot
        self.visualizer._save_plot_to_s3('test_plot')
        
        # Assertions
        mock_savefig.assert_called_once()
        mock_s3.upload_file.assert_called_once()
    
    @patch('src.visualization.SpotifyDataVisualizer._save_plot_to_s3')
    def test_plot_feature_distributions(self, mock_save_plot):
        """Test plotting feature distributions."""
        # Plot distributions
        self.visualizer.plot_feature_distributions(self.sample_data)
        
        # Assertions
        expected_calls = len(
            self.sample_data.select_dtypes(include=[np.number]).columns
        )
        self.assertEqual(mock_save_plot.call_count, expected_calls)
    
    @patch('src.visualization.SpotifyDataVisualizer._save_plot_to_s3')
    def test_plot_correlation_matrix(self, mock_save_plot):
        """Test plotting correlation matrix."""
        # Plot correlation matrix
        self.visualizer.plot_correlation_matrix(self.sample_data)
        
        # Assertions
        mock_save_plot.assert_called_once_with('correlation_matrix')
    
    @patch('src.visualization.SpotifyDataVisualizer._save_plot_to_s3')
    def test_plot_popularity_vs_features(self, mock_save_plot):
        """Test plotting popularity vs features."""
        # Plot relationships
        self.visualizer.plot_popularity_vs_features(self.sample_data)
        
        # Assertions
        expected_calls = len(
            [col for col in self.sample_data.select_dtypes(include=[np.number]).columns
             if col != 'popularity']
        )
        self.assertEqual(mock_save_plot.call_count, expected_calls)
    
    @patch('src.visualization.SpotifyDataVisualizer._load_data_from_s3')
    @patch('src.visualization.SpotifyDataVisualizer._save_plot_to_s3')
    def test_plot_prediction_analysis(self, mock_save_plot, mock_load_data):
        """Test plotting prediction analysis."""
        # Mock loading predictions
        mock_load_data.return_value = self.sample_predictions
        
        # Plot analysis
        self.visualizer.plot_prediction_analysis('predictions.json')
        
        # Assertions
        self.assertEqual(mock_save_plot.call_count, 2)  # Two plots should be created
    
    @patch('src.visualization.SpotifyDataVisualizer._save_plot_to_s3')
    def test_plot_feature_importance(self, mock_save_plot):
        """Test plotting feature importance."""
        # Create sample feature importance
        feature_importance = pd.Series(
            np.random.random(10),
            index=[f'feature_{i}' for i in range(10)]
        )
        
        # Plot feature importance
        self.visualizer.plot_feature_importance(
            feature_importance,
            'Test Feature Importance'
        )
        
        # Assertions
        mock_save_plot.assert_called_once_with('feature_importance')
    
    @patch('src.visualization.SpotifyDataVisualizer._save_plot_to_s3')
    def test_plot_popularity_trends(self, mock_save_plot):
        """Test plotting popularity trends."""
        # Plot trends
        self.visualizer.plot_popularity_trends(self.sample_data)
        
        # Assertions
        mock_save_plot.assert_called_once_with('popularity_trend')
    
    @patch('src.visualization.SpotifyDataVisualizer._save_plot_to_s3')
    def test_create_dashboard(self, mock_save_plot):
        """Test creating dashboard."""
        # Create dashboard
        self.visualizer.create_dashboard(self.sample_data)
        
        # Assertions
        mock_save_plot.assert_called_once_with('dashboard')
    
    def test_error_handling(self):
        """Test error handling in visualization."""
        # Test with invalid file format
        with self.assertRaises(VisualizationError):
            self.visualizer._load_data_from_s3('test.txt')
        
        # Test with invalid data
        with self.assertRaises(Exception):
            self.visualizer.plot_correlation_matrix(pd.DataFrame())
        
        # Test with missing required columns
        invalid_data = self.sample_data.drop('popularity', axis=1)
        with self.assertRaises(Exception):
            self.visualizer.plot_popularity_vs_features(invalid_data)

if __name__ == '__main__':
    unittest.main()

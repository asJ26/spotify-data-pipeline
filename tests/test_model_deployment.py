import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor

from src.model_deployment import SpotifyModelDeployer, ModelDeploymentError

class TestSpotifyModelDeployer(unittest.TestCase):
    """Test cases for SpotifyModelDeployer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.deployer = SpotifyModelDeployer()
        
        # Create a sample trained model
        np.random.seed(42)
        n_samples = 100
        
        # Sample features
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
        
        # Create target variable
        self.y = (
            0.3 * self.X['danceability'] +
            0.3 * self.X['energy'] +
            0.2 * self.X['valence'] +
            0.2 * np.random.random(n_samples)
        ) * 100
        
        # Train a simple model
        self.model = DecisionTreeRegressor(random_state=42)
        self.model.fit(self.X, self.y)
        
        # Sample track data for prediction
        self.sample_track = {
            'id': 'test_track_1',
            'danceability': 0.8,
            'energy': 0.6,
            'key': 5,
            'loudness': -5.5,
            'mode': 1,
            'speechiness': 0.1,
            'acousticness': 0.2,
            'instrumentalness': 0.0,
            'liveness': 0.3,
            'valence': 0.7,
            'tempo': 120.0,
            'duration_ms': 200000,
            'time_signature': 4
        }
    
    @patch('src.model_deployment.boto3.client')
    def test_get_latest_model(self, mock_boto3_client):
        """Test getting the latest model from S3."""
        # Mock S3 response
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'models/model_v1.joblib', 'LastModified': datetime(2023, 1, 1)},
                {'Key': 'models/model_v2.joblib', 'LastModified': datetime(2023, 1, 2)}
            ]
        }
        mock_boto3_client.return_value = mock_s3
        
        # Get latest model
        model_key = self.deployer._get_latest_model()
        
        # Assertions
        self.assertEqual(model_key, 'models/model_v2.joblib')
        mock_s3.list_objects_v2.assert_called_once()
    
    @patch('src.model_deployment.boto3.client')
    @patch('joblib.load')
    def test_load_model_from_s3(self, mock_joblib_load, mock_boto3_client):
        """Test loading model from S3."""
        # Mock S3 client and joblib
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        mock_joblib_load.return_value = self.model
        
        # Load model
        self.deployer._load_model_from_s3('models/test_model.joblib')
        
        # Assertions
        self.assertIsNotNone(self.deployer.model)
        mock_s3.download_file.assert_called_once()
        mock_joblib_load.assert_called_once()
    
    def test_prepare_features(self):
        """Test feature preparation for prediction."""
        # Prepare features
        features_df = self.deployer._prepare_features(self.sample_track)
        
        # Assertions
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 1)
        for feature in self.X.columns:
            self.assertIn(feature, features_df.columns)
    
    @patch('src.model_deployment.boto3.client')
    def test_save_prediction_to_s3(self, mock_boto3_client):
        """Test saving prediction results to S3."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        
        # Save prediction
        self.deployer._save_prediction_to_s3(
            'test_track_1',
            75.5,
            80.0
        )
        
        # Assertions
        mock_s3.put_object.assert_called_once()
        
        # Verify the prediction data format
        call_args = mock_s3.put_object.call_args[1]
        prediction_data = json.loads(call_args['Body'])
        
        self.assertEqual(prediction_data['track_id'], 'test_track_1')
        self.assertEqual(prediction_data['predicted_popularity'], 75.5)
        self.assertEqual(prediction_data['actual_popularity'], 80.0)
        self.assertEqual(prediction_data['error'], 4.5)
    
    @patch('src.model_deployment.SpotifyModelDeployer._get_latest_model')
    @patch('src.model_deployment.SpotifyModelDeployer._load_model_from_s3')
    def test_load_model(self, mock_load_model, mock_get_latest):
        """Test loading model with and without specified model key."""
        # Test with specific model key
        self.deployer.load_model('models/specific_model.joblib')
        mock_load_model.assert_called_with('models/specific_model.joblib')
        
        # Test without model key (should get latest)
        mock_get_latest.return_value = 'models/latest_model.joblib'
        self.deployer.load_model()
        mock_get_latest.assert_called_once()
        mock_load_model.assert_called_with('models/latest_model.joblib')
    
    @patch('src.model_deployment.SpotifyModelDeployer._save_prediction_to_s3')
    def test_predict(self, mock_save_prediction):
        """Test making predictions."""
        # Set up model
        self.deployer.model = self.model
        self.deployer.model_name = 'test_model'
        
        # Make prediction
        prediction = self.deployer.predict(
            self.sample_track,
            actual_popularity=80.0
        )
        
        # Assertions
        self.assertIsInstance(prediction, float)
        mock_save_prediction.assert_called_once()
    
    def test_batch_predict(self):
        """Test batch prediction."""
        # Set up model
        self.deployer.model = self.model
        self.deployer.model_name = 'test_model'
        
        # Create batch of tracks
        tracks = [self.sample_track for _ in range(3)]
        
        # Make predictions
        predictions = self.deployer.batch_predict(tracks)
        
        # Assertions
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(isinstance(p, float) for p in predictions))
    
    def test_error_handling(self):
        """Test error handling in model deployment."""
        # Test prediction without loaded model
        with self.assertRaises(Exception):
            self.deployer.predict(self.sample_track)
        
        # Test with invalid track data
        self.deployer.model = self.model
        with self.assertRaises(Exception):
            self.deployer.predict({'invalid': 'data'})
        
        # Test with missing features
        invalid_track = self.sample_track.copy()
        del invalid_track['danceability']
        with self.assertRaises(Exception):
            self.deployer.predict(invalid_track)

if __name__ == '__main__':
    unittest.main()

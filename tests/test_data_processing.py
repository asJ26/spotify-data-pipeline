import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_processing import SpotifyDataProcessor, DataProcessingError

class TestSpotifyDataProcessor(unittest.TestCase):
    """Test cases for SpotifyDataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = SpotifyDataProcessor()
        
        # Sample raw data
        self.sample_raw_data = {
            'tracks': [
                {
                    'id': 'track1',
                    'name': 'Test Track 1',
                    'artists': [{'name': 'Test Artist'}],
                    'album': {
                        'name': 'Test Album',
                        'album_type': 'album',
                        'release_date': '2023-01-01'
                    },
                    'popularity': 80,
                    'duration_ms': 200000,
                    'explicit': False
                }
            ],
            'audio_features': [
                {
                    'id': 'track1',
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
                    'time_signature': 4
                }
            ]
        }
        
        # Create sample DataFrame
        self.sample_df = pd.DataFrame({
            'id': ['track1'],
            'name': ['Test Track 1'],
            'popularity': [80],
            'duration_ms': [200000],
            'explicit': [False],
            'danceability': [0.8],
            'energy': [0.6],
            'key': [5],
            'loudness': [-5.5],
            'mode': [1],
            'speechiness': [0.1],
            'acousticness': [0.2],
            'instrumentalness': [0.0],
            'liveness': [0.3],
            'valence': [0.7],
            'tempo': [120.0],
            'time_signature': [4],
            'release_date': ['2023-01-01'],
            'album_type': ['album'],
            'artist_names': ['Test Artist']
        })
    
    @patch('src.data_processing.boto3.client')
    def test_load_data_from_s3(self, mock_boto3_client):
        """Test loading data from S3."""
        # Mock S3 response
        mock_s3 = MagicMock()
        mock_response = {
            'Body': MagicMock(
                read=lambda: json.dumps(self.sample_raw_data).encode()
            )
        }
        mock_s3.get_object.return_value = mock_response
        mock_boto3_client.return_value = mock_s3
        
        # Load data
        data = self.processor._load_data_from_s3('test_file.json')
        
        # Assertions
        self.assertEqual(len(data['tracks']), 1)
        self.assertEqual(len(data['audio_features']), 1)
        mock_s3.get_object.assert_called_once()
    
    def test_extract_features(self):
        """Test feature extraction from raw data."""
        # Extract features
        df = self.processor._extract_features(self.sample_raw_data)
        
        # Assertions
        self.assertEqual(len(df), 1)
        self.assertEqual(df['id'].iloc[0], 'track1')
        self.assertEqual(df['artist_names'].iloc[0], 'Test Artist')
        self.assertEqual(df['album_type'].iloc[0], 'album')
    
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        # Create DataFrame with missing values
        df = self.sample_df.copy()
        df.loc[0, 'danceability'] = np.nan
        df.loc[0, 'energy'] = np.nan
        
        # Handle missing values
        processed_df = self.processor._handle_missing_values(df)
        
        # Assertions
        self.assertFalse(processed_df.isnull().any().any())
    
    def test_create_time_features(self):
        """Test creation of time-based features."""
        # Create time features
        df = self.processor._create_time_features(self.sample_df)
        
        # Assertions
        self.assertIn('release_year', df.columns)
        self.assertIn('release_month', df.columns)
        self.assertIn('release_day', df.columns)
        self.assertEqual(df['release_year'].iloc[0], 2023)
        self.assertEqual(df['release_month'].iloc[0], 1)
        self.assertEqual(df['release_day'].iloc[0], 1)
    
    def test_normalize_features(self):
        """Test feature normalization."""
        # Normalize features
        df = self.processor._normalize_features(self.sample_df)
        
        # Assertions
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['id', 'popularity', 'release_year', 'release_month', 'release_day']:
                self.assertTrue(df[col].std() <= 1.0)
    
    def test_encode_categorical_features(self):
        """Test encoding of categorical features."""
        # Encode categorical features
        df = self.processor._encode_categorical_features(self.sample_df)
        
        # Assertions
        self.assertIn('album_type_album', df.columns)
        self.assertNotIn('album_type', df.columns)
        self.assertNotIn('artist_names', df.columns)
    
    @patch('src.data_processing.SpotifyDataProcessor._load_data_from_s3')
    @patch('src.data_processing.SpotifyDataProcessor._save_to_s3')
    def test_process_data(self, mock_save_s3, mock_load_s3):
        """Test the complete data processing pipeline."""
        # Mock loading data
        mock_load_s3.return_value = self.sample_raw_data
        
        # Process data
        self.processor.process_data('test_file.json')
        
        # Assertions
        mock_load_s3.assert_called_once()
        mock_save_s3.assert_called_once()
    
    @patch('src.data_processing.boto3.client')
    def test_save_to_s3(self, mock_boto3_client):
        """Test saving processed data to S3."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        
        # Save data
        self.processor._save_to_s3(self.sample_df, 'test_output.parquet')
        
        # Assertions
        mock_s3.put_object.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in data processing."""
        # Test with invalid data
        with self.assertRaises(DataProcessingError):
            self.processor._extract_features({'invalid': 'data'})
        
        # Test with invalid DataFrame
        with self.assertRaises(Exception):
            self.processor._normalize_features(pd.DataFrame())

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
from datetime import datetime

from src.data_ingestion import SpotifyDataCollector, SpotifyAPIError, DataIngestionError

class TestSpotifyDataCollector(unittest.TestCase):
    """Test cases for SpotifyDataCollector class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.collector = SpotifyDataCollector()
        
        # Mock Spotify API response data
        self.mock_playlist_response = {
            'playlists': {
                'items': [
                    {'id': 'playlist1'},
                    {'id': 'playlist2'}
                ]
            }
        }
        
        self.mock_tracks_response = {
            'items': [
                {
                    'track': {
                        'id': 'track1',
                        'name': 'Test Track 1',
                        'artists': [{'name': 'Test Artist'}],
                        'album': {
                            'name': 'Test Album',
                            'release_date': '2023-01-01'
                        },
                        'popularity': 80,
                        'duration_ms': 200000,
                        'explicit': False
                    }
                }
            ],
            'next': None
        }
        
        self.mock_audio_features_response = {
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
                    'duration_ms': 200000,
                    'time_signature': 4
                }
            ]
        }
    
    @patch('src.data_ingestion.requests.post')
    def test_get_access_token(self, mock_post):
        """Test getting Spotify access token."""
        # Mock the token response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'access_token': 'test_token',
            'expires_in': 3600
        }
        mock_post.return_value = mock_response
        
        # Get token
        token = self.collector._get_access_token()
        
        # Assertions
        self.assertEqual(token, 'test_token')
        mock_post.assert_called_once()
    
    @patch('src.data_ingestion.requests.post')
    def test_get_access_token_failure(self, mock_post):
        """Test handling of access token request failure."""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = 'Invalid client'
        mock_post.return_value = mock_response
        
        # Assert error is raised
        with self.assertRaises(SpotifyAPIError):
            self.collector._get_access_token()
    
    @patch('src.data_ingestion.requests.get')
    def test_make_api_request(self, mock_get):
        """Test making API requests."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_get.return_value = mock_response
        
        # Make request
        result = self.collector._make_api_request('test_endpoint')
        
        # Assertions
        self.assertEqual(result, {'data': 'test'})
        mock_get.assert_called_once()
    
    @patch('src.data_ingestion.requests.get')
    def test_get_featured_playlists(self, mock_get):
        """Test getting featured playlists."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_playlist_response
        mock_get.return_value = mock_response
        
        # Get playlists
        playlists = self.collector.get_featured_playlists()
        
        # Assertions
        self.assertEqual(len(playlists), 2)
        self.assertEqual(playlists, ['playlist1', 'playlist2'])
    
    @patch('src.data_ingestion.requests.get')
    def test_get_playlist_tracks(self, mock_get):
        """Test getting tracks from a playlist."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_tracks_response
        mock_get.return_value = mock_response
        
        # Get tracks
        tracks = self.collector.get_playlist_tracks('playlist1')
        
        # Assertions
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['id'], 'track1')
    
    @patch('src.data_ingestion.requests.get')
    def test_get_audio_features(self, mock_get):
        """Test getting audio features for tracks."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_audio_features_response
        mock_get.return_value = mock_response
        
        # Get audio features
        features = self.collector.get_audio_features(['track1'])
        
        # Assertions
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]['id'], 'track1')
    
    @patch('src.data_ingestion.SpotifyDataCollector.get_featured_playlists')
    @patch('src.data_ingestion.SpotifyDataCollector.get_playlist_tracks')
    @patch('src.data_ingestion.SpotifyDataCollector.get_audio_features')
    @patch('src.data_ingestion.SpotifyDataCollector.save_to_s3')
    def test_collect_and_save_data(
        self,
        mock_save_s3,
        mock_features,
        mock_tracks,
        mock_playlists
    ):
        """Test the complete data collection process."""
        # Mock responses
        mock_playlists.return_value = ['playlist1']
        mock_tracks.return_value = [
            self.mock_tracks_response['items'][0]['track']
        ]
        mock_features.return_value = [
            self.mock_audio_features_response['audio_features'][0]
        ]
        
        # Run collection
        self.collector.collect_and_save_data()
        
        # Assertions
        mock_playlists.assert_called_once()
        mock_tracks.assert_called_once()
        mock_features.assert_called_once()
        mock_save_s3.assert_called_once()
    
    def test_save_to_s3(self):
        """Test saving data to S3."""
        # Mock S3 client
        self.collector.s3_client = MagicMock()
        
        # Test data
        test_data = {'test': 'data'}
        
        # Save data
        self.collector.save_to_s3(test_data, 'test_file.json')
        
        # Assertions
        self.collector.s3_client.put_object.assert_called_once()

if __name__ == '__main__':
    unittest.main()

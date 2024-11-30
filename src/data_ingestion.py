import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import requests
import boto3
from botocore.exceptions import ClientError
from .config import config

class SpotifyAPIError(Exception):
    """Custom exception for Spotify API related errors."""
    pass

class DataIngestionError(Exception):
    """Custom exception for data ingestion related errors."""
    pass

class SpotifyDataCollector:
    """Class to handle data collection from Spotify API."""
    
    def __init__(self):
        self.logger = logging.getLogger('spotify_pipeline.data_ingestion')
        self.spotify_credentials = config.get_spotify_credentials()
        self.aws_credentials = config.get_aws_credentials()
        self.settings = config.get_data_collection_settings()
        
        # Initialize AWS S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_credentials['access_key_id'],
            aws_secret_access_key=self.aws_credentials['secret_access_key'],
            region_name=self.aws_credentials['region']
        )
        
        self.access_token = None
        self.token_expiry = 0
    
    def _get_access_token(self) -> str:
        """Get or refresh Spotify API access token."""
        current_time = time.time()
        
        # Check if token is still valid
        if self.access_token and current_time < self.token_expiry:
            return self.access_token
        
        # Request new token
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_response = requests.post(
            auth_url,
            data={'grant_type': 'client_credentials'},
            auth=(
                self.spotify_credentials['client_id'],
                self.spotify_credentials['client_secret']
            )
        )
        
        if auth_response.status_code != 200:
            error_msg = f"Failed to get access token: {auth_response.text}"
            self.logger.error(error_msg)
            raise SpotifyAPIError(error_msg)
        
        auth_data = auth_response.json()
        self.access_token = auth_data['access_token']
        self.token_expiry = current_time + auth_data['expires_in']
        
        return self.access_token
    
    def _make_api_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Make a request to the Spotify API with retry logic."""
        base_url = 'https://api.spotify.com/v1'
        headers = {'Authorization': f'Bearer {self._get_access_token()}'}
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{base_url}/{endpoint}",
                    headers=headers,
                    params=params
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    # Token might be expired, get a new one
                    self.access_token = None
                    headers['Authorization'] = f'Bearer {self._get_access_token()}'
                    continue
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    retry_after = int(response.headers.get('Retry-After', 1))
                    time.sleep(retry_after)
                    continue
                else:
                    error_msg = f"API request failed: {response.text}"
                    self.logger.error(error_msg)
                    raise SpotifyAPIError(error_msg)
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    error_msg = f"Failed to make API request after {max_retries} attempts: {str(e)}"
                    self.logger.error(error_msg)
                    raise SpotifyAPIError(error_msg)
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_featured_playlists(self) -> List[str]:
        """Get a list of featured playlist IDs."""
        response = self._make_api_request(
            'browse/featured-playlists',
            params={'limit': 50}
        )
        
        playlist_ids = [
            playlist['id']
            for playlist in response['playlists']['items']
        ]
        
        self.logger.info(f"Retrieved {len(playlist_ids)} featured playlists")
        return playlist_ids
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Get all tracks from a playlist."""
        tracks = []
        offset = 0
        
        while True:
            response = self._make_api_request(
                f'playlists/{playlist_id}/tracks',
                params={
                    'offset': offset,
                    'limit': self.settings['batch_size'],
                    'fields': 'items(track(id,name,artists,album,popularity,duration_ms,'
                             'explicit,external_urls)),next'
                }
            )
            
            # Extract track data
            tracks.extend([
                item['track'] for item in response['items']
                if item['track'] is not None  # Filter out None tracks
            ])
            
            # Check if there are more tracks
            if not response.get('next'):
                break
                
            offset += self.settings['batch_size']
        
        self.logger.info(f"Retrieved {len(tracks)} tracks from playlist {playlist_id}")
        return tracks
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Get audio features for a list of tracks."""
        audio_features = []
        
        # Process in batches of 100 (Spotify API limit)
        for i in range(0, len(track_ids), 100):
            batch_ids = track_ids[i:i + 100]
            response = self._make_api_request(
                'audio-features',
                params={'ids': ','.join(batch_ids)}
            )
            audio_features.extend(response['audio_features'])
        
        return audio_features
    
    def save_to_s3(self, data: Dict[str, Any], file_name: str) -> None:
        """Save data to S3 bucket."""
        try:
            bucket_name = self.aws_credentials['bucket_name']
            key = f"{self.settings['raw_data_prefix']}/{file_name}"
            
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=json.dumps(data),
                ContentType='application/json'
            )
            
            self.logger.info(f"Successfully saved {file_name} to S3")
            
        except ClientError as e:
            error_msg = f"Failed to save data to S3: {str(e)}"
            self.logger.error(error_msg)
            raise DataIngestionError(error_msg)
    
    def collect_and_save_data(self) -> None:
        """Main method to collect and save data."""
        try:
            # Create timestamp for file naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Get featured playlists
            playlist_ids = self.get_featured_playlists()
            
            all_tracks = []
            all_audio_features = []
            
            # Collect tracks and audio features
            for playlist_id in playlist_ids:
                tracks = self.get_playlist_tracks(playlist_id)
                track_ids = [track['id'] for track in tracks]
                audio_features = self.get_audio_features(track_ids)
                
                all_tracks.extend(tracks)
                all_audio_features.extend(audio_features)
            
            # Combine track data with audio features
            combined_data = {
                'metadata': {
                    'timestamp': timestamp,
                    'total_tracks': len(all_tracks)
                },
                'tracks': all_tracks,
                'audio_features': all_audio_features
            }
            
            # Save to S3
            self.save_to_s3(combined_data, f'spotify_data_{timestamp}.json')
            
            self.logger.info(
                f"Successfully collected and saved data for {len(all_tracks)} tracks"
            )
            
        except (SpotifyAPIError, DataIngestionError) as e:
            self.logger.error(f"Data collection failed: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error during data collection: {str(e)}"
            self.logger.error(error_msg)
            raise DataIngestionError(error_msg)

def main():
    """Main function to run data ingestion."""
    try:
        collector = SpotifyDataCollector()
        collector.collect_and_save_data()
    except Exception as e:
        logging.error(f"Data ingestion failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()

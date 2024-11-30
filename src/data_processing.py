import os
import json
from typing import Dict, List, Any, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .config import config

class DataProcessingError(Exception):
    """Custom exception for data processing related errors."""
    pass

class SpotifyDataProcessor:
    """Class to handle processing of Spotify track data."""
    
    def __init__(self):
        self.logger = logging.getLogger('spotify_pipeline.data_processing')
        self.aws_credentials = config.get_aws_credentials()
        self.settings = config.get_setting('data_processing')
        
        # Initialize AWS S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_credentials['access_key_id'],
            aws_secret_access_key=self.aws_credentials['secret_access_key'],
            region_name=self.aws_credentials['region']
        )
        
        self.bucket_name = self.aws_credentials['bucket_name']
        self.scaler = StandardScaler()
    
    def _load_data_from_s3(self, file_key: str) -> Dict[str, Any]:
        """Load JSON data from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            return json.loads(response['Body'].read().decode('utf-8'))
            
        except ClientError as e:
            error_msg = f"Failed to load data from S3: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg)
    
    def _save_to_s3(self, data: pd.DataFrame, file_name: str) -> None:
        """Save processed data to S3."""
        try:
            # Save as parquet file for efficient storage and retrieval
            parquet_buffer = data.to_parquet()
            key = f"{self.settings['processed_data_prefix']}/{file_name}"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=parquet_buffer
            )
            
            self.logger.info(f"Successfully saved processed data to {key}")
            
        except Exception as e:
            error_msg = f"Failed to save processed data to S3: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg)
    
    def _extract_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract and combine features from raw data."""
        tracks_df = pd.DataFrame(raw_data['tracks'])
        audio_features_df = pd.DataFrame(raw_data['audio_features'])
        
        # Extract artist and album information
        tracks_df['artist_names'] = tracks_df['artists'].apply(
            lambda x: ', '.join([artist['name'] for artist in x])
        )
        tracks_df['artist_count'] = tracks_df['artists'].apply(len)
        tracks_df['album_name'] = tracks_df['album'].apply(lambda x: x['name'])
        tracks_df['album_type'] = tracks_df['album'].apply(lambda x: x['album_type'])
        
        # Extract release date
        tracks_df['release_date'] = tracks_df['album'].apply(
            lambda x: x.get('release_date', None)
        )
        
        # Drop unnecessary columns
        tracks_df = tracks_df.drop(['artists', 'album', 'external_urls'], axis=1)
        
        # Merge with audio features
        merged_df = pd.merge(
            tracks_df,
            audio_features_df,
            left_on='id',
            right_on='id',
            how='inner'
        )
        
        return merged_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Check missing values
        missing_values = df.isnull().sum()
        self.logger.info(f"Missing values before handling:\n{missing_values}")
        
        strategy = self.settings['handle_missing_values']
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        elif strategy == 'median':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        elif strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        
        # Fill remaining categorical columns with 'unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('unknown')
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from release date."""
        if self.settings['feature_engineering']['create_year_feature']:
            df['release_year'] = pd.to_datetime(df['release_date']).dt.year
        
        if self.settings['feature_engineering']['create_month_feature']:
            df['release_month'] = pd.to_datetime(df['release_date']).dt.month
        
        if self.settings['feature_engineering']['create_day_feature']:
            df['release_day'] = pd.to_datetime(df['release_date']).dt.day
        
        # Drop original release_date column
        df = df.drop('release_date', axis=1)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        if not self.settings['normalize_features']:
            return df
        
        # Select numerical columns for normalization
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Exclude certain columns from normalization
        exclude_columns = ['id', 'popularity', 'release_year', 'release_month', 'release_day']
        normalize_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Normalize selected features
        df[normalize_columns] = self.scaler.fit_transform(df[normalize_columns])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        # One-hot encode album_type
        df = pd.get_dummies(df, columns=['album_type'], prefix='album_type')
        
        # Handle artist names (get top N most common artists)
        top_artists = df['artist_names'].value_counts().nlargest(50).index
        for artist in top_artists:
            df[f'artist_{artist}'] = df['artist_names'].str.contains(artist).astype(int)
        
        # Drop original artist_names column
        df = df.drop('artist_names', axis=1)
        
        return df
    
    def process_data(self, input_file_key: str) -> None:
        """Main method to process data."""
        try:
            # Load raw data
            raw_data = self._load_data_from_s3(input_file_key)
            
            # Extract features
            df = self._extract_features(raw_data)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Create time features
            df = self._create_time_features(df)
            
            # Remove duplicates if specified
            if self.settings['remove_duplicates']:
                initial_rows = len(df)
                df = df.drop_duplicates()
                dropped_rows = initial_rows - len(df)
                self.logger.info(f"Removed {dropped_rows} duplicate rows")
            
            # Normalize features
            df = self._normalize_features(df)
            
            # Encode categorical features
            df = self._encode_categorical_features(df)
            
            # Save processed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file_name = f'processed_data_{timestamp}.parquet'
            self._save_to_s3(df, output_file_name)
            
            self.logger.info(
                f"Data processing completed successfully. "
                f"Processed {len(df)} rows with {len(df.columns)} features."
            )
            
        except Exception as e:
            error_msg = f"Data processing failed: {str(e)}"
            self.logger.error(error_msg)
            raise DataProcessingError(error_msg)

def main():
    """Main function to run data processing."""
    try:
        processor = SpotifyDataProcessor()
        
        # Get latest raw data file from S3
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=processor.bucket_name,
            Prefix=config.get_setting('data_collection', 'raw_data_prefix')
        )
        
        if 'Contents' not in response:
            raise DataProcessingError("No raw data files found in S3")
        
        # Sort by last modified and get the latest file
        latest_file = sorted(
            response['Contents'],
            key=lambda x: x['LastModified'],
            reverse=True
        )[0]
        
        # Process the latest file
        processor.process_data(latest_file['Key'])
        
    except Exception as e:
        logging.error(f"Data processing failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()

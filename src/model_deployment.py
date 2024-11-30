import os
import json
import logging
from typing import Dict, List, Any, Union
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError
import joblib

from .config import config

class ModelDeploymentError(Exception):
    """Custom exception for model deployment related errors."""
    pass

class SpotifyModelDeployer:
    """Class to handle deployment and prediction using trained models."""
    
    def __init__(self):
        self.logger = logging.getLogger('spotify_pipeline.model_deployment')
        self.aws_credentials = config.get_aws_credentials()
        self.settings = config.get_setting('model_training')  # Reuse relevant settings
        
        # Initialize AWS S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_credentials['access_key_id'],
            aws_secret_access_key=self.aws_credentials['secret_access_key'],
            region_name=self.aws_credentials['region']
        )
        
        self.bucket_name = self.aws_credentials['bucket_name']
        self.model = None
        self.model_name = None
    
    def _get_latest_model(self) -> str:
        """Get the latest model file from S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='models/'
            )
            
            if 'Contents' not in response:
                raise ModelDeploymentError("No model files found in S3")
            
            # Get the latest model file
            latest_model = sorted(
                response['Contents'],
                key=lambda x: x['LastModified'],
                reverse=True
            )[0]
            
            return latest_model['Key']
            
        except ClientError as e:
            error_msg = f"Failed to get latest model from S3: {str(e)}"
            self.logger.error(error_msg)
            raise ModelDeploymentError(error_msg)
    
    def _load_model_from_s3(self, model_key: str) -> None:
        """Load model from S3."""
        try:
            # Download model file
            local_path = f"/tmp/{os.path.basename(model_key)}"
            self.s3_client.download_file(
                self.bucket_name,
                model_key,
                local_path
            )
            
            # Load model
            self.model = joblib.load(local_path)
            self.model_name = os.path.basename(model_key).split('_')[0]
            
            # Clean up
            os.remove(local_path)
            
            self.logger.info(f"Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            error_msg = f"Failed to load model from S3: {str(e)}"
            self.logger.error(error_msg)
            raise ModelDeploymentError(error_msg)
    
    def _prepare_features(
        self,
        track_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """Prepare features for prediction."""
        try:
            # Create DataFrame with single row
            df = pd.DataFrame([track_data])
            
            # Ensure all required features are present
            required_features = [
                'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms',
                'time_signature', 'explicit'
            ]
            
            for feature in required_features:
                if feature not in df.columns:
                    df[feature] = 0  # Default value
            
            # Handle categorical features
            if 'album_type' in df.columns:
                df = pd.get_dummies(df, columns=['album_type'], prefix='album_type')
            
            # Add missing columns that were present during training
            # This would need to be updated based on your actual training features
            for col in self.model.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            
            # Ensure columns are in the same order as during training
            df = df[self.model.feature_names_in_]
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to prepare features: {str(e)}"
            self.logger.error(error_msg)
            raise ModelDeploymentError(error_msg)
    
    def _save_prediction_to_s3(
        self,
        track_id: str,
        prediction: float,
        actual_popularity: Union[float, None]
    ) -> None:
        """Save prediction results to S3."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            prediction_data = {
                'track_id': track_id,
                'timestamp': timestamp,
                'model_name': self.model_name,
                'predicted_popularity': float(prediction),
                'actual_popularity': actual_popularity,
                'error': None if actual_popularity is None else float(actual_popularity - prediction)
            }
            
            key = f"predictions/{track_id}_{timestamp}.json"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(prediction_data, indent=2),
                ContentType='application/json'
            )
            
            self.logger.info(f"Saved prediction to {key}")
            
        except Exception as e:
            error_msg = f"Failed to save prediction: {str(e)}"
            self.logger.error(error_msg)
            raise ModelDeploymentError(error_msg)
    
    def load_model(self, model_key: str = None) -> None:
        """Load the model for deployment."""
        try:
            if model_key is None:
                model_key = self._get_latest_model()
            
            self._load_model_from_s3(model_key)
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.logger.error(error_msg)
            raise ModelDeploymentError(error_msg)
    
    def predict(
        self,
        track_data: Dict[str, Any],
        actual_popularity: Union[float, None] = None
    ) -> float:
        """Make popularity prediction for a track."""
        try:
            if self.model is None:
                self.load_model()
            
            # Prepare features
            features_df = self._prepare_features(track_data)
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            
            # Save prediction
            self._save_prediction_to_s3(
                track_data.get('id', 'unknown'),
                prediction,
                actual_popularity
            )
            
            return float(prediction)
            
        except Exception as e:
            error_msg = f"Failed to make prediction: {str(e)}"
            self.logger.error(error_msg)
            raise ModelDeploymentError(error_msg)
    
    def batch_predict(
        self,
        tracks_data: List[Dict[str, Any]]
    ) -> List[float]:
        """Make predictions for multiple tracks."""
        try:
            predictions = []
            
            for track_data in tracks_data:
                prediction = self.predict(
                    track_data,
                    actual_popularity=track_data.get('popularity')
                )
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            error_msg = f"Failed to make batch predictions: {str(e)}"
            self.logger.error(error_msg)
            raise ModelDeploymentError(error_msg)

def main():
    """Main function to demonstrate model deployment."""
    try:
        deployer = SpotifyModelDeployer()
        
        # Example track data
        example_track = {
            'id': 'example_track_id',
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
            'time_signature': 4,
            'explicit': 0,
            'album_type': 'album'
        }
        
        # Make prediction
        prediction = deployer.predict(example_track)
        print(f"Predicted popularity: {prediction:.2f}")
        
    except Exception as e:
        logging.error(f"Model deployment failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()

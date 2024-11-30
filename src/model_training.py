import os
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from .config import config

class ModelTrainingError(Exception):
    """Custom exception for model training related errors."""
    pass

class SpotifyModelTrainer:
    """Class to handle training of models for Spotify song popularity prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger('spotify_pipeline.model_training')
        self.aws_credentials = config.get_aws_credentials()
        self.settings = config.get_model_training_settings()
        
        # Initialize AWS S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_credentials['access_key_id'],
            aws_secret_access_key=self.aws_credentials['secret_access_key'],
            region_name=self.aws_credentials['region']
        )
        
        self.bucket_name = self.aws_credentials['bucket_name']
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('-inf')
    
    def _load_data_from_s3(self, file_key: str) -> pd.DataFrame:
        """Load processed data from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            return pd.read_parquet(response['Body'])
            
        except ClientError as e:
            error_msg = f"Failed to load data from S3: {str(e)}"
            self.logger.error(error_msg)
            raise ModelTrainingError(error_msg)
    
    def _save_model_to_s3(self, model: Any, model_name: str) -> None:
        """Save trained model to S3."""
        try:
            # Save model locally first
            local_path = f"/tmp/{model_name}.joblib"
            joblib.dump(model, local_path)
            
            # Upload to S3
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            key = f"models/{model_name}_{timestamp}.joblib"
            
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                key
            )
            
            # Clean up local file
            os.remove(local_path)
            
            self.logger.info(f"Successfully saved model to {key}")
            
        except Exception as e:
            error_msg = f"Failed to save model to S3: {str(e)}"
            self.logger.error(error_msg)
            raise ModelTrainingError(error_msg)
    
    def _save_metrics_to_s3(self, metrics: Dict[str, Any]) -> None:
        """Save training metrics to S3."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            key = f"metrics/training_metrics_{timestamp}.json"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(metrics, indent=2),
                ContentType='application/json'
            )
            
            self.logger.info(f"Successfully saved metrics to {key}")
            
        except ClientError as e:
            error_msg = f"Failed to save metrics to S3: {str(e)}"
            self.logger.error(error_msg)
            raise ModelTrainingError(error_msg)
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Separate features and target
        X = df.drop(['id', 'name', 'album_name', 'popularity'], axis=1)
        y = df['popularity']
        
        # Convert to numpy arrays
        X = X.to_numpy()
        y = y.to_numpy()
        
        return X, y
    
    def _evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=self.settings['cross_validation_folds'],
            scoring='r2'
        )
        
        metrics['cv_mean_r2'] = cv_scores.mean()
        metrics['cv_std_r2'] = cv_scores.std()
        
        return metrics
    
    def _plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str
    ) -> None:
        """Plot feature importance."""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title(f'Feature Importances ({model_name})')
            plt.bar(range(len(indices[:20])), importances[indices[:20]])
            plt.xticks(
                range(len(indices[:20])),
                [feature_names[i] for i in indices[:20]],
                rotation=45,
                ha='right'
            )
            
            # Save plot to S3
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f"/tmp/feature_importance_{timestamp}.png"
            plt.savefig(plot_path)
            
            key = f"plots/feature_importance_{model_name}_{timestamp}.png"
            self.s3_client.upload_file(plot_path, self.bucket_name, key)
            
            os.remove(plot_path)
            plt.close()
    
    def train_models(self) -> None:
        """Main method to train models."""
        try:
            # Get latest processed data file from S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.settings['processed_data_prefix']
            )
            
            if 'Contents' not in response:
                raise ModelTrainingError("No processed data files found in S3")
            
            latest_file = sorted(
                response['Contents'],
                key=lambda x: x['LastModified'],
                reverse=True
            )[0]
            
            # Load and prepare data
            df = self._load_data_from_s3(latest_file['Key'])
            X, y = self._prepare_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.settings['test_size'],
                random_state=self.settings['random_state']
            )
            
            # Train models
            models_config = self.settings['models']
            all_metrics = {}
            
            for model_config in models_config:
                if not model_config['enabled']:
                    continue
                
                model_name = model_config['name']
                self.logger.info(f"Training {model_name}...")
                
                # Initialize model
                if model_name == 'decision_tree':
                    model = DecisionTreeRegressor(**model_config['params'])
                elif model_name == 'random_forest':
                    model = RandomForestRegressor(**model_config['params'])
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingRegressor(**model_config['params'])
                else:
                    self.logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                metrics = self._evaluate_model(
                    model, X_train, X_test, y_train, y_test
                )
                all_metrics[model_name] = metrics
                
                # Plot feature importance
                feature_names = df.drop(
                    ['id', 'name', 'album_name', 'popularity'],
                    axis=1
                ).columns.tolist()
                self._plot_feature_importance(model, feature_names, model_name)
                
                # Save model
                self._save_model_to_s3(model, model_name)
                
                # Track best model
                if metrics['test_r2'] > self.best_score:
                    self.best_score = metrics['test_r2']
                    self.best_model = model
                    self.best_model_name = model_name
            
            # Save all metrics
            self._save_metrics_to_s3(all_metrics)
            
            self.logger.info(
                f"Model training completed. Best model: {self.best_model_name} "
                f"with R² score: {self.best_score:.4f}"
            )
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            self.logger.error(error_msg)
            raise ModelTrainingError(error_msg)

def main():
    """Main function to run model training."""
    try:
        trainer = SpotifyModelTrainer()
        trainer.train_models()
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()

import os
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime

import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from .config import config

class VisualizationError(Exception):
    """Custom exception for visualization related errors."""
    pass

class SpotifyDataVisualizer:
    """Class to handle visualization of Spotify data and model results."""
    
    def __init__(self):
        self.logger = logging.getLogger('spotify_pipeline.visualization')
        self.aws_credentials = config.get_aws_credentials()
        
        # Initialize AWS S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_credentials['access_key_id'],
            aws_secret_access_key=self.aws_credentials['secret_access_key'],
            region_name=self.aws_credentials['region']
        )
        
        self.bucket_name = self.aws_credentials['bucket_name']
        
        # Set style for all plots
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def _load_data_from_s3(self, file_key: str) -> pd.DataFrame:
        """Load data from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            
            if file_key.endswith('.parquet'):
                return pd.read_parquet(response['Body'])
            elif file_key.endswith('.json'):
                return pd.read_json(response['Body'])
            else:
                raise VisualizationError(f"Unsupported file format: {file_key}")
            
        except ClientError as e:
            error_msg = f"Failed to load data from S3: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def _save_plot_to_s3(self, plot_name: str) -> None:
        """Save plot to S3."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            local_path = f"/tmp/{plot_name}_{timestamp}.png"
            
            # Save plot locally
            plt.savefig(local_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Upload to S3
            key = f"visualizations/{plot_name}_{timestamp}.png"
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                key
            )
            
            # Clean up local file
            os.remove(local_path)
            
            self.logger.info(f"Successfully saved plot to {key}")
            
        except Exception as e:
            error_msg = f"Failed to save plot: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def plot_feature_distributions(self, data: pd.DataFrame) -> None:
        """Plot distributions of numerical features."""
        try:
            numerical_features = data.select_dtypes(include=[np.number]).columns
            
            for feature in numerical_features:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=data, x=feature, kde=True)
                plt.title(f'Distribution of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Count')
                
                self._save_plot_to_s3(f'distribution_{feature}')
            
        except Exception as e:
            error_msg = f"Failed to plot feature distributions: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def plot_correlation_matrix(self, data: pd.DataFrame) -> None:
        """Plot correlation matrix of numerical features."""
        try:
            numerical_data = data.select_dtypes(include=[np.number])
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                numerical_data.corr(),
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f'
            )
            plt.title('Feature Correlation Matrix')
            
            self._save_plot_to_s3('correlation_matrix')
            
        except Exception as e:
            error_msg = f"Failed to plot correlation matrix: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def plot_popularity_vs_features(self, data: pd.DataFrame) -> None:
        """Plot relationships between features and popularity."""
        try:
            numerical_features = [
                col for col in data.select_dtypes(include=[np.number]).columns
                if col != 'popularity'
            ]
            
            for feature in numerical_features:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=data, x=feature, y='popularity', alpha=0.5)
                plt.title(f'Popularity vs {feature}')
                
                # Add trend line
                z = np.polyfit(data[feature], data['popularity'], 1)
                p = np.poly1d(z)
                plt.plot(data[feature], p(data[feature]), "r--", alpha=0.8)
                
                self._save_plot_to_s3(f'popularity_vs_{feature}')
            
        except Exception as e:
            error_msg = f"Failed to plot popularity relationships: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def plot_prediction_analysis(
        self,
        predictions_file: str
    ) -> None:
        """Plot analysis of model predictions."""
        try:
            # Load predictions
            predictions_data = self._load_data_from_s3(predictions_file)
            
            # Scatter plot of predicted vs actual
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=predictions_data,
                x='actual_popularity',
                y='predicted_popularity',
                alpha=0.5
            )
            
            # Add perfect prediction line
            min_val = min(
                predictions_data['actual_popularity'].min(),
                predictions_data['predicted_popularity'].min()
            )
            max_val = max(
                predictions_data['actual_popularity'].max(),
                predictions_data['predicted_popularity'].max()
            )
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            plt.title('Predicted vs Actual Popularity')
            plt.xlabel('Actual Popularity')
            plt.ylabel('Predicted Popularity')
            
            self._save_plot_to_s3('predicted_vs_actual')
            
            # Error distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=predictions_data['error'], kde=True)
            plt.title('Prediction Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Count')
            
            self._save_plot_to_s3('error_distribution')
            
        except Exception as e:
            error_msg = f"Failed to plot prediction analysis: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def plot_feature_importance(
        self,
        feature_importance: pd.Series,
        title: str = 'Feature Importance'
    ) -> None:
        """Plot feature importance."""
        try:
            plt.figure(figsize=(12, 6))
            feature_importance.sort_values(ascending=True).plot(
                kind='barh',
                title=title
            )
            plt.xlabel('Importance Score')
            
            self._save_plot_to_s3('feature_importance')
            
        except Exception as e:
            error_msg = f"Failed to plot feature importance: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def plot_popularity_trends(self, data: pd.DataFrame) -> None:
        """Plot popularity trends over time."""
        try:
            # Ensure release_date is datetime
            data['release_date'] = pd.to_datetime(data['release_date'])
            
            # Group by release date and calculate mean popularity
            popularity_trend = data.groupby(
                data['release_date'].dt.to_period('M')
            )['popularity'].mean()
            
            plt.figure(figsize=(15, 6))
            popularity_trend.plot(kind='line', marker='o')
            plt.title('Average Song Popularity Over Time')
            plt.xlabel('Release Date')
            plt.ylabel('Average Popularity')
            plt.xticks(rotation=45)
            
            self._save_plot_to_s3('popularity_trend')
            
        except Exception as e:
            error_msg = f"Failed to plot popularity trends: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def create_dashboard(self, data: pd.DataFrame) -> None:
        """Create a comprehensive dashboard of visualizations."""
        try:
            # Create a multi-panel figure
            fig = plt.figure(figsize=(20, 15))
            fig.suptitle('Spotify Data Analysis Dashboard', size=16)
            
            # 1. Popularity Distribution
            ax1 = plt.subplot(321)
            sns.histplot(data=data, x='popularity', kde=True, ax=ax1)
            ax1.set_title('Popularity Distribution')
            
            # 2. Top Features Correlation with Popularity
            correlations = data.corr()['popularity'].sort_values(ascending=False)
            ax2 = plt.subplot(322)
            correlations[1:11].plot(kind='bar', ax=ax2)
            ax2.set_title('Top 10 Feature Correlations with Popularity')
            plt.xticks(rotation=45)
            
            # 3. Popularity vs Danceability
            ax3 = plt.subplot(323)
            sns.scatterplot(
                data=data,
                x='danceability',
                y='popularity',
                alpha=0.5,
                ax=ax3
            )
            ax3.set_title('Popularity vs Danceability')
            
            # 4. Popularity vs Energy
            ax4 = plt.subplot(324)
            sns.scatterplot(
                data=data,
                x='energy',
                y='popularity',
                alpha=0.5,
                ax=ax4
            )
            ax4.set_title('Popularity vs Energy')
            
            # 5. Average Popularity by Genre
            if 'genre' in data.columns:
                ax5 = plt.subplot(325)
                data.groupby('genre')['popularity'].mean().sort_values(
                    ascending=False
                )[:10].plot(kind='bar', ax=ax5)
                ax5.set_title('Average Popularity by Top Genres')
                plt.xticks(rotation=45)
            
            # 6. Popularity Over Time
            ax6 = plt.subplot(326)
            if 'release_date' in data.columns:
                data['release_date'] = pd.to_datetime(data['release_date'])
                data.groupby(
                    data['release_date'].dt.to_period('M')
                )['popularity'].mean().plot(ax=ax6)
                ax6.set_title('Popularity Trend Over Time')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            self._save_plot_to_s3('dashboard')
            
        except Exception as e:
            error_msg = f"Failed to create dashboard: {str(e)}"
            self.logger.error(error_msg)
            raise VisualizationError(error_msg)

def main():
    """Main function to create visualizations."""
    try:
        visualizer = SpotifyDataVisualizer()
        
        # Get latest processed data file from S3
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=visualizer.bucket_name,
            Prefix='processed/'
        )
        
        if 'Contents' not in response:
            raise VisualizationError("No processed data files found in S3")
        
        latest_file = sorted(
            response['Contents'],
            key=lambda x: x['LastModified'],
            reverse=True
        )[0]
        
        # Load data
        data = visualizer._load_data_from_s3(latest_file['Key'])
        
        # Create visualizations
        visualizer.plot_feature_distributions(data)
        visualizer.plot_correlation_matrix(data)
        visualizer.plot_popularity_vs_features(data)
        visualizer.plot_popularity_trends(data)
        visualizer.create_dashboard(data)
        
        # If predictions exist, analyze them
        try:
            predictions_response = s3_client.list_objects_v2(
                Bucket=visualizer.bucket_name,
                Prefix='predictions/'
            )
            
            if 'Contents' in predictions_response:
                latest_predictions = sorted(
                    predictions_response['Contents'],
                    key=lambda x: x['LastModified'],
                    reverse=True
                )[0]
                visualizer.plot_prediction_analysis(latest_predictions['Key'])
        except Exception as e:
            visualizer.logger.warning(
                f"Could not analyze predictions: {str(e)}"
            )
        
    except Exception as e:
        logging.error(f"Visualization creation failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()

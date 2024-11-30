import os
import logging
import argparse
from datetime import datetime
from typing import List, Optional

from .config import config
from .data_ingestion import SpotifyDataCollector
from .data_processing import SpotifyDataProcessor
from .model_training import SpotifyModelTrainer
from .model_deployment import SpotifyModelDeployer
from .visualization import SpotifyDataVisualizer

class PipelineError(Exception):
    """Custom exception for pipeline related errors."""
    pass

class SpotifyPipeline:
    """Main class to orchestrate the Spotify data pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger('spotify_pipeline.main')
        self.collector = SpotifyDataCollector()
        self.processor = SpotifyDataProcessor()
        self.trainer = SpotifyModelTrainer()
        self.deployer = SpotifyModelDeployer()
        self.visualizer = SpotifyDataVisualizer()
    
    def run_data_collection(self) -> None:
        """Run the data collection step."""
        try:
            self.logger.info("Starting data collection...")
            self.collector.collect_and_save_data()
            self.logger.info("Data collection completed successfully")
            
        except Exception as e:
            error_msg = f"Data collection failed: {str(e)}"
            self.logger.error(error_msg)
            raise PipelineError(error_msg)
    
    def run_data_processing(self) -> None:
        """Run the data processing step."""
        try:
            self.logger.info("Starting data processing...")
            self.processor.process_data()
            self.logger.info("Data processing completed successfully")
            
        except Exception as e:
            error_msg = f"Data processing failed: {str(e)}"
            self.logger.error(error_msg)
            raise PipelineError(error_msg)
    
    def run_model_training(self) -> None:
        """Run the model training step."""
        try:
            self.logger.info("Starting model training...")
            self.trainer.train_models()
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            self.logger.error(error_msg)
            raise PipelineError(error_msg)
    
    def run_model_deployment(self) -> None:
        """Run the model deployment step."""
        try:
            self.logger.info("Starting model deployment...")
            self.deployer.load_model()
            self.logger.info("Model deployment completed successfully")
            
        except Exception as e:
            error_msg = f"Model deployment failed: {str(e)}"
            self.logger.error(error_msg)
            raise PipelineError(error_msg)
    
    def run_visualization(self) -> None:
        """Run the visualization step."""
        try:
            self.logger.info("Starting visualization creation...")
            self.visualizer.create_dashboard()
            self.logger.info("Visualization creation completed successfully")
            
        except Exception as e:
            error_msg = f"Visualization creation failed: {str(e)}"
            self.logger.error(error_msg)
            raise PipelineError(error_msg)
    
    def run_full_pipeline(self) -> None:
        """Run the complete pipeline."""
        try:
            self.logger.info("Starting full pipeline execution...")
            
            # Run each step
            self.run_data_collection()
            self.run_data_processing()
            self.run_model_training()
            self.run_model_deployment()
            self.run_visualization()
            
            self.logger.info("Full pipeline execution completed successfully")
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)
            raise PipelineError(error_msg)

def setup_logging() -> None:
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/pipeline_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Spotify Data Pipeline - Predict Song Popularity'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=[
            'collect',
            'process',
            'train',
            'deploy',
            'visualize',
            'full'
        ],
        default='full',
        help='Specify which pipeline step to run (default: full)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging()
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize pipeline
        pipeline = SpotifyPipeline()
        
        # Run specified step
        if args.step == 'collect':
            pipeline.run_data_collection()
        elif args.step == 'process':
            pipeline.run_data_processing()
        elif args.step == 'train':
            pipeline.run_model_training()
        elif args.step == 'deploy':
            pipeline.run_model_deployment()
        elif args.step == 'visualize':
            pipeline.run_visualization()
        else:  # full pipeline
            pipeline.run_full_pipeline()
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise
    finally:
        logging.info("Pipeline execution completed")

if __name__ == '__main__':
    main()

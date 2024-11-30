import os
import yaml
from typing import Dict, Any
import logging
from logging.handlers import RotatingFileHandler

class ConfigurationError(Exception):
    """Custom exception for configuration related errors."""
    pass

class Config:
    """Configuration management class for the Spotify Data Pipeline."""
    
    def __init__(self):
        self.settings: Dict[str, Any] = {}
        self.secrets: Dict[str, Any] = {}
        self.logger = self._setup_logging()
        
        # Load configurations
        self._load_settings()
        self._load_secrets()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('spotify_pipeline')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            'logs/application.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatting
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Load and parse a YAML file."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {file_path}")
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML file {file_path}: {str(e)}")
            raise ConfigurationError(f"Error parsing YAML file {file_path}: {str(e)}")
    
    def _load_settings(self) -> None:
        """Load general settings from settings.yaml."""
        settings_path = os.path.join('config', 'settings.yaml')
        self.settings = self._load_yaml(settings_path)
        self.logger.info("Settings loaded successfully")
    
    def _load_secrets(self) -> None:
        """Load secrets from secrets.yaml."""
        secrets_path = os.path.join('config', 'secrets.yaml')
        try:
            self.secrets = self._load_yaml(secrets_path)
            self.logger.info("Secrets loaded successfully")
        except ConfigurationError:
            self.logger.warning(
                "secrets.yaml not found. Using environment variables for secrets."
            )
            self._load_secrets_from_env()
    
    def _load_secrets_from_env(self) -> None:
        """Load secrets from environment variables."""
        self.secrets = {
            'spotify': {
                'client_id': os.getenv('SPOTIFY_CLIENT_ID'),
                'client_secret': os.getenv('SPOTIFY_CLIENT_SECRET')
            },
            'aws': {
                'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                'region': os.getenv('AWS_REGION', 'us-east-1')
            }
        }
        
        # Verify required secrets are present
        missing_secrets = []
        if not self.secrets['spotify']['client_id']:
            missing_secrets.append('SPOTIFY_CLIENT_ID')
        if not self.secrets['spotify']['client_secret']:
            missing_secrets.append('SPOTIFY_CLIENT_SECRET')
            
        if missing_secrets:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_secrets)}"
            )
    
    def get_spotify_credentials(self) -> Dict[str, str]:
        """Get Spotify API credentials."""
        return self.secrets['spotify']
    
    def get_aws_credentials(self) -> Dict[str, str]:
        """Get AWS credentials."""
        return self.secrets['aws']
    
    def get_setting(self, *keys: str) -> Any:
        """
        Get a setting value using dot notation.
        
        Example:
            config.get_setting('data_collection', 'batch_size')
        """
        value = self.settings
        for key in keys:
            try:
                value = value[key]
            except KeyError:
                self.logger.error(f"Setting not found: {'.'.join(keys)}")
                raise ConfigurationError(f"Setting not found: {'.'.join(keys)}")
        return value
    
    def get_data_collection_settings(self) -> Dict[str, Any]:
        """Get all data collection settings."""
        return self.settings['data_collection']
    
    def get_model_training_settings(self) -> Dict[str, Any]:
        """Get all model training settings."""
        return self.settings['model_training']
    
    def get_aws_settings(self) -> Dict[str, Any]:
        """Get all AWS-related settings."""
        return self.settings['aws']

# Create a singleton instance
config = Config()

# Example usage:
if __name__ == '__main__':
    try:
        # Get specific settings
        batch_size = config.get_setting('data_collection', 'batch_size')
        print(f"Batch size: {batch_size}")
        
        # Get Spotify credentials
        spotify_creds = config.get_spotify_credentials()
        print("Spotify credentials loaded successfully")
        
        # Get AWS credentials
        aws_creds = config.get_aws_credentials()
        print("AWS credentials loaded successfully")
        
    except ConfigurationError as e:
        print(f"Configuration error: {str(e)}")

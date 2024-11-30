# Spotify Data Pipeline

A comprehensive data pipeline that collects, processes, and analyzes Spotify song data to predict song popularity using AWS services.

## Project Overview

This project implements an end-to-end data pipeline that:
1. Collects song data from Spotify API
2. Processes and stores the data in AWS S3
3. Transforms the data using AWS Glue
4. Trains machine learning models using AWS SageMaker
5. Visualizes insights using Amazon QuickSight

## Project Structure

```
spotify-data-pipeline/
├── src/                    # Source code
│   ├── config.py          # Configuration management
│   ├── data_ingestion.py  # Spotify API data collection
│   ├── data_processing.py # Data transformation
│   ├── model_training.py  # ML model training
│   ├── model_deployment.py# Model deployment
│   └── visualization.py   # Data visualization
├── tests/                 # Unit tests
├── notebooks/            # Jupyter notebooks for exploration
├── config/               # Configuration files
├── scripts/             # Utility scripts
├── aws/                 # AWS-related configurations
├── docs/               # Documentation
├── data/               # Data directory
│   ├── raw/           # Raw data
│   └── processed/     # Processed data
└── logs/              # Application logs
```

## Prerequisites

- Python 3.8+
- AWS Account with appropriate permissions
- Spotify Developer Account
- Git

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd spotify-data-pipeline
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:
- Set up AWS CLI
- Configure credentials in `~/.aws/credentials`

5. Set up Spotify API credentials:
- Create a Spotify Developer account
- Create an application to get API credentials
- Copy `config/secrets_template.yaml` to `config/secrets.yaml`
- Add your Spotify API credentials to `config/secrets.yaml`

## Usage

1. Data Collection:
```bash
python src/data_ingestion.py
```

2. Data Processing:
```bash
python src/data_processing.py
```

3. Model Training:
```bash
python src/model_training.py
```

4. Model Deployment:
```bash
python src/model_deployment.py
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Spotify Web API
- AWS Documentation
- Contributors and maintainers

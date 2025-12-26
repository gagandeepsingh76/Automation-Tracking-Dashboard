#!/usr/bin/env python3
"""
Trend Scope - Project Setup & Initialization Script
===================================================

Automated setup script for the Trend Scope project including
dependency installation, configuration validation, and initial data setup.

Author: Neelanjan Chakraborty
Website: https://neelanjanchakraborty.in/
"""

import os
import sys
import subprocess
import json
import requests
import zipfile
from pathlib import Path
import logging
import time
import shutil
from typing import Dict, List, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrendScopeSetup:
    """Comprehensive setup manager for Trend Scope project."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_dirs = [
            'data/raw', 'data/processed', 'data/schemas',
            'logs', 'models/trained', 'models/forecasting',
            'dashboards/exports', 'dashboards/tableau_templates',
            'dashboards/powerbi_templates', 'tests/unit',
            'tests/integration', 'tests/performance',
            'config', 'monitoring/grafana/dashboards',
            'monitoring/grafana/datasources'
        ]
        
        self.sample_datasets = {
            'sales_performance': {
                'url': 'https://www.kaggle.com/datasets/ramyelbouhy/sales-performance-dashboardpower-bi',
                'filename': 'sales_performance_sample.csv'
            },
            'customer_analytics': {
                'url': 'https://www.kaggle.com/datasets/graceegbe12/sales-and-customer-analytics-interactive-dashboard',
                'filename': 'customer_analytics_sample.csv'
            }
        }
    
    def create_directory_structure(self):
        """Create required project directories."""
        logger.info("📁 Creating project directory structure...")
        
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ Created: {dir_path}")
        
        logger.info("✅ Directory structure created successfully")
    
    def install_dependencies(self, environment: str = "development"):
        """Install Python dependencies."""
        logger.info("📦 Installing Python dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            # Install main requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True)
            
            # Install development dependencies if needed
            if environment == "development":
                dev_requirements = [
                    "pytest>=7.4.3",
                    "pytest-cov>=4.1.0",
                    "black>=23.11.0",
                    "flake8>=6.1.0",
                    "mypy>=1.7.1",
                    "pre-commit>=3.6.0",
                    "jupyter>=1.0.0"
                ]
                
                for package in dev_requirements:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], check=True, capture_output=True)
            
            logger.info("✅ Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            raise
    
    def setup_configuration(self):
        """Setup configuration files."""
        logger.info("⚙️ Setting up configuration files...")
        
        # Create .env file from template if it doesn't exist
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            logger.info("✅ Created .env file from template")
            logger.warning("⚠️ Please update .env with your actual credentials")
        
        # Create additional config files
        self._create_monitoring_configs()
        self._create_git_configs()
        
        logger.info("✅ Configuration setup completed")
    
    def _create_monitoring_configs(self):
        """Create monitoring configuration files."""
        # Prometheus configuration
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'scrape_configs': [
                {
                    'job_name': 'trend-scope',
                    'static_configs': [
                        {'targets': ['localhost:8000']}
                    ]
                }
            ]
        }
        
        prometheus_path = self.project_root / "monitoring" / "prometheus.yml"
        prometheus_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(prometheus_path, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        # Grafana datasource configuration
        grafana_datasource = {
            'apiVersion': 1,
            'datasources': [
                {
                    'name': 'Prometheus',
                    'type': 'prometheus',
                    'url': 'http://prometheus:9090',
                    'access': 'proxy',
                    'isDefault': True
                }
            ]
        }
        
        grafana_ds_path = self.project_root / "monitoring" / "grafana" / "datasources" / "prometheus.yml"
        grafana_ds_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(grafana_ds_path, 'w') as f:
            yaml.dump(grafana_datasource, f, default_flow_style=False)
    
    def _create_git_configs(self):
        """Create Git configuration files."""
        # .gitignore
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
.env
config/credentials.json
logs/*.log
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
models/trained/*
!models/trained/.gitkeep
dashboards/exports/*
!dashboards/exports/.gitkeep

# Tableau files
*.twb
*.tde
*.hyper

# Power BI files
*.pbix
*.pbit

# Jupyter
.ipynb_checkpoints/

# Docker
.dockerignore

# Monitoring
monitoring/grafana/dashboards/*
!monitoring/grafana/dashboards/.gitkeep
"""
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content.strip())
        
        # Create .gitkeep files
        gitkeep_dirs = [
            'data/raw', 'data/processed', 'models/trained',
            'dashboards/exports', 'monitoring/grafana/dashboards'
        ]
        
        for dir_path in gitkeep_dirs:
            gitkeep_path = self.project_root / dir_path / ".gitkeep"
            gitkeep_path.parent.mkdir(parents=True, exist_ok=True)
            gitkeep_path.touch()
    
    def download_sample_datasets(self):
        """Download sample datasets for development and testing."""
        logger.info("📊 Setting up sample datasets...")
        
        # Create synthetic sample data since we can't directly download Kaggle datasets
        self._create_synthetic_sales_data()
        self._create_synthetic_customer_data()
        
        logger.info("✅ Sample datasets created successfully")
    
    def _create_synthetic_sales_data(self):
        """Create synthetic sales performance data."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate date range
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate sample data
        n_records = len(date_range) * 50  # ~50 sales per day
        
        regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East & Africa']
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E', 'Product F']
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Health']
        
        data = {
            'date': np.random.choice(date_range, n_records),
            'region': np.random.choice(regions, n_records),
            'product': np.random.choice(products, n_records),
            'category': np.random.choice(categories, n_records),
            'sales_amount': np.random.lognormal(6, 1, n_records),  # Log-normal distribution
            'quantity': np.random.poisson(5, n_records) + 1,  # Poisson distribution
            'customer_id': np.random.randint(1, 10000, n_records),
            'order_id': range(1, n_records + 1)
        }
        
        # Calculate profit (15-25% of sales)
        profit_margin = np.random.uniform(0.15, 0.25, n_records)
        data['profit'] = data['sales_amount'] * profit_margin
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some seasonal trends
        df['month'] = df['date'].dt.month
        seasonal_multiplier = 1 + 0.3 * np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['sales_amount'] *= seasonal_multiplier
        df['profit'] *= seasonal_multiplier
        
        # Save to CSV
        output_path = self.project_root / "data" / "raw" / "sales_performance_sample.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"  ✅ Created synthetic sales data: {len(df):,} records")
    
    def _create_synthetic_customer_data(self):
        """Create synthetic customer analytics data."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        np.random.seed(42)
        
        # Generate customer data
        n_customers = 5000
        
        customer_segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 
                           'New Customers', 'At Risk', 'Cannot Lose Them', 'Others']
        
        acquisition_channels = ['Online', 'Social Media', 'Email', 'Direct', 'Referral', 'Paid Ads']
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'Brazil']
        
        data = {
            'customer_id': range(1, n_customers + 1),
            'customer_segment': np.random.choice(customer_segments, n_customers),
            'acquisition_channel': np.random.choice(acquisition_channels, n_customers),
            'country': np.random.choice(countries, n_customers),
            'registration_date': pd.date_range('2020-01-01', '2023-12-31', periods=n_customers),
            'last_purchase_date': pd.date_range('2023-01-01', '2023-12-31', periods=n_customers),
            'total_orders': np.random.poisson(8, n_customers) + 1,
            'total_spent': np.random.lognormal(7, 1, n_customers),
            'avg_order_value': np.random.lognormal(4, 0.5, n_customers),
            'customer_lifetime_value': np.random.lognormal(8, 1, n_customers),
            'satisfaction_score': np.random.normal(7.5, 1.5, n_customers).clip(1, 10),
            'churn_probability': np.random.beta(2, 5, n_customers)  # Most customers have low churn
        }
        
        # Calculate derived metrics
        df = pd.DataFrame(data)
        
        # Recency (days since last purchase)
        df['recency'] = (df['last_purchase_date'].max() - df['last_purchase_date']).dt.days
        
        # Frequency (orders per year)
        account_age = (df['last_purchase_date'] - df['registration_date']).dt.days / 365
        df['frequency'] = df['total_orders'] / (account_age + 0.1)  # Avoid division by zero
        
        # Monetary (average spending)
        df['monetary'] = df['total_spent']
        
        # Save to CSV
        output_path = self.project_root / "data" / "raw" / "customer_analytics_sample.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"  ✅ Created synthetic customer data: {len(df):,} records")
    
    def validate_setup(self):
        """Validate the project setup."""
        logger.info("🔍 Validating project setup...")
        
        validation_results = {
            'directories': self._validate_directories(),
            'dependencies': self._validate_dependencies(),
            'configuration': self._validate_configuration(),
            'datasets': self._validate_datasets()
        }
        
        # Calculate overall score
        total_checks = sum(len(checks) for checks in validation_results.values())
        passed_checks = sum(
            sum(1 for result in checks.values() if result) 
            for checks in validation_results.values()
        )
        
        success_rate = passed_checks / total_checks if total_checks > 0 else 0
        
        logger.info(f"📊 Validation Results:")
        logger.info(f"  Overall Success Rate: {success_rate:.1%}")
        logger.info(f"  Checks Passed: {passed_checks}/{total_checks}")
        
        for category, results in validation_results.items():
            category_passed = sum(1 for result in results.values() if result)
            category_total = len(results)
            logger.info(f"  {category.title()}: {category_passed}/{category_total}")
        
        if success_rate >= 0.9:
            logger.info("✅ Project setup validation PASSED")
            return True
        else:
            logger.warning("⚠️ Project setup validation had issues")
            return False
    
    def _validate_directories(self) -> Dict[str, bool]:
        """Validate required directories exist."""
        results = {}
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            results[dir_path] = full_path.exists() and full_path.is_dir()
        return results
    
    def _validate_dependencies(self) -> Dict[str, bool]:
        """Validate Python dependencies are installed."""
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'tensorflow',
            'plotly', 'tableauserverclient', 'msal', 'azure-storage-blob'
        ]
        
        results = {}
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                results[package] = True
            except ImportError:
                results[package] = False
        
        return results
    
    def _validate_configuration(self) -> Dict[str, bool]:
        """Validate configuration files exist."""
        config_files = [
            'config/settings.yaml',
            '.env.example',
            'docker-compose.yml',
            'Dockerfile',
            'requirements.txt'
        ]
        
        results = {}
        for config_file in config_files:
            file_path = self.project_root / config_file
            results[config_file] = file_path.exists() and file_path.is_file()
        
        return results
    
    def _validate_datasets(self) -> Dict[str, bool]:
        """Validate sample datasets exist."""
        dataset_files = [
            'data/raw/sales_performance_sample.csv',
            'data/raw/customer_analytics_sample.csv'
        ]
        
        results = {}
        for dataset_file in dataset_files:
            file_path = self.project_root / dataset_file
            results[dataset_file] = file_path.exists() and file_path.stat().st_size > 0
        
        return results
    
    def create_initial_documentation(self):
        """Create initial project documentation."""
        logger.info("📝 Creating project documentation...")
        
        # Create CONTRIBUTING.md
        contributing_content = """# Contributing to Trend Scope

Thank you for your interest in contributing to Trend Scope! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository
2. Run the setup script: `python scripts/setup.py`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting: `black .`
- Run linting with flake8: `flake8 .`
- Type hints are encouraged

## Testing

- Write unit tests for new features
- Run tests with: `pytest tests/`
- Ensure test coverage remains above 90%

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features.
"""
        
        contrib_path = self.project_root / "docs" / "CONTRIBUTING.md"
        contrib_path.parent.mkdir(parents=True, exist_ok=True)
        with open(contrib_path, 'w') as f:
            f.write(contributing_content)
        
        # Create API documentation template
        api_docs_content = """# Trend Scope API Documentation

## Overview

The Trend Scope API provides programmatic access to the automated dashboard pipeline functionality.

## Authentication

All API endpoints require authentication via API key or OAuth2 token.

## Endpoints

### Pipeline Management

- `POST /api/v1/pipeline/execute` - Execute the full pipeline
- `GET /api/v1/pipeline/status` - Get pipeline execution status
- `GET /api/v1/pipeline/metrics` - Get performance metrics

### Data Management

- `GET /api/v1/data/sources` - List available data sources
- `POST /api/v1/data/extract` - Trigger data extraction
- `GET /api/v1/data/quality` - Get data quality metrics

### Dashboard Management

- `GET /api/v1/dashboards` - List published dashboards
- `POST /api/v1/dashboards/publish` - Publish dashboard
- `GET /api/v1/dashboards/{id}/analytics` - Get dashboard analytics

## Error Handling

All endpoints return standard HTTP status codes and JSON error responses.

## Rate Limiting

API calls are limited to 1000 requests per hour per API key.
"""
        
        api_docs_path = self.project_root / "docs" / "API.md"
        with open(api_docs_path, 'w') as f:
            f.write(api_docs_content)
        
        logger.info("✅ Documentation created successfully")
    
    def run_full_setup(self, environment: str = "development", skip_deps: bool = False):
        """Run complete project setup."""
        logger.info("🚀 Starting Trend Scope project setup...")
        logger.info("=" * 60)
        
        try:
            # Step 1: Create directories
            self.create_directory_structure()
            
            # Step 2: Install dependencies (optional)
            if not skip_deps:
                self.install_dependencies(environment)
            
            # Step 3: Setup configuration
            self.setup_configuration()
            
            # Step 4: Download sample data
            self.download_sample_datasets()
            
            # Step 5: Create documentation
            self.create_initial_documentation()
            
            # Step 6: Validate setup
            validation_passed = self.validate_setup()
            
            # Final summary
            logger.info("=" * 60)
            if validation_passed:
                logger.info("🎉 Trend Scope setup completed successfully!")
                logger.info("✅ Project is ready for development and execution")
                logger.info("📖 Next steps:")
                logger.info("  1. Update .env file with your credentials")
                logger.info("  2. Run: python schedule/run_workflow.py --tool both")
                logger.info("  3. Check logs/ directory for execution details")
                logger.info("  4. View generated dashboards in dashboards/exports/")
            else:
                logger.warning("⚠️ Setup completed with some issues")
                logger.warning("Please review the validation results above")
            
            logger.info("=" * 60)
            return validation_passed
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise


def main():
    """Main setup script entry point."""
    parser = argparse.ArgumentParser(
        description="Trend Scope Project Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--environment',
        choices=['development', 'production'],
        default='development',
        help='Setup environment (default: development)'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation, skip setup'
    )
    
    args = parser.parse_args()
    
    # Initialize setup manager
    setup_manager = TrendScopeSetup()
    
    if args.validate_only:
        # Run validation only
        validation_passed = setup_manager.validate_setup()
        sys.exit(0 if validation_passed else 1)
    else:
        # Run full setup
        success = setup_manager.run_full_setup(
            environment=args.environment,
            skip_deps=args.skip_deps
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

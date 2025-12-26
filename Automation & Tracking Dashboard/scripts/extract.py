#!/usr/bin/env python3
"""
Trend Scope - Advanced Data Extraction Module
=============================================

Multi-source data extraction with intelligent caching, rate limiting,
and adaptive retry mechanisms for enterprise-grade data pipelines.

Author: Neelanjan Chakraborty
Website: https://neelanjanchakraborty.in/
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import os
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import kaggle
import redis
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import pyarrow.parquet as pq
import pyarrow as pa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    """Performance metrics for data extraction operations."""
    source: str
    records_extracted: int
    file_size_mb: float
    extraction_time: float
    cache_hit_ratio: float
    error_count: int
    data_quality_score: float


class AdaptiveRateLimiter:
    """Intelligent rate limiter that adapts to API response patterns."""
    
    def __init__(self, initial_rate: float = 1.0, max_rate: float = 10.0):
        self.current_rate = initial_rate
        self.max_rate = max_rate
        self.last_request_time = 0
        self.success_count = 0
        self.error_count = 0
        self.rate_adjustment_threshold = 10
        
    async def acquire(self):
        """Acquire rate limit token with adaptive adjustment."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < (1.0 / self.current_rate):
            await asyncio.sleep((1.0 / self.current_rate) - time_since_last)
        
        self.last_request_time = time.time()
        
    def record_success(self):
        """Record successful request and potentially increase rate."""
        self.success_count += 1
        if self.success_count % self.rate_adjustment_threshold == 0:
            self.current_rate = min(self.current_rate * 1.1, self.max_rate)
            logger.debug(f"Rate increased to {self.current_rate:.2f} req/s")
            
    def record_error(self):
        """Record error and decrease rate."""
        self.error_count += 1
        self.current_rate = max(self.current_rate * 0.8, 0.1)
        logger.warning(f"Rate decreased to {self.current_rate:.2f} req/s")


class DataSourceConnector:
    """Base connector class for data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = None
        self.rate_limiter = AdaptiveRateLimiter()
        
    def setup_cache(self):
        """Initialize Redis cache if available."""
        try:
            if self.config.get('performance', {}).get('caching', {}).get('enable_redis'):
                redis_config = self.config['performance']['caching']
                self.cache = redis.Redis(
                    host=redis_config['redis_host'],
                    port=redis_config['redis_port'],
                    db=redis_config['redis_db'],
                    decode_responses=True
                )
                self.cache.ping()
                logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Cache initialization failed: {e}")
            self.cache = None
    
    def get_cache_key(self, source: str, params: Dict[str, Any]) -> str:
        """Generate cache key for dataset."""
        key_data = f"{source}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and valid."""
        if not self.cache:
            return None
            
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                # In production, this would deserialize from cache format
                logger.info(f"Cache hit for key: {cache_key}")
                return pd.read_json(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def cache_data(self, cache_key: str, data: pd.DataFrame, ttl: int = 3600):
        """Cache processed data with TTL."""
        if not self.cache:
            return
            
        try:
            # In production, use more efficient serialization (parquet/arrow)
            cached_value = data.to_json()
            self.cache.setex(cache_key, ttl, cached_value)
            logger.info(f"Data cached with key: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")


class KaggleExtractor(DataSourceConnector):
    """Advanced Kaggle dataset extractor with intelligent caching."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.setup_kaggle_auth()
        self.setup_cache()
        
    def setup_kaggle_auth(self):
        """Configure Kaggle API authentication."""
        kaggle_config = self.config['data_sources']['kaggle']
        os.environ['KAGGLE_USERNAME'] = kaggle_config['username']
        os.environ['KAGGLE_KEY'] = kaggle_config['api_key']
        
        # Authenticate with Kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api = KaggleApi()
        self.api.authenticate()
        logger.info("Kaggle API authentication successful")
    
    async def extract_dataset(self, dataset_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """Extract dataset with intelligent caching and processing."""
        cache_key = self.get_cache_key("kaggle", {"dataset_id": dataset_id})
        
        # Check cache first unless force refresh
        if not force_refresh:
            cached_data = self.get_cached_data(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            await self.rate_limiter.acquire()
            
            # Download dataset files
            dataset_path = Path(f"data/raw/kaggle_{dataset_id.replace('/', '_')}")
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading Kaggle dataset: {dataset_id}")
            self.api.dataset_download_files(
                dataset_id,
                path=str(dataset_path),
                unzip=True
            )
            
            # Process downloaded files
            csv_files = list(dataset_path.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in dataset {dataset_id}")
            
            # Load and combine CSV files
            dataframes = []
            for csv_file in csv_files:
                logger.info(f"Processing file: {csv_file.name}")
                df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
                df['source_file'] = csv_file.name
                df['extraction_timestamp'] = datetime.utcnow()
                dataframes.append(df)
            
            # Combine dataframes
            combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            
            # Data quality checks
            quality_score = self._calculate_data_quality(combined_df)
            logger.info(f"Data quality score: {quality_score:.2f}")
            
            # Cache the processed data
            cache_ttl = self.config['data_sources']['kaggle']['datasets'].get(
                dataset_id.split('/')[-1], {}
            ).get('cache_ttl', 3600)
            
            self.cache_data(cache_key, combined_df, ttl=cache_ttl)
            
            self.rate_limiter.record_success()
            return combined_df
            
        except Exception as e:
            self.rate_limiter.record_error()
            logger.error(f"Failed to extract dataset {dataset_id}: {e}")
            raise
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness and validity."""
        if df.empty:
            return 0.0
        
        # Completeness score (non-null values)
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        
        # Validity score (basic data type consistency)
        validity = 1.0  # Simplified for demo
        
        # Uniqueness score for potential key columns
        uniqueness = 1.0  # Simplified for demo
        
        return (completeness * 0.5 + validity * 0.3 + uniqueness * 0.2)


class DatabaseExtractor(DataSourceConnector):
    """Production-grade database extractor with connection pooling."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.setup_database_connection()
        self.setup_cache()
        
    def setup_database_connection(self):
        """Initialize database connection with pooling."""
        db_config = self.config['data_sources']['database']
        
        connection_string = (
            f"postgresql://{db_config['username']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        self.engine = sa.create_engine(
            connection_string,
            pool_size=db_config['pool_size'],
            max_overflow=db_config['max_overflow'],
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("Database connection pool initialized")
    
    async def extract_table(self, table_name: str, query: Optional[str] = None, 
                          chunk_size: int = 10000) -> pd.DataFrame:
        """Extract table data with chunked processing for large datasets."""
        cache_key = self.get_cache_key("database", {
            "table": table_name, 
            "query": query or f"SELECT * FROM {table_name}"
        })
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            query_sql = query or f"SELECT * FROM {table_name}"
            
            # Use chunked reading for large tables
            chunks = []
            for chunk in pd.read_sql(
                query_sql, 
                self.engine, 
                chunksize=chunk_size
            ):
                chunk['extraction_timestamp'] = datetime.utcnow()
                chunks.append(chunk)
            
            combined_df = pd.concat(chunks, ignore_index=True)
            
            # Cache the result
            self.cache_data(cache_key, combined_df, ttl=3600)
            
            logger.info(f"Extracted {len(combined_df)} records from {table_name}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Database extraction failed for {table_name}: {e}")
            raise


class AzureBlobExtractor(DataSourceConnector):
    """Azure Blob Storage extractor for cloud-native data lakes."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.setup_azure_client()
        self.setup_cache()
        
    def setup_azure_client(self):
        """Initialize Azure Blob Storage client."""
        azure_config = self.config['data_sources']['azure_storage']
        
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{azure_config['account_name']}.blob.core.windows.net",
            credential=azure_config['account_key']
        )
        
        self.container_name = azure_config['container_name']
        logger.info("Azure Blob Storage client initialized")
    
    async def extract_blob_data(self, blob_name: str, file_format: str = "parquet") -> pd.DataFrame:
        """Extract data from Azure Blob Storage with format detection."""
        cache_key = self.get_cache_key("azure_blob", {"blob": blob_name})
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            # Download blob data
            blob_data = blob_client.download_blob().readall()
            
            # Process based on file format
            if file_format.lower() == "parquet":
                df = pd.read_parquet(io.BytesIO(blob_data))
            elif file_format.lower() == "csv":
                df = pd.read_csv(io.BytesIO(blob_data))
            elif file_format.lower() == "json":
                df = pd.read_json(io.BytesIO(blob_data))
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            df['extraction_timestamp'] = datetime.utcnow()
            df['source_blob'] = blob_name
            
            # Cache the result
            self.cache_data(cache_key, df, ttl=3600)
            
            logger.info(f"Extracted {len(df)} records from blob: {blob_name}")
            return df
            
        except Exception as e:
            logger.error(f"Azure blob extraction failed for {blob_name}: {e}")
            raise


class TrendScopeExtractor:
    """Main extraction orchestrator for Trend Scope pipeline."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.metrics: List[ExtractionMetrics] = []
        
        # Initialize extractors
        self.kaggle_extractor = KaggleExtractor(self.config)
        self.database_extractor = DatabaseExtractor(self.config)
        self.azure_extractor = AzureBlobExtractor(self.config)
        
        logger.info("Trend Scope Extractor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Expand environment variables
        config = self._expand_env_vars(config)
        return config
    
    def _expand_env_vars(self, config: Any) -> Any:
        """Recursively expand environment variables in config."""
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config
    
    async def extract_all_sources(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Extract data from all configured sources concurrently."""
        extraction_tasks = []
        
        # Kaggle datasets
        kaggle_datasets = self.config['data_sources']['kaggle']['datasets']
        for dataset_key, dataset_config in kaggle_datasets.items():
            if isinstance(dataset_config, dict) and 'id' in dataset_config:
                task = self._extract_kaggle_with_metrics(
                    dataset_config['id'], 
                    force_refresh
                )
                extraction_tasks.append((f"kaggle_{dataset_key}", task))
        
        # Execute all extractions concurrently
        results = {}
        for source_name, task in extraction_tasks:
            try:
                start_time = time.time()
                data = await task
                extraction_time = time.time() - start_time
                
                # Record metrics
                metrics = ExtractionMetrics(
                    source=source_name,
                    records_extracted=len(data),
                    file_size_mb=data.memory_usage(deep=True).sum() / 1024 / 1024,
                    extraction_time=extraction_time,
                    cache_hit_ratio=0.0,  # Would be calculated from cache stats
                    error_count=0,
                    data_quality_score=self._calculate_overall_quality(data)
                )
                self.metrics.append(metrics)
                
                results[source_name] = data
                logger.info(f"✅ Extracted {source_name}: {len(data)} records in {extraction_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to extract {source_name}: {e}")
                continue
        
        return results
    
    async def _extract_kaggle_with_metrics(self, dataset_id: str, force_refresh: bool) -> pd.DataFrame:
        """Extract Kaggle dataset with detailed metrics tracking."""
        return await self.kaggle_extractor.extract_dataset(dataset_id, force_refresh)
    
    def _calculate_overall_quality(self, df: pd.DataFrame) -> float:
        """Calculate comprehensive data quality score."""
        if df.empty:
            return 0.0
        
        # Completeness (non-null ratio)
        completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        
        # Consistency (data type consistency)
        consistency = 1.0  # Simplified
        
        # Accuracy (basic validation)
        accuracy = 1.0  # Simplified
        
        return (completeness * 0.4 + consistency * 0.3 + accuracy * 0.3)
    
    def save_extraction_report(self, output_path: str = "logs/extraction_report.json"):
        """Generate detailed extraction performance report."""
        report = {
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "total_sources": len(self.metrics),
            "total_records": sum(m.records_extracted for m in self.metrics),
            "total_size_mb": sum(m.file_size_mb for m in self.metrics),
            "average_quality_score": np.mean([m.data_quality_score for m in self.metrics]),
            "total_extraction_time": sum(m.extraction_time for m in self.metrics),
            "sources": [
                {
                    "source": m.source,
                    "records": m.records_extracted,
                    "size_mb": round(m.file_size_mb, 2),
                    "time_seconds": round(m.extraction_time, 2),
                    "quality_score": round(m.data_quality_score, 3),
                    "cache_hit_ratio": round(m.cache_hit_ratio, 3)
                }
                for m in self.metrics
            ]
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Extraction report saved to: {output_path}")
        return report


async def main():
    """Main extraction workflow entry point."""
    extractor = TrendScopeExtractor()
    
    try:
        logger.info("🚀 Starting Trend Scope data extraction...")
        
        # Extract all data sources
        extracted_data = await extractor.extract_all_sources(force_refresh=False)
        
        # Save extracted data
        for source_name, data in extracted_data.items():
            output_path = f"data/processed/{source_name}_extracted.parquet"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data.to_parquet(output_path, compression='snappy')
            logger.info(f"💾 Saved {source_name} to {output_path}")
        
        # Generate performance report
        report = extractor.save_extraction_report()
        
        # Summary statistics
        total_records = sum(len(data) for data in extracted_data.values())
        total_time = sum(m.extraction_time for m in extractor.metrics)
        
        logger.info(f"✅ Extraction completed successfully!")
        logger.info(f"📈 Total records extracted: {total_records:,}")
        logger.info(f"⏱️  Total extraction time: {total_time:.2f} seconds")
        logger.info(f"🚀 Average processing speed: {total_records/total_time:.0f} records/second")
        
        return extracted_data
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        raise


if __name__ == "__main__":
    # Run the extraction pipeline
    asyncio.run(main())

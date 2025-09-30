#!/usr/bin/env python3

import pandas as pd
import numpy as np
import time
import requests
import sys
import os
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import logging
from datetime import datetime

# --- Configuration ---
CONFIG = {
    "PROMETHEUS_URL": "http://localhost:9090",
    "MODELS_DIR": "trained_models",
    "QUERIES": {
        "cpu": '(((count by (instance,dc,job) (count(node_cpu_seconds_total{}) by (cpu, instance,dc,job))) - avg by (instance,dc,job) (sum by (instance, mode,dc,job)(irate(node_cpu_seconds_total{mode="idle"}[5m])))) * 100) / count by(instance,dc,job) (count(node_cpu_seconds_total{}) by (cpu, instance,dc,job))',
        "memory": 'round((1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100, 0.01)',
        "disk": 'round(max((1 - (node_filesystem_free_bytes{mountpoint=~"/var|/home|/tmp"} / node_filesystem_size_bytes{mountpoint=~"/var|/home|/tmp"})) * 100) by (dc, instance, mountpoint), 0.01)'
    },
    "STEP_INTERVAL": "30s",
    "FETCH_INTERVAL_SECONDS": 10, # wait-time between requesting to prometheus
    "MIN_DATA_POINTS": 10, # minimum data points required to train/retrain
    "MIN_RETRAIN_MINUTES": 30
}

# Create log directory if it doesn't exist
LOG_DIR = "/var/serversage/anomaly/ProphetTrainingJsonSerialization/logs_incremental"
os.makedirs(LOG_DIR, exist_ok=True)


# Build log filename with timestamp
log_filename = os.path.join(
    LOG_DIR,
    f"incremental_retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_model_paths(models_dir: str) -> list:
    """
    Get all model paths from the trained_models directory structure.
    Returns list of tuples: (metric, dc, model_path)
    """
    model_paths = []
    if not os.path.exists(models_dir):
        logger.error(f"Models directory {models_dir} does not exist!")
        return model_paths
        
    for metric in os.listdir(models_dir):
        metric_dir = os.path.join(models_dir, metric)
        if os.path.isdir(metric_dir):
            for model_file in os.listdir(metric_dir):
                if model_file.endswith('.json'):
                    dc = model_file.replace('.json', '')
                    model_path = os.path.join(metric_dir, model_file)
                    model_paths.append((metric, dc, model_path))
    
    logger.info(f"Found {len(model_paths)} models in {models_dir}")
    return model_paths


def get_model_training_info(model_path: str) -> tuple:
    """
    Gets training information from a saved model.
    Returns: (last_timestamp, total_training_days, existing_data)
    """
    logger.debug(f"Reading model training info from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist")
    
    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(fin.read())
        
        if not hasattr(model, 'history') or model.history is None or model.history.empty:
            raise ValueError(f"Model {model_path} has no training history")
        
        # Get existing training data
        existing_data = model.history[['ds', 'y']].copy()
        existing_data = existing_data.sort_values('ds').reset_index(drop=True)
        
        # Calculate training period info
        first_timestamp = existing_data['ds'].min().to_pydatetime()
        last_timestamp = existing_data['ds'].max().to_pydatetime()
        total_days = (last_timestamp - first_timestamp).days + 1
        
        logger.info(f"Model {os.path.basename(model_path)} training info:")
        logger.info(f"  - First training date: {first_timestamp}")
        logger.info(f"  - Last training date: {last_timestamp}")
        logger.info(f"  - Total training period: {total_days} days")
        logger.info(f"  - Training data points: {len(existing_data)}")
        
        return last_timestamp, total_days, existing_data
        
    except Exception as e:
        logger.error(f"Error reading model {model_path}: {str(e)}")
        raise


def fetch_new_data(metric: str, dc: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Queries Prometheus for new data for a specific metric and DC from start_time to end_time.
    """
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())
    
    # Use the correct query from config
    query = CONFIG['QUERIES'][metric]
    
    url = f"{CONFIG['PROMETHEUS_URL']}/api/v1/query_range"
    params = {
        'query': query, 
        'start': start_ts, 
        'end': end_ts, 
        'step': CONFIG['STEP_INTERVAL']
    }

    logger.info(f"Fetching new {metric} data for DC '{dc}' from {start_time} to {end_time}")
    logger.debug(f"Prometheus query: {query}")
    
    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] != 'success':
            error_msg = data.get('error', 'Unknown error')
            logger.error(f"Prometheus query failed: {error_msg}")
            logger.error(f"Query used: {query}")
            raise Exception(f"Prometheus query failed: {error_msg}")
        
        result = data['data']['result']
        
        if not result:
            logger.warning(f"No data returned for {metric} in the specified time range")
            return pd.DataFrame()

        # Process all time series and filter by the specific DC
        all_dfs = []
        for series in result:
            # Check if this series belongs to our target DC
            series_dc = series['metric'].get('dc', '')
            if series_dc != dc:
                continue
                
            values = series.get('values', [])
            if not values:
                continue
                
            df = pd.DataFrame(values, columns=['timestamp', 'y'])
            df['ds'] = pd.to_datetime(df['timestamp'], unit='s')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df = df.dropna().reset_index(drop=True)
            
            if not df.empty:
                all_dfs.append(df[['ds', 'y']])
        
        if all_dfs:
            # Combine all instances and average values at same timestamps
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.groupby('ds', as_index=False)['y'].mean()
            combined_df = combined_df.sort_values('ds').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(combined_df)} new data points for {metric}-{dc}")
            if len(combined_df) > 0:
                logger.debug(f"New data range: {combined_df['ds'].min()} to {combined_df['ds'].max()}")
            return combined_df
        else:
            logger.warning(f"No valid data points found for {metric}-{dc}")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed for {metric}-{dc}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing data for {metric}-{dc}: {str(e)}")
        raise


def extract_warm_start_params(model: Prophet) -> dict:
    """
    Extract parameters from a trained Prophet model for warm starting.
    Returns parameters that can be used to initialize a new model.
    """
    logger.debug("Extracting warm start parameters from existing model")
    
    try:
        if not hasattr(model, 'params') or model.params is None:
            logger.warning("Model has no parameters for warm start")
            return None
            
        params = {}
        
        # Extract scalar parameters
        for param_name in ['k', 'm', 'sigma_obs']:
            if param_name in model.params:
                param_value = model.params[param_name]
                if isinstance(param_value, (list, np.ndarray)) and len(param_value) > 0:
                    if isinstance(param_value[0], (list, np.ndarray)) and len(param_value[0]) > 0:
                        params[param_name] = float(param_value[0][0])
                    else:
                        params[param_name] = float(param_value[0])
                else:
                    params[param_name] = float(param_value)
        
        # Extract array parameters
        for param_name in ['delta', 'beta']:
            if param_name in model.params:
                param_value = model.params[param_name]
                if isinstance(param_value, (list, np.ndarray)) and len(param_value) > 0:
                    params[param_name] = param_value[0] if isinstance(param_value[0], np.ndarray) else param_value
        
        logger.debug(f"Extracted warm start parameters: {list(params.keys())}")
        return params if params else None
        
    except Exception as e:
        logger.warning(f"Could not extract warm start parameters: {str(e)}")
        return None


def incremental_retrain_model(metric: str, dc: str, model_path: str) -> bool:
    """
    Perform cumulative incremental retraining for a specific model.
    """
    logger.info(f"=== Starting cumulative incremental retraining for {metric}-{dc} ===")
    
    try:
        # 1. Load existing model and get training information
        logger.info(f"Loading existing model from: {model_path}")
        with open(model_path, 'r') as fin:
            existing_model = model_from_json(fin.read())
        
        # Get training info from existing model
        last_timestamp, current_training_days, existing_data = get_model_training_info(model_path)
        current_time = datetime.now()
        
        # Check if enough time has passed since last training
        time_since_last = current_time - last_timestamp
        if time_since_last < timedelta(minutes=CONFIG['MIN_RETRAIN_MINUTES']):
            logger.info(f"Model {metric}-{dc} was recently trained ({time_since_last} ago). Skipping retraining.")
            return False

        logger.info(f"Time since last training: {time_since_last}")

        # 2. Fetch new data from last timestamp to current time
        logger.info(f"Fetching new data to extend training period...")
        
        # Start from slightly after last timestamp to avoid overlap
        fetch_start_time = last_timestamp + timedelta(minutes=5)  # Add buffer to avoid overlap
        new_data = fetch_new_data(metric, dc, fetch_start_time, current_time)
        
        if new_data.empty:
            logger.info(f"No new data available for {metric}-{dc}. Model remains at {current_training_days} days of training.")
            return False
            
        if len(new_data) < CONFIG['MIN_DATA_POINTS']:
            logger.info(f"Insufficient new data for {metric}-{dc} ({len(new_data)} points < {CONFIG['MIN_DATA_POINTS']}). Skipping retraining.")
            return False

        # 3. Combine existing training data with new data for complete dataset
        logger.info(f"Combining existing training data ({len(existing_data)} points) with new data ({len(new_data)} points)")
        
        complete_training_data = pd.concat([existing_data, new_data], ignore_index=True)
        complete_training_data = complete_training_data.sort_values('ds').reset_index(drop=True)
        
        # Remove any potential duplicates based on timestamp
        complete_training_data = complete_training_data.drop_duplicates(subset=['ds'], keep='last').reset_index(drop=True)
        
        # Calculate new training period
        new_first_date = complete_training_data['ds'].min().to_pydatetime()
        new_last_date = complete_training_data['ds'].max().to_pydatetime()
        new_training_days = (new_last_date - new_first_date).days + 1
        
        logger.info(f"Complete training dataset prepared:")
        logger.info(f"  - Total data points: {len(complete_training_data)}")
        logger.info(f"  - Training period: {new_first_date} to {new_last_date}")
        logger.info(f"  - Total training days: {new_training_days} (was {current_training_days})")
        logger.info(f"  - Added {len(new_data)} new data points")

        # 4. Extract warm start parameters from existing model
        warm_start_params = extract_warm_start_params(existing_model)
        
        # 5. Create new model with same configuration and train with complete dataset
        logger.info(f"Creating new Prophet model for cumulative training...")
        new_model = Prophet(
            interval_width=0.8,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        
        # 6. Train model with complete dataset using warm start
        logger.info(f"Training model on complete dataset ({len(complete_training_data)} points) with warm start...")
        start_time = time.time()
        
        if warm_start_params:
            logger.info("Using warm start parameters from existing model for faster convergence")
            new_model.fit(complete_training_data, init=warm_start_params)
        else:
            logger.warning("No warm start parameters available, training from scratch")
            new_model.fit(complete_training_data)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # 7. Save the updated model (replace existing)
        logger.info(f"Saving updated model to: {model_path}")
        
        # # Create backup of original model
        # backup_path = model_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # os.rename(model_path, backup_path)
        # logger.debug(f"Created backup at: {backup_path}")
        
        # Save new model
        with open(model_path, 'w') as fout:
            fout.write(model_to_json(new_model))
        
        logger.info(f"✓ Successfully retrained model for {metric}-{dc}")
        logger.info(f"✓ Model training extended from {current_training_days} to {new_training_days} days")
        logger.info(f"✓ Model now trained up to: {new_last_date}")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found for {metric}-{dc}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Failed to retrain {metric}-{dc}: {str(e)}", exc_info=True)
        return False


def main():
    """
    Main function to perform cumulative incremental retraining of all models.
    """
    logger.info("=== Starting Cumulative Incremental Model Retraining Process ===")
    start_time = time.time()
    
    # Get all model paths
    model_paths = get_model_paths(CONFIG['MODELS_DIR'])
    if not model_paths:
        logger.error("No models found to retrain!")
        sys.exit(1)
    
    logger.info(f"Found {len(model_paths)} models to process")
    
    # Process each model
    successful = 0
    failed = 0
    skipped = 0
    
    for i, (metric, dc, model_path) in enumerate(model_paths, 1):
        logger.info(f"\n--- Processing model {i}/{len(model_paths)}: {metric}-{dc} ---")
        
        try:
            result = incremental_retrain_model(metric, dc, model_path)
            if result:
                successful += 1
                logger.info(f"✓ Model {i}/{len(model_paths)} ({metric}-{dc}) successfully extended training period")
            else:
                skipped += 1
                logger.info(f"○ Model {i}/{len(model_paths)} ({metric}-{dc}) skipped (no new data or recent)")
                
        except Exception as e:
            failed += 1
            logger.error(f"✗ Model {i}/{len(model_paths)} ({metric}-{dc}) failed: {str(e)}")
        
        # Add delay between models to avoid overwhelming Prometheus
        if i < len(model_paths):
            time.sleep(CONFIG['FETCH_INTERVAL_SECONDS'])
    
    total_time = time.time() - start_time
    
    logger.info("\n=== Cumulative Incremental Retraining Process Completed ===")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Models processed: {len(model_paths)}")
    logger.info(f"  ✓ Successfully extended training period: {successful}")
    logger.info(f"  ○ Skipped (no new data/recent): {skipped}")
    logger.info(f"  ✗ Failed: {failed}")
    
    if failed > 0:
        logger.warning(f"Some models failed to retrain. Check logs for details.")
        sys.exit(1)
    
    logger.info("All models processed successfully! Training periods extended cumulatively.")


if __name__ == "__main__":
    main()

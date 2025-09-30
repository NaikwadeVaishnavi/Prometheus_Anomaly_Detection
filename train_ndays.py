#!/usr/bin/env python3
"""
Prometheus Metrics Data Exporter and Prophet Model Trainer

This script queries Prometheus for various metrics (CPU, memory, disk) over a specified period,
fetches data in batches to handle Prometheus limits, and trains Prophet models for each metric and DC.
"""

import os
import logging
import pandas as pd
import requests
import csv
import json
import time
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.serialize import model_to_json
from dotenv import load_dotenv

# --- Setup ---
load_dotenv(".env")
PROM_URL = os.getenv("PROM_URL")

# Parse metrics from comma-separated string to list
METRICS_LIST = os.getenv("METRICS", "cpu,memory,disk").split(',')

# Get queries for each metric
QUERIES = {}
for metric in METRICS_LIST:
    query = os.getenv(f"{metric.upper()}_QUERY")
    if query:
        QUERIES[metric] = query
    else:
        print(f"⚠️ Warning: No query found for metric '{metric}'")

# Get configuration from .env with defaults
FETCH_INTERVAL_SECONDS = int(os.getenv("FETCH_INTERVAL_SECONDS", 30))
QUERY_DAYS = int(os.getenv("QUERY_DAYS", 3))
BATCH_HOURS = int(os.getenv("BATCH_HOURS", 48))
STEP_INTERVAL = os.getenv("STEP_INTERVAL", "30s")
MIN_DATA_POINTS = int(os.getenv("MIN_DATA_POINTS", 100))

# Prophet model settings
DAILY_SEASONALITY = os.getenv("DAILY_SEASONALITY", "True").lower() == "true"
WEEKLY_SEASONALITY = os.getenv("WEEKLY_SEASONALITY", "True").lower() == "true"
YEARLY_SEASONALITY = os.getenv("YEARLY_SEASONALITY", "False").lower() == "true"

# Directory setup
MODELS_DIR = "trained_models"
LOG_DIR = "logs"
RAW_DATA_DIR = "raw_data"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"train_{datetime.now().strftime('%Y-%m-%d')}.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Fetch data from Prometheus with pagination ---
def fetch_metric_paginated(query, days=QUERY_DAYS, step=STEP_INTERVAL, batch_hours=BATCH_HOURS):
    """Fetch metric data in batches to handle Prometheus limits"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    print(f"📊 Fetching data for {days} days from {start_time} to {end_time}")
    print(f"🔍 Query: {query}")
    print(f"📏 Step interval: {step}")
    
    # Calculate number of batches
    total_hours = days * 24
    num_batches = (total_hours + batch_hours - 1) // batch_hours
    
    print(f"📦 Will fetch in {num_batches} batches of {batch_hours} hours each")
    
    all_dfs = []
    current_start = start_time
    batch_count = 0
    
    while current_start < end_time:
        batch_count += 1
        current_end = min(current_start + timedelta(hours=batch_hours), end_time)
        
        print(f"🔄 Processing batch {batch_count}/{num_batches}: {current_start} to {current_end}")
        
        try:
            url = f"{PROM_URL}/api/v1/query_range"
            params = {
                "query": query,
                "start": current_start.isoformat() + "Z",
                "end": current_end.isoformat() + "Z",
                "step": step,
            }
            
            print(f"🌐 Request URL: {url}")
            
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            result = r.json()["data"]["result"]
            
            print(f"✅ Batch {batch_count} successful, got {len(result)} time series")
            
            # Process each time series in this batch
            for series in result:
                dc = series["metric"].get("dc", "unknown")
                instance = series["metric"].get("instance", "unknown")
                
                df = pd.DataFrame(series["values"], columns=["ds", "y"])
                df["ds"] = pd.to_datetime(df["ds"], unit="s")
                df["y"] = pd.to_numeric(df["y"], errors="coerce")
                df = df.dropna().reset_index(drop=True)
                
                # Add metadata
                df["dc"] = dc
                df["instance"] = instance
                
                all_dfs.append(df)
            
            # Add delay between requests to avoid overwhelming Prometheus
            if current_start + timedelta(hours=batch_hours) < end_time:
                print(f"⏳ Waiting {FETCH_INTERVAL_SECONDS} seconds before next batch...")
                time.sleep(FETCH_INTERVAL_SECONDS)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for batch {batch_count}: {e}")
            logging.error(f"Request failed for batch {batch_count}: {e}")
        except Exception as e:
            print(f"❌ Error processing batch {batch_count}: {e}")
            logging.error(f"Error processing batch {batch_count}: {e}")
        
        # Move to next batch
        current_start = current_end
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"🎉 Finished fetching all batches. Total data points: {len(combined_df)}")
        return combined_df
    else:
        print("⚠️ No data fetched from any batch")
        return pd.DataFrame()

# --- Save raw data to CSV for debugging ---
def save_raw_data(metric, df):
    """Save raw data to CSV for debugging and analysis"""
    if df.empty:
        print(f"⚠️ No data to save for {metric}")
        return
    
    filename = os.path.join(RAW_DATA_DIR, f"{metric}_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(filename, index=False)
    print(f"💾 Saved raw data for {metric} to {filename}")
    logging.info(f"Saved raw data for {metric} to {filename}")

# --- Train and save Prophet model ---
def train_and_save(metric, dc, df):
    """Train a Prophet model and save it to disk"""
    if len(df) < MIN_DATA_POINTS:
        print(f"⚠️ Skipping {metric}-{dc}: insufficient data ({len(df)} points, need {MIN_DATA_POINTS})")
        logging.warning(f"Skipping {metric}-{dc}: insufficient data ({len(df)} points, need {MIN_DATA_POINTS})")
        return
    
    print(f"🧠 Training model for {metric}-{dc} with {len(df)} data points")
    
    try:
        # Prepare data for Prophet
        prophet_df = df[['ds', 'y']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Train model with settings from .env
        model = Prophet(
            daily_seasonality=DAILY_SEASONALITY, 
            weekly_seasonality=WEEKLY_SEASONALITY,
            yearly_seasonality=YEARLY_SEASONALITY
        )
        model.fit(prophet_df)
        
        # Create metric subdirectory if it doesn't exist
        metric_dir = os.path.join(MODELS_DIR, metric)
        os.makedirs(metric_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(metric_dir, f"{dc}.json")
        with open(model_path, "w") as f:
            f.write(model_to_json(model))
        
        print(f"✅ Trained and saved model for {metric}-{dc}: {model_path}")
        logging.info(f"Trained and saved model for {metric}-{dc}: {model_path}")
        
    except Exception as e:
        print(f"❌ Failed to train model for {metric}-{dc}: {e}")
        logging.error(f"Failed to train model for {metric}-{dc}: {e}")

# --- Main pipeline ---
def main():
    print("🚀 Starting Prometheus metrics export and model training")
    print(f"📊 Metrics to process: {list(QUERIES.keys())}")
    print(f"📅 Query days: {QUERY_DAYS}")
    print(f"⏱️ Step interval: {STEP_INTERVAL}")
    print(f"📦 Batch hours: {BATCH_HOURS}")
    print(f"📈 Minimum data points: {MIN_DATA_POINTS}")
    print(f"🔁 Fetch interval: {FETCH_INTERVAL_SECONDS} seconds")
    
    for metric, query in QUERIES.items():
        print(f"\n{'='*50}")
        print(f"📈 Processing metric: {metric}")
        print(f"{'='*50}")
        
        try:
            # Fetch data with pagination using settings from .env
            df = fetch_metric_paginated(query, days=QUERY_DAYS, step=STEP_INTERVAL, batch_hours=BATCH_HOURS)
            
            if df.empty:
                print(f"⚠️ No data fetched for {metric}, skipping")
                continue
            
            # Save raw data for debugging
            save_raw_data(metric, df)
            
            # Group by DC and train models
            dcs = df['dc'].unique()
            print(f"🏢 Found {len(dcs)} DCs: {list(dcs)}")
            
            for dc in dcs:
                dc_data = df[df['dc'] == dc]
                print(f"📊 DC {dc} has {len(dc_data)} data points")
                train_and_save(metric, dc, dc_data)
                
        except Exception as e:
            print(f"❌ Failed to process {metric}: {e}")
            logging.error(f"Failed to process {metric}: {e}")
    
    print("\n🎉 All metrics processed successfully!")

if __name__ == "__main__":
    main()

import os
import time
import logging
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from prophet.serialize import model_from_json
from prometheus_client import Gauge, start_http_server
from dotenv import load_dotenv
import threading

# ---------------- Load Environment Variables ----------------
load_dotenv()

PROM_URL = os.environ["PROM_URL"]
EXPORT_PORT_FORCAST = int(os.environ.get("EXPORT_PORT_FORCAST"))
METRICS = os.environ.get("METRICS").split(",")
QUERIES = {metric: os.environ.get(f"{metric.upper()}_QUERY") for metric in METRICS}

ANOMALY_MINUTES = int(os.environ.get("ANOMALY_MINUTES", 5))
FETCH_INTERVAL_SECONDS = int(os.environ.get("FETCH_INTERVAL_SECONDS", 30))
FORECAST_HOURS = [int(x) for x in os.environ.get("FORECAST_HOURS", "1,6,24").split(",")]
FORECAST_FREQUENCY_SECONDS = int(os.environ.get("FORECAST_FREQUENCY_SECONDS", 30))

MODELS_DIR_JSON = os.environ.get("MODELS_DIR_JSON")

# Global cache
model_cache = {}
forecast_storage = {}

# ---------------- Logging ----------------
LOG_DIR = "Log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"fifty_json{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    filemode='w',
    format="%(asctime)s %(filename)s:%(lineno)d | %(levelname)s |  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Logging initialized.")

# ---------------- Load Trained Model ----------------
def load_trained_model(metric, labels):
    instance = labels.get("instance", "default")
    dc = labels.get("dc") or labels.get("instance.dc", "default")
    cache_key = f"{metric}_{dc}_{instance}"

    model_filename = os.path.join(MODELS_DIR_JSON, metric, f"{dc}.json")
    logging.info(f"Model filename:{model_filename}")
    try:
        mtime = os.path.getmtime(model_filename)
        if cache_key in model_cache:
            model, cached_mtime = model_cache[cache_key]
            if mtime == cached_mtime:   # file unchanged
                logging.debug(f"Model {model_filename} unchanged, using cached")
                return model

        # ✅ FIX: read raw string, not dict
        with open(model_filename, "r") as f:
            model_json = f.read()
        model = model_from_json(model_json)

        model_cache[cache_key] = (model, mtime)
        logging.info(f"Reloaded model: {model_filename}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_filename}: {e}")
        return None

# ---------------- Fetch Latest Points ----------------
def fetch_latest_points(query):
    """Fetch the most recent point for ALL matching servers"""
    try:
        response = requests.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=10)
        if response.status_code != 200:
            logging.error(f"Prometheus query failed: {response.text}")
            return []
        results = response.json()["data"]["result"]
        points = []
        for item in results:
            metric_labels = item.get("metric", {})
            ts, val = item["value"]
            df = pd.DataFrame([[pd.to_datetime(float(ts), unit="s"), float(val)]],
                              columns=["ds", "y"])
            df.attrs['labels'] = metric_labels
            points.append((df, metric_labels))
        return points
    except Exception as e:
        logging.error(f"Error fetching latest data: {e}")
        return []

# ---------------- Forecast Generation ----------------
'''
def generate_forecasts(metric, model, label_values):
    try:
        now = datetime.utcnow()
        for h in FORECAST_HOURS:
            future_times = [now + timedelta(seconds=i * 30) for i in range(int((h * 3600) / 30))]
            if not future_times:
                continue
            future_df = pd.DataFrame({"ds": future_times})
            forecast = model.predict(future_df)
            latest = forecast.iloc[-1]
            pred = max(0, min(100, latest["yhat"]))
            lower = max(0, min(pred, latest["yhat_lower"]))
            upper = min(100, max(pred, latest["yhat_upper"]))
            gauges[metric][f"forecast_{h}h_predicted_value"].labels(**label_values).set(pred)
            gauges[metric][f"forecast_{h}h_lower_bound"].labels(**label_values).set(lower)
            gauges[metric][f"forecast_{h}h_upper_bound"].labels(**label_values).set(upper)
            logging.info(f"[{metric}] {h}h forecast: {pred:.2f} [{lower:.2f}, {upper:.2f}]")
    except Exception as e:
        logging.error(f"Forecast generation failed for {metric}: {e}")
'''
def generate_forecasts(metric, model, label_values):
    try:
        now = datetime.utcnow()

        for h in FORECAST_HOURS:
            # ask Prophet for exactly 1 point: now + horizon
            target_time = now + timedelta(hours=h)
            future_df = pd.DataFrame({"ds": [target_time]})
            forecast = model.predict(future_df).iloc[0]

            pred = clip_value(forecast["yhat"])
            lower = clip_value(forecast["yhat_lower"])
            upper = clip_value(forecast["yhat_upper"])

            # 🚨 backward shift: store this forecast at "now"
            gauges[metric][f"forecast_{h}h_predicted_value"].labels(**label_values).set(pred)
            gauges[metric][f"forecast_{h}h_lower_bound"].labels(**label_values).set(lower)
            gauges[metric][f"forecast_{h}h_upper_bound"].labels(**label_values).set(upper)

            logging.info(
                f"[{metric}] {h}h forecast (for {target_time}, stored at {now}): "
                f"{pred:.2f} [{lower:.2f}, {upper:.2f}]"
            )

    except Exception as e:
        logging.error(f"Forecast generation failed for {metric}: {e}")

# ---------------- Anomaly Detection ----------------

'''
def detect_anomaly(metric, df, model):
    try:
        future = pd.DataFrame({'ds': df['ds']})
        forecast = model.predict(future)
        actual = df["y"].iloc[-1]
        pred = forecast["yhat"].iloc[-1]
        lower = forecast["yhat_lower"].iloc[-1]
        upper = forecast["yhat_upper"].iloc[-1]
        anomaly = 1 if actual > upper else 0
        return actual, pred, lower, upper, anomaly
    except Exception as e:
        logging.error(f"Anomaly detection failed: {e}")
        return None, None, None, None, 0
'''
# ---------------- Anomaly persistence config ----------------
ANOMALY_REQUIRED_POINTS = (ANOMALY_MINUTES * 60) // FETCH_INTERVAL_SECONDS
anomaly_streak = {}   # key: (metric, instance, dc), value: count

def detect_anomaly(metric, df, model, labels):
    try:
        future = pd.DataFrame({'ds': df['ds']})
        forecast = model.predict(future)

        actual = float(df["y"].iloc[-1])
        pred = float(forecast["yhat"].iloc[-1])
        lower = float(forecast["yhat_lower"].iloc[-1])
        upper = float(forecast["yhat_upper"].iloc[-1])

        # use labels directly (more reliable than df)
        instance = labels.get("instance", "default")
        dc = labels.get("dc", "default")

        key = (metric, instance, dc)
        current_time = datetime.now()


        # check if outside bounds
        is_outside = (actual > upper) and (actual > 50)  # only care if >50%

        if is_outside:
            # Get current streak info or initialize
            streak_info = anomaly_streak.get(key, {'count': 0, 'start_time': None})
            
            if streak_info['count'] == 0:
                # Starting a new streak
                streak_info = {'count': 1, 'start_time': current_time}
            else:
                # Continuing existing streak
                streak_info['count'] += 1
            
            anomaly_streak[key] = streak_info
            
            # Check if streak has lasted at least 5 minutes
            streak_duration = (current_time - streak_info['start_time']).total_seconds() / 60
            has_5min_streak = streak_duration >= ANOMALY_MINUTES
            
            anomaly = 1 if has_5min_streak else 0
            
        else:
            # Reset streak if conditions not met
            anomaly_streak[key] = {'count': 0, 'start_time': None}
            anomaly = 0

        logging.info(
            f"[{metric}] (instance={instance}, dc={dc}) "
            f"Actual={actual:.2f}, Pred={pred:.2f}, "
            f"Bounds=[{lower:.2f}, {upper:.2f}], "
            f"Outside={is_outside}, Streak={anomaly_streak[key]['count']} points, "
            f"Duration={(current_time - anomaly_streak[key]['start_time']).total_seconds()/60 if anomaly_streak[key]['start_time'] else 0:.1f}min, "
            f"Anomaly={anomaly}"
        )

        return actual, pred, lower, upper, anomaly
        
    except Exception as e:
        logging.error(f"Anomaly detection failed for {metric}: {e}")
        return None, None, None, None, 0

# ---------------- Gauges ----------------
gauges = {}
def create_gauges(metric, label_columns):
    if metric in gauges:
        return
    gauges[metric] = {
        "anomaly": Gauge(f"{metric}_realtime_anomaly_value", "1 if anomaly detected, else 0", label_columns),
        "predicted": Gauge(f"{metric}_realtime_predicted_value", "Predicted value", label_columns),
        "actual": Gauge(f"{metric}_actual_value", "Actual value", label_columns),
        "lower": Gauge(f"{metric}_realtime_predicted_lower_bound", "Lower bound", label_columns),
        "upper": Gauge(f"{metric}_realtime_predicted_upper_bound", "Upper bound", label_columns),
    }
    for h in FORECAST_HOURS:
        gauges[metric][f"forecast_{h}h_predicted_value"] = Gauge(f"{metric}_forecast_{h}h_predicted_value", f"{h}h forecast mean", label_columns)
        gauges[metric][f"forecast_{h}h_lower_bound"] = Gauge(f"{metric}_forecast_{h}h_lower_bound", f"{h}h forecast lower", label_columns)
        gauges[metric][f"forecast_{h}h_upper_bound"] = Gauge(f"{metric}_forecast_{h}h_upper_bound", f"{h}h forecast upper", label_columns)


def clip_value(v):
    return max(0, min(100, v))

# ---------------- Realtime Loop ----------------
def run_realtime():
    logging.info("Realtime anomaly detection and forecasting started")
    while True:
        for metric in METRICS:
            points = fetch_latest_points(QUERIES[metric])
            for df, labels in points:
                model = load_trained_model(metric, labels)
                if model is None:
                    continue
                label_columns = list(labels.keys()) if labels else ["instance"]
                create_gauges(metric, label_columns)
                actual, pred, lower, upper, anomaly = detect_anomaly(metric, df, model,labels)
                if actual is None:
                    continue
                label_values = {col: str(labels.get(col, "")) for col in label_columns}
                gauges[metric]["actual"].labels(**label_values).set(clip_value(actual))
                gauges[metric]["predicted"].labels(**label_values).set(clip_value(pred))
                gauges[metric]["lower"].labels(**label_values).set(clip_value(lower))
                gauges[metric]["upper"].labels(**label_values).set(clip_value(upper))
                gauges[metric]["anomaly"].labels(**label_values).set(anomaly)
                generate_forecasts(metric, model, label_values)
                logging.info(f"[{metric}] {label_values} Actual: {actual:.2f}, Pred: {pred:.2f}, Bounds: [{lower:.2f}, {upper:.2f}], Anomaly={anomaly}")
        time.sleep(FETCH_INTERVAL_SECONDS)

# ---------------- Main ----------------
if __name__ == "__main__":
    logging.info("Starting JSON-based Anomaly Detection + Forecasting")
    start_http_server(EXPORT_PORT_FORCAST)
    logging.info(f"Prometheus metrics exposed on port {EXPORT_PORT_FORCAST}")
    run_realtime()


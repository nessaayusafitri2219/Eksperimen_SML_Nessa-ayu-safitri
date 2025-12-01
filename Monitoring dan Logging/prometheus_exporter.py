from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import random

# --- DEFINISI METRIK (Minimal 10) ---
# 1-3. Counters
REQUEST_COUNT = Counter('app_request_total', 'Total Requests')
ERROR_4XX = Counter('app_error_4xx_total', 'Client Errors')
ERROR_5XX = Counter('app_error_5xx_total', 'Server Errors')

# 4-6. Gauges
LAST_PREDICTION = Gauge('app_last_prediction_sales', 'Last predicted sales value')
CPU_USAGE = Gauge('sys_cpu_usage_pct', 'System CPU Usage')
MEMORY_USAGE = Gauge('sys_memory_usage_mb', 'System Memory Usage')
TEMP_INPUT = Gauge('feature_temperature_input', 'Input Temperature Feature')

# 7-8. Histograms
LATENCY = Histogram('app_latency_seconds', 'Response Latency')
PREDICTION_DIST = Histogram('app_prediction_dist', 'Distribution of predictions')

# 9-10. Business Metrics
HIGH_VALUE_SALES = Counter('biz_high_value_sales', 'Sales predicted > 100k')
LOW_VALUE_SALES = Counter('biz_low_value_sales', 'Sales predicted < 20k')

def simulate_traffic():
    while True:
        with LATENCY.time():
            # Simulasi Data
            sales_pred = random.uniform(10000, 150000)
            temp = random.uniform(30, 90)
            
            # Update Metrics
            REQUEST_COUNT.inc()
            LAST_PREDICTION.set(sales_pred)
            TEMP_INPUT.set(temp)
            PREDICTION_DIST.observe(sales_pred)
            
            CPU_USAGE.set(random.uniform(10, 60))
            MEMORY_USAGE.set(random.uniform(200, 800))
            
            # Logic Counter
            if sales_pred > 100000:
                HIGH_VALUE_SALES.inc()
            elif sales_pred < 20000:
                LOW_VALUE_SALES.inc()
            
            # Simulasi Error Random
            if random.random() < 0.02:
                ERROR_5XX.inc()
            
            print(f"Request processed. Pred: {sales_pred:.2f}")
            time.sleep(random.uniform(0.5, 2.0))

if __name__ == '__main__':
    # Start Server di port 8000
    start_http_server(8000)
    print("Prometheus Exporter running on port 8000...")
    simulate_traffic()

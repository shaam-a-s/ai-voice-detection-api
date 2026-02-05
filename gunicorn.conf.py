# Render Configuration for Free Tier Optimization
#
# This file tells Render how to run your service with minimal resource usage

# Use only 1 worker to minimize memory
# (Render sets WEB_CONCURRENCY=1 automatically, but we specify it here too)
workers = 1

# Use threads instead of multiple processes
worker_class = "uvicorn.workers.UvicornWorker"

# Increase timeout for slow audio processing
timeout = 120  # 2 minutes

# Reduce memory usage
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 10

# Preload app to share model across workers (though we only use 1)
preload_app = True

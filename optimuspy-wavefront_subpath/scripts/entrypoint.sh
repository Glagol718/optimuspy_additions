#!/bin/sh
# service rabbitmq-server start

python3 manage.py wait_for_db &&
python3 manage.py migrate
celery -A optimuspy worker --concurrency 1 -P solo --loglevel=info &
uvicorn optimuspy.asgi:application --host 0.0.0.0 --port 8000

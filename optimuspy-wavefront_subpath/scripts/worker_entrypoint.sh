#!/bin/sh
QUEUE=${CELERY_QUEUE:-"default"}
CONCURRENCY=${WORKER_CONCURRENCY:-1}
HOSTNAME="worker_${QUEUE}"

celery -A optimuspy worker \
    -Q "${QUEUE}" \
    --hostname="${HOSTNAME}" \
    --concurrency="${CONCURRENCY}" \
    -P prefork \
    --loglevel=info

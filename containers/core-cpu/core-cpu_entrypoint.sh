#!/bin/bash
source /app/venv/bin/activate

/app/prmon/bin/prmon --filename /app/training_data/logs/prmon.txt \
    --json-summary /app/training_data/logs/prmon.json --interval 10  \
    --log-filename prmon.log] [--interval 30] \
    --fast-memmon \
    -- exec "$@"
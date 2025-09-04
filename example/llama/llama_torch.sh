#!/bin/bash
# Source common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# expriment setting
export STALLOC_MODE=Torch
export SEGMENT=0
export GMLAKE=0

# Model parallelism and batch sizes
export MODEL_NAME=llama

export TRAIN_ITERS=20
export STALLOC_DYNAMIC=0

# Torch
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/naive.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/V.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/R.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/VR.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/ZR.sh

sleep 5s

#  Analyze all logs
BASE_LOG_DIR="${STALLOC_DIR}/STAlloc/log/${MODEL_NAME}"
echo "Scanning log directory: ${BASE_LOG_DIR}"

ana_total_dirs=$(find ${BASE_LOG_DIR} -maxdepth 1 -type d | wc -l)
ana_total_dirs=$((ana_total_dirs - 1))
current=1

for dir in ${BASE_LOG_DIR}/*; do
    if [ -d "$dir" ]; then
        log_name=$(basename "$dir")
        
        echo "[$current/$ana_total_dirs] Processing log: $log_name"
        
        export LOG_NAME=$log_name
        python ${STALLOC_DIR}/STAlloc/example/analyze/analyze.py --log_name $log_name
        
        echo "Completed analysis for $log_name"
        echo "-----------------------------------------"
        
        current=$((current + 1))
    fi
done

echo "All analyses completed! Processed $((current-1)) log directories."
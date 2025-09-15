#!/bin/bash
# Source common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# expriment setting
export STALLOC_MODE=Torch

# Model parallelism and batch sizes
export MODEL_NAME=llama

export TRAIN_ITERS=3
export STALLOC_DYNAMIC=0

# Torch mode for warm up
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/naive.sh
# bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/V.sh
# bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/R.sh
# bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/VR.sh
# bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/ZR.sh

# Trace
export STALLOC_MODE=Trace
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/naive.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/V.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/R.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/VR.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/ZR.sh

# Plan
PLAN_PATH="${STALLOC_DIR}/STAlloc/Synthesizer"
cd $PLAN_PATH
PLAN_PATH="${PLAN_PATH}/main.py"
BASE_LOG_DIR="${STALLOC_DIR}/STAlloc/allocator_case"
echo "Scanning log directory: ${BASE_LOG_DIR}"

total_dirs=$(find ${BASE_LOG_DIR} -maxdepth 1 -type d | wc -l)
total_dirs=$((total_dirs - 1))
current=1

for dir in ${BASE_LOG_DIR}/*; do
    if [ -d "$dir" ]; then
        allcator_case=$(basename "$dir")
        
        echo "[$current/$total_dirs] Processing log: $allcator_case"
        TRACE_PATH=${BASE_LOG_DIR}/$allcator_case
        
        python ${PLAN_PATH} --model-memory-dir=$TRACE_PATH
        
        echo "Plan for $allcator_case"
        echo "-----------------------------------------"
        
        current=$((current + 1))
    fi
done

echo "All Plan completed! Processed $((current-1)) log directories."

# Alloc
export STALLOC_MODE=Alloc
export TRAIN_ITERS=20
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/naive.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/V.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/R.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/VR.sh
bash ${STALLOC_DIR}/STAlloc/example/${MODEL_NAME}/ZR.sh

sleep 3s

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
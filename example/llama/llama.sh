#!/bin/bash
# Env setting
export PYTHONPATH=/workspace/Megatron-LM-080
export STALLOC_DIR=/workspace # path to STAlloc

# Dataset and tokenizer paths
export DATA_PATH=/dataset/RedPajama-Data-1T-Sample/RedPajama-Data-1T-Sample_text_document
export VOCAB_FILE=/dataset/gpt2/gpt2-vocab.json
export MERGE_FILE=/dataset/gpt2/gpt2-merges.txt
export TOKENIZER_PATH=/dataset/gpt2/llama2_tokenizer.model

# Model parallelism and batch sizes
export TP=4
export PP=2
export VPP=2  # virtual pipeline stages
export MBS=4
export GBS=128
export MODEL_SIZE=7  # 7, 13, 70, 130, tiny
export MODEL_NAME=llama

export SEGMENT=0
export GMLAKE=0

export TRAIN_ITERS=20
export STALLOC_DYNAMIC=0

# Torch
export STALLOC_MODE=Torch
export GRANULARITY=selective
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/naive.sh
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/V.sh

export RCP=16
export GRANULARITY=full
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/R.sh
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/VR.sh

# Trace
export STALLOC_MODE=Trace
export GRANULARITY=selective
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/naive.sh
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/V.sh

export RCP=16
export GRANULARITY=full
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/R.sh
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/VR.sh

# Plan
PLAN_PATH="${STALLOC_DIR}/stalloc/Synthesizer"
cd $PLAN_PATH
PLAN_PATH="${PLAN_PATH}/main.py"
BASE_LOG_DIR="${STALLOC_DIR}/stalloc/allocator_case"
echo "Scanning log directory: ${BASE_LOG_DIR}"

total_dirs=$(find ${BASE_LOG_DIR} -maxdepth 1 -type d | wc -l)
total_dirs=$((total_dirs - 1))
current=1

for dir in ${BASE_LOG_DIR}/*; do
    if [ -d "$dir" ]; then
        allcator_case=$(basename "$dir")
        
        echo "[$current/$total_dirs] Processing log: $allcator_case"
        TRACE_PATH=${BASE_LOG_DIR}/$allcator_case
        
        for device in {0..7}; do
            python ${PLAN_PATH} --model-memory-dir=$TRACE_PATH --device=$device &
        done
        
        echo "Plan for $allcator_case"
        echo "-----------------------------------------"
        
        current=$((current + 1))
    fi
done

wait

echo "All Plan completed! Processed $((current-1)) log directories."

# Alloc
export STALLOC_MODE=Alloc
export GRANULARITY=selective
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/naive.sh
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/V.sh

export RCP=16
export GRANULARITY=full
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/R.sh
bash ${STALLOC_DIR}/stalloc/example/${MODEL_NAME}/VR.sh

sleep 5s
#  Analyze all logs
BASE_LOG_DIR="${STALLOC_DIR}/stalloc/log/${MODEL_NAME}"
echo "Scanning log directory: ${BASE_LOG_DIR}"

ana_total_dirs=$(find ${BASE_LOG_DIR} -maxdepth 1 -type d | wc -l)
ana_total_dirs=$((ana_total_dirs - 1))
current=1

for dir in ${BASE_LOG_DIR}/*; do
    if [ -d "$dir" ]; then
        log_name=$(basename "$dir")
        
        echo "[$current/$ana_total_dirs] Processing log: $log_name"
        
        export LOG_NAME=$log_name
        python ${STALLOC_DIR}/stalloc/example/analyze/analyze.py --log_name $log_name
        
        echo "Completed analysis for $log_name"
        echo "-----------------------------------------"
        
        current=$((current + 1))
    fi
done

echo "All analyses completed! Processed $((current-1)) log directories."
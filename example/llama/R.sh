#!/bin/bash
set -ex

export PYTHONPATH=${PYTHONPATH:-"/workspace/Megatron-LM-080"}
export STALLOC_DIR=${STALLOC_DIR:-"/workspace"}
SRC_PATH=${PYTHONPATH}/pretrain_gpt.py
export PYTHONPATH=${PYTHONPATH}:${STALLOC_DIR}
DATA_PATH=${DATA_PATH}
VOCAB_FILE=${VOCAB_FILE}
MERGE_FILE=${MERGE_FILE}
TOKENIZER_PATH=${TOKENIZER_PATH}

# Env vars
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1  #PP
export NCCL_CROSS_NIC=0

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6366
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# note: Using fp16 makes the first several iterations have skipped iterations, after which the loss becomes normal.
# Using bf16 does not have this problem, the first iteration has normal loss.#      
TP=${TP:-4}
PP=${PP:-2}
MBS=${MBS:-4}
GBS=${GBS:-128}  # should be multiple of MBS*TP*PP
GRANULARITY=${GRANULARITY:-full}
RCP=${RCP:-16}
ITERS=${TRAIN_ITERS:-10}
SEQ=4096
MODEL_SIZE=${MODEL_SIZE:-7}  # 7, 13, 70, 130, tiny
GMLAKE=${GMLAKE:-0}
SEGMENT=${SEGMENT:-0}
export STALLOC_MODE=${STALLOC_MODE:-Torch}  # Alloc, Trace, Torch
export STALLOC_TRACE_FAST_MODE=${TRACE_MODE:-1}  # may lead to OOM
export STALLOC_DYNAMIC=${STALLOC_DYNAMIC:-0}
# export STALLOC_DYNAMIC_UTILIZATION=90
# export STALLOC_DYNAMIC_DIFF=1
export STALLOC_LOG_LEVEL=0
export STALLOC_STATIC_FALLBACK=1
export STALLOC_LIB_PATH=${STALLOC_DIR}/STAlloc/Allocator

MODEL_TAG=llama-${MODEL_SIZE}b_WS${WORLD_SIZE}_TP${TP}_PP${PP}_MBS${MBS}_GBS${GBS}_SEQ${SEQ}_${GRANULARITY}_RCP${RCP}
MEMORY_SAVED_DIR=${STALLOC_DIR}/STAlloc/allocator_case
export STALLOC_MODEL_INFO_PATH=${MEMORY_SAVED_DIR}/${MODEL_TAG}
if [ "$STALLOC_MODE" == "Trace" ]; then
    if [ -e "${STALLOC_MODEL_INFO_PATH}/trace" ]; then
       rm -rf ${STALLOC_MODEL_INFO_PATH}/trace
    fi
    mkdir -p ${STALLOC_MODEL_INFO_PATH}/trace
    mkdir -p ${STALLOC_MODEL_INFO_PATH}/output
    ITERS=3
elif [ "$STALLOC_MODE" == "Alloc" ]; then
    export STALLOC_LOG_LEVEL=3
    if [ ! -e "${STALLOC_MODEL_INFO_PATH}/output/plan" ]; then
       exit 1
    fi
fi

if   [ ${MODEL_SIZE} == 7 ];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 13 ];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 70 ];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=16; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 130 ];  then HIDDEN_SIZE=12288;  NUM_HEAD=96; NUM_QUERY_GROUP=8;  NUM_LAYERS=88; FFN_HIDDEN_SIZE=31232; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == "tiny" ]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi


if [ "$GRANULARITY" == "full" ]; then
       RECOMPUTE_ARGS="
              --recompute-method block \
              --recompute-num-layers ${RCP} \
       "
else
       RECOMPUTE_ARGS="
       "
fi

if [ "$GMLAKE" == "1" ]; then
    export vmmDefragment=1
    export autoGC=10000
    export fragLimit=536870912
    export reuseLimit=10
    export defragLevel=0 # 0-greedy  1-lazy
    # export GMLAKE_INFO=INFO
    LOG_NAME=gmlake
fi

if [ "$SEGMENT" == "1" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    LOG_NAME=segment
fi

if [ "$STALLOC_MODE" == "Alloc" ]; then
    LOG_NAME=alloc
fi
if [ "$STALLOC_MODE" == "Trace" ]; then
    LOG_NAME=trace
fi
if [ "$STALLOC_MODE" == "Torch" ]; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    LOG_NAME=torch_${TORCH_VERSION}
fi

LOG_PATH=${STALLOC_DIR}/STAlloc/log/llama/${MODEL_TAG}/${LOG_NAME}.log
mkdir -p ${STALLOC_DIR}/STAlloc/log/llama/${MODEL_TAG}


CMD="torchrun $DISTRIBUTED_ARGS \
       ${SRC_PATH} \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --sequence-parallel \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --micro-batch-size ${MBS} \
       --global-batch-size ${GBS} \
       --seq-length ${SEQ} \
       --max-position-embeddings ${SEQ} \
       --norm-epsilon ${NORM_EPS} \
       --group-query-attention \
       --train-iters ${ITERS} \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --split 95,5,0 \
       --distributed-backend nccl \
       --no-gradient-accumulation-fusion \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --bf16 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --no-masked-softmax-fusion \
       --no-position-embedding \
       --swiglu \
       --tokenizer-model ${TOKENIZER_PATH} \
       --transformer-impl local \
       --normalization RMSNorm \
       --use-legacy-models \
       --attention-softmax-in-fp32 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --recompute-granularity ${GRANULARITY} \
       --use-flash-attn \
       --log-throughput \
       ${RECOMPUTE_ARGS} \
       "


echo ${CMD} | tee ${LOG_PATH}
${CMD} 2>&1 | tee -a ${LOG_PATH}

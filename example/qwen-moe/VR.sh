#!/bin/bash

# CUDA
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ENABLE_SAME_RAND_A100=1
export MHA_BWD_NO_ATOMIC_F64=1
export MAX_JOBS=20

export PYTHONPATH=${PYTHONPATH:-"/workspace/Megatron-LM-010"}
export STALLOC_DIR=${STALLOC_DIR:-"/workspace"}
SRC_PATH=${PYTHONPATH}/pretrain_gpt.py
export PYTHONPATH=${PYTHONPATH}:${STALLOC_DIR}
DATA_PATH=${DATA_PATH}
VOCAB_FILE=${VOCAB_FILE}
MERGE_FILE=${MERGE_FILE}
TOKENIZER_PATH=${TOKENIZER_PATH}

export OMP_NUM_THREADS=10
# Distributed training variables
NNODES=${NNODES:-1}
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

# Parallelism and performance variables
TP=4
PP=2
TPE=1
EP=4
VPP=${VPP:-2}
TD=None
RCP=12
GRANULARITY=full
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=128

TRAIN_ITERS=${TRAIN_ITERS:-10}

SEED=${SEED:-1234}
# Network size variables

## ORIG:
MODEL_SIZE=Qwen-moe
NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
FFN_HIDDEN_SIZE=5632
MOE_FFN_HIDDEN_SIZE=1408
MAX_POSITION_EMBEDDINGS=8192
EXTRA_VOCAB_SIZE=2399
RMS_NORM_EPS=1e-6
MAX_SEQ_LEN=4096
MAX_PAD_LEN=4096

# moe
NUM_EXPERTS=60
ROUTER_TOPK=4
NUM_SHARED_EXPERTS=4
FIRST_K_DENSE_REPLACE=1

## trains
TRAIN_TOKENS=1000000000
WARMUP_TOKENS=10000
# TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${MAX_SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${MAX_SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${MAX_SEQ_LEN} ))
LR=1e-5
MIN_LR=1e-6

GMLAKE=${GMLAKE:-0}
SEGMENT=${SEGMENT:-0}
export STALLOC_MODE=${STALLOC_MODE:-Torch}
export STALLOC_TRACE_FAST_MODE=1  # may lead to OOM
export STALLOC_DYNAMIC=1
export STALLOC_DYNAMIC_UTILIZATION=90
export STALLOC_DYNAMIC_DIFF=512
export STALLOC_LOG_LEVEL=3
export STALLOC_STATIC_FALLBACK=1
export STALLOC_LIB_PATH=${STALLOC_DIR}/STAlloc/Allocator
export STALLOC_REUSE_STATIC=1


MODEL_TAG=${MODEL_SIZE}-tp${TP}pp${PP}ep${EP}tpe${TPE}mbs${MICRO_BATCH_SIZE}gbs${GLOBAL_BATCH_SIZE}_RCP${RCP}_VPP${VPP}
MEMORY_SAVED_DIR=${STALLOC_DIR}/STAlloc/allocator_case
export STALLOC_MODEL_INFO_PATH=${MEMORY_SAVED_DIR}/${MODEL_TAG}
if [ "$STALLOC_MODE" == "Trace" ]; then
    if [ -e "${STALLOC_MODEL_INFO_PATH}/trace" ]; then
       rm -rf ${STALLOC_MODEL_INFO_PATH}/trace
    fi
    mkdir -p ${STALLOC_MODEL_INFO_PATH}/trace
    mkdir -p ${STALLOC_MODEL_INFO_PATH}/output
    TRAIN_ITERS=3
elif [ "$STALLOC_MODE" == "Alloc" ]; then
    export STALLOC_LOG_LEVEL=3
    if [ ! -e "${STALLOC_MODEL_INFO_PATH}/output/plan" ]; then
       exit 1
    fi
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

BASE_PATH=${STALLOC_DIR}
LOG_PATH=${BASE_PATH}/log/${MODEL_NAME}/${MODEL_TAG}/${LOG_NAME}.log
mkdir -p ${BASE_PATH}/log/${MODEL_NAME}/${MODEL_TAG}

ROPE_THETA=10000
SCALE_FACTOR=40

PARALLEL_PERFORMANCE_ARGS=" \
    --sequence-parallel \
    --use-distributed-optimizer \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size ${TPE} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --moe-token-dispatcher-type alltoall \
    --recompute-granularity $GRANULARITY \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    "

MIXED_PRECISION_ARGS="
    --bf16 \
    "

MOE_ARGS=" \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${ROUTER_TOPK} \
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE} \
    --moe-shared-expert-intermediate-size $(($MOE_FFN_HIDDEN_SIZE * $NUM_SHARED_EXPERTS)) \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
    --moe-router-pre-softmax \
    "

OTHER_NETWORK_ARGS=" \
    --use-mcore-models \
    --disable-bias-linear \
    --tokenizer-model ${TOKENIZER_PATH} \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --swiglu \
    --normalization LayerNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --use-rotary-position-embeddings \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --position-embedding-type rope \
    --untie-embeddings-and-output-weights \
    --rotary-base ${ROPE_THETA} \
    --rotary-scaling-factor ${SCALE_FACTOR} \
    --rotary-seq-len-interpolation-factor 1 \
    "

NETWORK_SIZE_ARGS="  \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --seq-length ${MAX_SEQ_LEN} \
    --ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE} \
    "

TRAINING_ARGS=" \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --eval-interval 10000 \
    --eval-iters 1 \
    "

LEARNING_ARGS=" \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --lr-decay-iters ${LR_DECAY_ITERS} \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --init-method-std 0.008 \
    --seed ${SEED} \
    "

LOAD_SAVE_ARGS=" \
    --no-load-optim \
    --no-load-rng \
    --num-workers 8 \
    --no-save-optim \
    "
LOGGING_ARGS=" \
    --log-interval 1 \
    --log-throughput \
    --save-interval 500 \
    "

DATASET_ARGS=" \
    --data-path ${DATA_PATH} \
    --split 99,1,0 \
    "

LAUNCHER=" \
    torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    "

export EXPERT_PER_GPU=$((${NUM_EXPERTS}/${EP}))
CMD="${LAUNCHER} ${SRC_PATH} \
    ${PARALLEL_PERFORMANCE_ARGS} \
    ${MIXED_PRECISION_ARGS} \
    ${MOE_ARGS} \
    ${MLA_ARGS} \
    ${OTHER_NETWORK_ARGS} \
    ${NETWORK_SIZE_ARGS} \
    ${TRAINING_ARGS} \
    ${LEARNING_ARGS} \
    ${LOAD_SAVE_ARGS} \
    ${LOGGING_ARGS} \
    ${DATASET_ARGS} \
    ${RECOMPUTE_ARGS} \
    "


echo ${CMD} | tee -a ${LOG_PATH}
${CMD} 2>&1 | tee -a ${LOG_PATH}

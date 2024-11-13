#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=$1 #<Specify path>
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH=$5 #<Specify path and file prefix>_text_document

CHECKPOINT_PATH=/pscratch/sd/k/klhhhhh/Megatron/huggingface/llama3.1_8b
TENSORBOARD_LOGS_PATH=/pscratch/sd/k/klhhhhh/Megatron/gpt/tensorboard
VOCAB_FILE=/pscratch/sd/k/klhhhhh/Megatron/gpt/cache/gpt2-vocab.json
MERGE_FILE=/pscratch/sd/k/klhhhhh/Megatron/gpt/cache/gpt2-merges.txt
DATA_PATH=/pscratch/sd/k/klhhhhh/wiki/my-gpt2-wiki_text_document

TOKENIZER_MODEL=meta-llama/Llama-3.1-8B
CHECKPOINT_DIR=/pscratch/sd/k/klhhhhh/Megatron/huggingface/llama3.1_8b

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --seq-length 8192
    --max-position-embeddings 8192
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --exit-on-missing-checkpoint
    --use-checkpoint-args
    --no-load-optim
    --no-load-rng
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --position-embedding-type rope
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --disable-bias-linear
    --transformer-impl transformer_engine
    --group-query-attention
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --rotary-base 500000
    --rotary-percent 1.0
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --swiglu
    --bf16
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1536 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 8
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

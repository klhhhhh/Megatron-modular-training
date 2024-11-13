TRAIN_DATA="/pscratch/sd/k/klhhhhh/dataset/RACE/train/middle"
VALID_DATA="/pscratch/sd/k/klhhhhh/dataset/RACE/dev/middle \
            /pscratch/sd/k/klhhhhh/dataset/RACE/dev/high"
TOKENIZER_MODEL=meta-llama/Llama-3.1-8B
CHECKPOINT_DIR=/pscratch/sd/k/klhhhhh/Megatron/huggingface/iter_0000001/mp_rank_00
CHECKPOINT_PATH=/pscratch/sd/k/klhhhhh/Megatron/huggingface/llama3.1_7b/run
TP=1

COMMON_TASK_ARGS="--tensor-model-parallel-size ${TP} \
              --pipeline-model-parallel-size 1 \
              --seq-length 8192 \
              --max-position-embeddings 8192 \
              --tokenizer-type HuggingFaceTokenizer \
              --tokenizer-model ${TOKENIZER_MODEL} \
              --load ${CHECKPOINT_DIR} \
              --exit-on-missing-checkpoint \
              --use-checkpoint-args \
              --no-load-optim \
              --no-load-rng \
              --untie-embeddings-and-output-weights \
              --normalization RMSNorm \
              --position-embedding-type rope \
              --no-masked-softmax-fusion \
              --attention-softmax-in-fp32 \
              --disable-bias-linear \
              --transformer-impl transformer_engine \
              --group-query-attention 8 \
              --attention-dropout 0.0 \
              --hidden-dropout 0.0 \
              --rotary-base 500000 \
              --rotary-percent 1.0 \
              --ffn-hidden-size 14336 \
              --num-attention-heads 32 \
              --swiglu \
              --bf16"

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --save-interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 1000 \
                      --eval-iters 10 \
                      --weight-decay 1.0e-1"

python /global/homes/k/klhhhhh/Megatron-modular-training/tasks/race/finetune.py \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --epochs 3 \
       --micro-batch-size 4 \
       --lr 1.0e-5 \
       --lr-warmup-fraction 0.06
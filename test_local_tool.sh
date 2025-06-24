#ibstatus
nvidia-smi topo -m
set -x

export WANDB_NAME='hyper_config'
export WANDB_PROJECT='reasoningrl-tool-joykirat'
export WANDB_TOKEN='c8f694b1460eaf8f06beec994e5aa1bb56183688'
export EXPERIMENT_NAME='qwen_dummy_unified'
export WANDB_HOST='https://api.wandb.ai'

export PROJECT_NAME=$WANDB_PROJECT
# export PRETRAINED="/data/checkpoints/phi-4-o3-sft-04_12_25_32k"
export PRETRAINED="Qwen/Qwen2.5-7B-Instruct"
export DATA_TRAIN="/home/aiscuser/ReasoningRL/verl/scripts/data/tool_dataset/train.parquet"
export DATA_TEST="/home/aiscuser/ReasoningRL/verl/scripts/data/tool_dataset/test.parquet"
export OUTPUT_DIR="data/log_dir/${EXPERIMENT_NAME}/output"
export GPUS=4
export NODES=1
export CUDA_LAUNCH_BLOCKING=1
export RAY_DEDUP_LOGS=0

# Set run variables
export RUN_N=4
export PPO_EPOCHS=4
export MAX_RESPONSE_LENGTH=32768
#export MAX_RESPONSE_LENGTH=20480 # hacked 20k
export PPO_MAX_TOKEN_LENGTH=32768
export PER_GPU_TRAIN_BATCH_SIZE=2
export PER_GPU_VAL_BATCH_SIZE=2
#export BATCH_SIZE=1024
export LR=1e-7
export GRADCLIP=0.2
export KLCOEFF=0.001
export ROLLOUT_TEMP=1.0
export EPOCHS=1
export TENSOR_PARALLEL_SIZE=2
export ULYSSES_PARALLEL_SIZE=1
export FP8_ADAM=true
export FP8_KVCACHE=true

if [ -z $AMLT_OUTPUT_DIR]; then
    export AMLT_OUTPUT_DIR=$OUTPUT_DIR 
fi

# Run the script
chmod +x ./run_tool_azure.sh
bash ./run_tool_azure.sh
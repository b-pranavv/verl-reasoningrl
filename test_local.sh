#ibstatus
nvidia-smi topo -m
set -x

export PROJECT_NAME=reasoning_mojavaheripi 
export EXPERIMENT_NAME="test_local1"
export PRETRAINED="/data/checkpoints/phi-4-o3-sft-04_12_25_32k"
export DATA_TRAIN="/data/phi-4-reasoning-plus-data/phi_math_new/train.parquet"
export DATA_TEST="/data/phi-4-reasoning-plus-data/phi_math_new/test.parquet"
export OUTPUT_DIR="/data/log_dir/${EXPERIMENT_NAME}/output"
export GPUS=4
export NODES=1
export TENSORBOARD_DIR=/data/tensorboard_logs/${EXPERIMENT_NAME}
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
chmod +x ./run_exp_azure.sh
bash ./run_exp_azure.sh
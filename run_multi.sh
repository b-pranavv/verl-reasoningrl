ibstatus
ray stop
nvidia-smi topo -m

set -x

export AZUREML_COMPUTE_USE_COMMON_RUNTIME=true
export VLLM_ATTENTION_BACKEND=XFORMERS
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NODES=1
export GPUS=8
# HF_TOKEN: $HF_TOKEN
export WANDB_NAME='hyper_config'
export WANDB_PROJECT='reasoning'
export WANDB_TOKEN='c8f694b1460eaf8f06beec994e5aa1bb56183688'
export EXPERIMENT_NAME='phi_sandbox'
export WANDB_HOST='https://api.wandb.ai'

echo "Running on $HOSTNAME"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "AMLT_OUTPUT_DIR=$AMLT_OUTPUT_DIR"

# installing verification libraries
python -m pip install "math-verify[antlr4-9-3]"
pip install --upgrade torchdata

# Set run variables
export RUN_N=8
export PPO_EPOCHS=4
# export DATASET_NAME="phi_math_tool"
export DATASET_NAME="phi_math_new"
# export MAX_RESPONSE_LENGTH=16384
# export MAX_RESPONSE_LENGTH=20992
# export MAX_RESPONSE_LENGTH=25600
export MAX_RESPONSE_LENGTH=32768
# export MAX_RESPONSE_LENGTH=36864
# export MAX_RESPONSE_LENGTH=40960
# export MAX_RESPONSE_LENGTH=49152
# export MAX_RESPONSE_LENGTH=65536
# export BASE_MODEL="phi-4"
# export BASE_MODEL="phi-4-o3-sft-long"
# export BASE_MODEL="models/phi-4-o3-sft-32"
# export BASE_MODEL="models/phi-4-o3-sft-4_1_25"
# export BASE_MODEL="models/phi-4-o3-sft-4_1_25_long"
export BASE_MODEL="phi-4-o3-sft-04_12_25_32k"
# export BASE_MODEL="models/phi-4-o3-sft-04_13_25_32k"
# export BASE_MODEL="phi-4-o3-sft-4_1_25_128k"
export PPO_MAX_TOKEN_LENGTH=32768
# export PPO_MAX_TOKEN_LENGTH=36864
# export PPO_MAX_TOKEN_LENGTH=40960
# export PPO_MAX_TOKEN_LENGTH=65536
export PPO_BATCH_SIZE=$((NODES*8*2)) # This is batchsize of ppo
export TRAIN_BATCH_SIZE=$((PPO_BATCH_SIZE)) # This is batchsize of the data loader
export LR=1e-7
export TENSOR_PARALLEL_SIZE=2
export ULYSSES_PARALLEL_SIZE=1
# export ULYSSES_PARALLEL_SIZE=2
export SAVE_FREQ=5
export TEST_FREQ=10
export FP8_ADAM=true
export FP8_KVCACHE=true
export RAY_DEDUP_LOGS=0

pip install -q vllm==0.8.1
pip install -q tensordict==0.6.2
pip install -q transformers==4.47.1
pip install -q fastapi langchain azure-identity langchain-openai python-dotenv langchainhub langchain-azure-dynamic-sessions

# older params
export WANDB_API_KEY=$WANDB_TOKEN
export WANDB_PROJECT=$WANDB_PROJECT
export PRETRAINED='/mnt/models/models/phi-4'
export DATA_TRAIN='/mnt/data/data/phi_math_new/train.parquet'
export DATA_TEST='/mnt/data/data/phi_math_new/test.parquet'

# if node rank is 0, start ray as head
if [ $NODE_RANK -eq 0 ]; then
    ray start --head --port=$MASTER_PORT
    sleep 60
else
    # wait for ray head to start
    sleep 10
    ray start --address=$MASTER_ADDR:$MASTER_PORT --block

    # finish with a sleep to keep the process alive
    echo "Ray should have started, sleeping..."
    sleep infinity
fi

# export RAY_ADDRESS="http://$MASTER_ADDR:$MASTER_PORT"

# check if ray is running on all nodes
ray status

# Run the script
chmod +x ./run_python.sh
bash ./run_python.sh
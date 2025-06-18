#ibstatus
nvidia-smi topo -m
set -x

echo "Running on $HOSTNAME"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODES=$NODES"
echo "GPUS=$GPUS"

# Set run variables
export RUN_N=8
export PPO_EPOCHS=4
export MAX_RESPONSE_LENGTH=32768
#export MAX_RESPONSE_LENGTH=20480 # hacked 20k
export PPO_MAX_TOKEN_LENGTH=32768
export PER_GPU_TRAIN_BATCH_SIZE=16
export PER_GPU_VAL_BATCH_SIZE=16
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
export RAY_DEDUP_LOGS=0

if [ -z $AMLT_OUTPUT_DIR]; then
    export AMLT_OUTPUT_DIR=$OUTPUT_DIR 
fi

# if node rank is 0, start ray as head
if [ $NODE_RANK -eq 0 ]; then
    ray start --head --port=$MASTER_PORT
    sleep 60
else
    # wait for ray head to start
    sleep 10
    ray start --address=$MASTER_ADDR:$MASTER_PORT

    # graceful automatic ray client exit
    while true; do
        sleep 300 # check every 5 mins

        status_code=$(ray status > /dev/null 2>&1; echo $?)

        # ray cluster down
        if [ $status_code -ne 0 ]; then
            echo "Ray cluster is down. Exiting..."
            exit 0 # do not remove this line
        fi
    done
fi

# below executed by ray head only
# check if ray is running on all nodes
ray status

# Run the script
chmod +x ./run_exp_azure.sh
bash ./run_exp_azure.sh

# stop the cluster
ray stop
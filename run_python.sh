git config --global credential.helper cache

echo "AMLT_JOB_NAME=$AMLT_JOB_NAME"
export RUN_NAME=$AMLT_JOB_NAME

# login to huggingface and wandb
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login --host $WANDB_HOST $WANDB_TOKEN

# Set run variables
    # actor_rollout_ref.model.path=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/../../cap/data-capability/wd/INPUT_inputs/PHI_training/assets/$BASE_MODEL \

CMD="python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=scripts/data/numina_dataset/train.parquet \
    data.val_files=scripts/data/eval_dataset/AMC.parquet \
    data.train_batch_size=$((TRAIN_BATCH_SIZE)) \
    data.val_batch_size=$((2*NODES*8)) \
    data.max_prompt_length=3000 \
    data.max_response_length=$((MAX_RESPONSE_LENGTH-3000)) \
    reward_model.reward_manager=python \
    actor_rollout_ref.model.path=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/../../cap/data-capability/wd/INPUT_inputs/PHI_training/assets/$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((PPO_BATCH_SIZE)) \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$((ULYSSES_PARALLEL_SIZE)) \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((PPO_MAX_TOKEN_LENGTH)) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=$((PPO_EPOCHS)) \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$((TENSOR_PARALLEL_SIZE)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_RESPONSE_LENGTH+3000)) \
    actor_rollout_ref.rollout.name=vllm_with_python \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$((RUN_N)) \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_turns=${MAX_TURNS} \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_generations_to_log_to_wandb=0 \
    trainer.project_name="reasoning" \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$((GPUS)) \
    trainer.nnodes=$((NODES)) \
    trainer.save_freq=$((SAVE_FREQ)) \
    trainer.test_freq=$((TEST_FREQ)) \
    trainer.val_before_train=False \
    trainer.default_local_dir=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/../../cap/data-capability/wd/INPUT_inputs/PHI_training/assets/verl_checkpoints/$EXPERIMENT_NAME/checkpoints \
    trainer.save_rollout_path=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/../../cap/data-capability/wd/INPUT_inputs/PHI_training/assets/verl_checkpoints/$EXPERIMENT_NAME/rollouts \
    trainer.total_epochs=1"


# trainer.default_local_dir=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/../../cap/data-capability/wd/INPUT_inputs/PHI_training/assets/$EXPERIMENT_NAME/checkpoints \
# trainer.save_rollout_path=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/../../cap/data-capability/wd/INPUT_inputs/PHI_training/assets/$EXPERIMENT_NAME/rollouts \


if [ "$FP8_ADAM" = true ]; then
    CMD="$CMD \
    +actor_rollout_ref.actor.optim.eight_bit=True"
fi

if [ "$FP8_KVCACHE" = true ]; then
    CMD="$CMD \
    actor_rollout_ref.rollout.kv_cache_dtype=\"fp8\""
fi

CMD="$CMD $@"

eval $CMD
    # actor_rollout_ref.model.path=/mnt/models/phi-4 \
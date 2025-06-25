##git config --global credential.helper cache

# login to huggingface and wandb
##huggingface-cli login --token $HF_TOKEN --add-to-git-credential
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login --host $WANDB_HOST $WANDB_TOKEN

CMD="python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_TRAIN \
    data.val_files=$DATA_TEST \
    data.train_batch_size=$((PER_GPU_TRAIN_BATCH_SIZE*NODES*GPUS)) \
    data.val_batch_size=$((PER_GPU_VAL_BATCH_SIZE*NODES*GPUS)) \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.max_prompt_length=5000 \
    data.max_response_length=$((MAX_RESPONSE_LENGTH-5000)) \
    data.use_tool=True \
    data.prompt_key=question \
    data.prompt_template_name=re_tool_qwen_template_sys \
    reward_model.reward_manager=tool \
    actor_rollout_ref.model.path=$PRETRAINED \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.grad_clip=$GRADCLIP \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((PER_GPU_TRAIN_BATCH_SIZE*NODES*GPUS)) \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$((ULYSSES_PARALLEL_SIZE)) \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((PPO_MAX_TOKEN_LENGTH)) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KLCOEFF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=$((PPO_EPOCHS)) \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$((TENSOR_PARALLEL_SIZE)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_RESPONSE_LENGTH+5000)) \
    actor_rollout_ref.rollout.name=vllm_with_tool \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$((RUN_N)) \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    algorithm.kl_ctrl.kl_coef=$KLCOEFF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$((GPUS)) \
    trainer.nnodes=$((NODES)) \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$AMLT_OUTPUT_DIR/checkpoints \
    trainer.total_epochs=$EPOCHS \
    trainer.log_val_generations=5"

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
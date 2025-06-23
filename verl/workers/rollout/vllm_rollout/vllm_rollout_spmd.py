# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union
import re

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
import json

import time
import asyncio
from typing import Any, Union
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, pad_sequence_to_length

from verl.workers.rollout.base import BaseRollout
from vllm.lora.request import LoRARequest
from verl.workers.rollout.vllm_rollout.python_executor_sandbox import executeCode
from verl.workers.rollout.vllm_rollout.tool_router import ToolRouter


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        lora_kwargs = kwargs.pop('lora_kwargs', {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            swap_space=80,
            # quantization="fp8",
            # cpu_offload_gb=40,
            kv_cache_dtype=config.kv_cache_dtype,
            calculate_kv_scales=(config.kv_cache_dtype == "fp8"),
            enable_prefix_caching=False,
            preemption_mode="swap",
            # enable_prefix_caching=True, ##### this is in new verl
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            temperature=1.,
            max_tokens=config.response_length,
            # repetition_penalty=1.,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()
        else:
            self.inference_engine.wake_up()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                # "top_k": -1, # reasoning_rl comments
                # "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id=lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}",lora_int_id=lora_int_id,lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                'rollout_log_probs': rollout_log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        else:
            time_now = time.time()
            # print("waiting for engine to sleep {}".format(time_now))
            self.inference_engine.sleep(level=1)
            # print("engine is sleeping {}".format(time_now))

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

class vLLMRolloutWithTool(vLLMRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

    
    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            call_json = json.loads(tool_call_str)
            func_name = call_json['name']
            arguments = call_json.get('arguments', {})
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            return f"Parse tool call failed: {e}"

    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r'<tool_call>', output_str)
        end_tags = re.findall(r'</tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []
    
    def extract_classes(self, input_str):
        """Extract class names from the input string, it would be present as [Classes Involved: {classes_involved}]"""
        pattern = r'\[Classes Involved: (.*?)\]'
        match = re.search(pattern, input_str)
        if match:
            classes_str = match.group(1)
            classes_str_alphabets = re.sub(r'[^a-zA-Z\s]', '', classes_str)
            classes_list = [cls.strip() for cls in classes_str_alphabets.split(' ')]
            return classes_list
        else:
            return []

    def batch_execute(self, tool_router_list: List[ToolRouter], env_list: List[str] = None):
        """Execute a batch of tool calls."""
        tool_response_list = []
        print(f"Tool Router List: {tool_router_list}")
        
        if len(env_list) == 0:
            env_list = [None] * len(tool_router_list)
        else:
            print(f"Env List: {env_list}")
            
            assert len(tool_router_list) == len(env_list), "tool_router_list and env_list must have the same length."
        
        # Execute tool calls in parallel
        for tool_router, env in zip(tool_router_list, env_list):
            tool_response = tool_router.execute_tool_calls(env=env)
            tool_response_list.append(tool_response)

        return tool_response_list

    # def batch_execute(self, env_list: List[str], tool_calls_list: List[List[str]], classes_list: List[List[str]]):
    #     results_list = []
    #     for tool_calls, classes, initial_config in zip(tool_calls_list, classes_list, env_list):
    #         executor = BfclExecutor(timeout_seconds=8, capture_stdout=True)
    #         response, _ = executor.execute_list(tool_calls, classes, initial_config)
    #         # Only return the latest response
    #         results_list.append(response[-1])
    #     return results_list

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:        
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = ori_input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]

            # if there are envs, prepare n copies for each env
            env_list = None
            if 'env' in prompts.non_tensor_batch:
                env_list = []
                for env in prompts.non_tensor_batch['env']:
                    for _ in range(self.sampling_params.n):
                        env_list.append(env)
            
            elif 'initial_config' in prompts.non_tensor_batch:
                env_list = []
                for env in prompts.non_tensor_batch['initial_config']:
                    for _ in range(self.sampling_params.n):
                        env_list.append(env)

            # track the status of each input
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            # collect the result mask of each rollout, 1 for non-result, 0 for tool call result or pad
            result_mask_list = [[] for _ in range(len(curr_inputs))]

            # generate until all inputs are completed
            for step in range(self.config.max_turns):
                if len(active_indices) == 0:
                    break

                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                
                with self.update_sampling_params(
                    n=1, 
                    max_tokens=min(512, max(active_max_tokens)),
                    stop=['</tool_call>'],
                    top_p=0.99,
                    detokenize=True,
                ):  # 512 at most, and add <|im_start|> as stop for corner case
                    vllm_inputs = [{
                        'prompt_token_ids': raw_prompt_ids
                    } for raw_prompt_ids in active_inputs]
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=True
                    )

                # collect all tool calls
                tool_router_list: List[ToolRouter] = []
                # extract_classes_list: List[List[str]] = []
                call_indices: List[int] = []

                # process each output
                new_active_indices = []
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[i].outputs[0].token_ids
                    finish_reason = outputs[i].outputs[0].finish_reason
                    stop_reason = outputs[i].outputs[0].stop_reason
                    
                    if finish_reason == 'stop' and isinstance(stop_reason, str) and '</tool_call>' in stop_reason:
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)

                        output_str = self.tokenizer.decode(output_ids, skip_special_tokens=False)
                        full_str = self.tokenizer.decode(curr_inputs[idx], skip_special_tokens=False)
                        
                        print("full_str: ", full_str)
                        
                        '''
                        Removing first two calls because they come from the prompt (will need change if prompt changes)
                        ['', '[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]',
                        '''
                        ## TODO: Check how many tools calls should be removed based on the prompt being used
                        
                        toolRouter = ToolRouter(full_str, output_str)
                        
                        print("Tool calls: ", toolRouter.tool_calls)
                        
                        if toolRouter.tool_call_present():
                            tool_router_list.append(toolRouter)
                            call_indices.append(idx)
                            new_active_indices.append(idx)
                        else:
                            pass
                            

                        # if tool_calls:
                        #     extract_classes_list.append(self.extract_classes(full_str))
                        #     tool_calls_list.append()
                        #     tool_calls_list.append(tool_calls)
                        #     call_indices.append(idx)
                        #     new_active_indices.append(idx)
                        # else:
                        #     pass # no tool calls
                    elif finish_reason == 'length':
                        # output over max tokens
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and (stop_reason == None or stop_reason == self.tokenizer.pad_token_id): # 151644 is the id of <|im_start|>, is a illigal stop, we stop here
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    else:
                        raise ValueError(f"unknown stop reason. finish_reason: {finish_reason}, stop_reason: {stop_reason}")

                # batch process tool calls
                if tool_router_list:
                    # Only tp_rank 0 executes the tools
                    if self.tp_rank == 0:
                        active_env_list = [env_list[i] for i in call_indices if env_list is not None]
                        tool_responses_list = self.batch_execute(tool_router_list, active_env_list)
                        
                        # Prepare data for broadcasting
                        broadcast_data = {
                            'tool_router_list': tool_router_list,
                            'call_indices': call_indices,
                            'tool_responses_list': tool_responses_list
                        }
                    else:
                        broadcast_data = None
                    
                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
                    
                    # All ranks process the broadcasted data
                    if broadcast_data is not None:
                        tool_router_list = broadcast_data['tool_router_list']
                        call_indices = broadcast_data['call_indices']
                        tool_responses_list = broadcast_data['tool_responses_list']

                        for idx, tool_router, tool_responses in zip(call_indices, tool_router_list, tool_responses_list):
                            tool_response_str = f"<tool_response>\n{tool_responses}\n</tool_response>\n"
                            print("Tool Response: ", tool_response_str)
                            output_ids = self.tokenizer.encode(tool_response_str, add_special_tokens=False)
                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [0] * len(output_ids)

                # check if need to truncate, if yes, truncate, and remove from active; if no, update curr_max_tokens
                length_checked_active_indices = []
                for idx in active_indices:
                    assert len(curr_inputs[idx]) - len(init_inputs[idx]) == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        curr_inputs[idx] = init_inputs[idx] \
                            + curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                active_indices = length_checked_active_indices

            output_ids_list = []
            # collect the all rollouts
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    idx = i * self.sampling_params.n + j
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])

        response_attention_mask_list = []
        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            # to tensor 
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            # response attention mask, 1 for valid, 0 for invalid
            response_attention_mask = torch.ones_like(response, dtype=torch.int64)
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_attention_mask_list.append(response_attention_mask)
            # response, pad to response_length
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_list.append(response)
            # result mask, 1 for non-result, 0 for result or pad
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            result_mask_list_padded.append(result_mask)
        response_attention_mask = torch.stack(response_attention_mask_list, dim=0)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        if self.config.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                
        # concat attenion_mask for input and response
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        return DataProto(batch=batch)
    
    
class vLLMRolloutWithPython(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        self.gen_str = "\n<|im_start|>assistant\n<think>"
        self.gen_ids = self.tokenizer.encode(self.gen_str)

    def is_json(self, s):
        # print(f"string in is_json: {s}[string end]")
        try:
            json.loads(s)
            return True
        except Exception as e:
            # print(e)
            # print(f'not a json string: {s}[string end]')
            try:
                s = json.dumps(eval(s))
                json.loads(s)
                return True
            except:
                return False

    def parse_json(self, s):
        try:
            data = json.loads(s)
        except ValueError:
            s = json.dumps(eval(s))
            data = json.loads(s)

        return data

    def execute_query(self, query):

        print("Query: ", query)

        if self.is_json(query):
            parsed = self.parse_json(query)
            if 'name' not in parsed or 'arguments' not in parsed:
                return "Wrong tool format. Name or argument key missing."
            tool_arguments = parsed['arguments']
            if isinstance(tool_arguments, dict):
                arguments = parsed['arguments']
            elif self.is_json(tool_arguments):
                arguments = self.parse_json(tool_arguments)
            else:
                return "Wrong arguments format."

            if 'code' not in arguments:
                return 'code key is missing from the arguments.'

            print("Arguments: ", arguments)
            code = arguments['code']


            code_execution = executeCode([code])

            print("code_Execution: ", code_execution)

            code_execution = code_execution[0]

            success = code_execution['success']

            if(success):
                return 'Compiled successfully. Output: ' + code_execution['output']
            else:
                return 'Compilation error: ' + code_execution['error']
        return "Wrong tool format. Tool call should be in json format."

    def batch_executor(self, query: Union[str, List[str]]) -> List[str]:
        if len(query) == 0:
            return 'invalid query'

        results = []
        for q in query:
            results.append(self.execute_query(q))

        return results

    def _formatOutput(self, result):

        if(result[1] == 'Success'):
            if(len(result[0]) == 0):
                return 'Compiled successfully. However print statement is missing at the end of the python code, therefore output is empty.'
                # return 'Compiled successfully. Print empty'

            else:
                return 'Compiled successfully. Output: ' + result[0]
        else:
            return 'Compilation error: ' + result[1]

    def extract_python_code(self, text: str) -> str:
        try:
            start_tag = '<tool_call>'
            end_tag = '</tool_call>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return "" 

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        else:
            self.inference_engine.wake_up()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = ori_input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]

            # if there are envs, prepare n copies for each env
            env_list = None
            if 'env' in prompts.non_tensor_batch:
                env_list = []
                for env in prompts.non_tensor_batch['env']:
                    for _ in range(self.sampling_params.n):
                        env_list.append(env)

            # track the status of each input
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            # collect the result mask of each rollout, 1 for non-result, 0 for tool call result or pad
            result_mask_list = [[] for _ in range(len(curr_inputs))]

            # generate until all inputs are completed
            for step in range(20):
                print("Step: {}, active inputs: {}".format(step, len(active_indices)))
                if len(active_indices) == 0:
                    break

                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]

                with self.update_sampling_params(
                    n=1, 
                    max_tokens=max(active_max_tokens),
                    stop=['</tool_call>'],
                    top_p=0.99,
                    detokenize=True,
                ):  # 512 at most, and add <|im_start|> as stop for corner case
                    vllm_inputs = [{
                        'prompt_token_ids': raw_prompt_ids
                    } for raw_prompt_ids in active_inputs]
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,
                        sampling_params=self.sampling_params,
                        use_tqdm=True
                    )

                # collect all tool calls
                tool_calls_list: List[List[str]] = []
                call_indices: List[int] = []

                # process each output
                new_active_indices = []
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[i].outputs[0].token_ids
                    finish_reason = outputs[i].outputs[0].finish_reason
                    stop_reason = outputs[i].outputs[0].stop_reason

                    if finish_reason == 'stop' and isinstance(stop_reason, str) and '</tool_call>' in stop_reason:
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)

                        output_str = self.tokenizer.decode(output_ids)
                        tool_calls: List[str] = self.extract_python_code(output_str)
                        if tool_calls:
                            tool_calls_list.append(tool_calls)
                            call_indices.append(idx)
                            new_active_indices.append(idx)
                        else:
                            pass # no tool calls
                    elif finish_reason == 'length':
                        # output over max tokens
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and (stop_reason == None or stop_reason == self.tokenizer.pad_token_id): # 151644 is the id of <|im_start|>, is a illigal stop, we stop here
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    else:
                        raise ValueError(f"unknown stop reason. finish_reason: {finish_reason}, stop_reason: {stop_reason}")

                # batch process tool calls
                if tool_calls_list:
                    # Only tp_rank 0 executes the tools
                    if self.tp_rank == 0:
                        python_results = self.batch_executor(tool_calls_list)

                        # Prepare data for broadcasting
                        broadcast_data = {
                            'python_queries': tool_calls_list,
                            'python_indices': call_indices,
                            'python_results': python_results
                        }
                    else:
                        broadcast_data = None

                    broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)

                    # All ranks process the broadcasted data
                    if broadcast_data is not None:
                        python_queries = broadcast_data['python_queries']
                        python_indices = broadcast_data['python_indices']
                        python_results = broadcast_data['python_results']

                        for idx, result in zip(python_indices, python_results):
                            # update the output, add the search result
                            output_ids = self.tokenizer.encode(f" <tool_result>\n{result}\n</tool_result>")
                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [0] * len(output_ids)

                # check if need to truncate, if yes, truncate, and remove from active; if no, update curr_max_tokens
                length_checked_active_indices = []
                for idx in active_indices:
                    assert len(curr_inputs[idx]) - len(init_inputs[idx]) == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        curr_inputs[idx] = init_inputs[idx] \
                            + curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                active_indices = length_checked_active_indices

            output_ids_list = []
            # collect the all rollouts
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    idx = i * self.sampling_params.n + j
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])

        response_attention_mask_list = []
        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            # to tensor 
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            # response attention mask, 1 for valid, 0 for invalid
            response_attention_mask = torch.ones_like(response, dtype=torch.int64)
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_attention_mask_list.append(response_attention_mask)
            # response, pad to response_length
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_list.append(response)
            # result mask, 1 for non-result, 0 for result or pad
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            result_mask_list_padded.append(result_mask)
        response_attention_mask = torch.stack(response_attention_mask_list, dim=0)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        if self.config.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # concat attenion_mask for input and response
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
    
class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import json

class PythonRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, save_path=None, rewardType=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.save_path = save_path
        self.rewardType = rewardType
        self.reward_fn_key = reward_fn_key


    def __call__(self, data: DataProto, curr_save_path=None, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        if curr_save_path is not None:
            save_path = curr_save_path
        else:
            save_path = self.save_path

        print("Saving reward data to: ", save_path)

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path is not None:
            save_file = open(save_path, 'a')

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            score = self.compute_score(
                data_source=data_source,
                # tokenizer=self.tokenizer,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            # breakpoint()
            # if isinstance(score, dict):
            reason = score['reason']
            if(self.rewardType == 'val'):
                score = score['answer']
                if(score <= 0):
                    score = 0
                else:
                    score = 1
            elif(self.rewardType == 'train'):
                score = score['format'] + score['answer'] + score['tool']
            # else:
            #     reason = ''
            if score is None:
                print(f"score is None, data_source: {data_source}, sequences_str: {sequences_str}, ground_truth: {ground_truth}")
                print(f"reason: {reason}")
                score = 0
            reward_tensor[i, valid_response_length - 1] = score

            if save_path is not None:
                save_json_line = {
                    'data_source': data_source,
                    'sequences_str': sequences_str,
                    'ground_truth': ground_truth,
                    'score': score,
                    'reason': reason
                }
                save_file.write(json.dumps(save_json_line, ensure_ascii=False) + '\n')

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print('-' * 20)
                print(f"data_source: \n{data_source}")
                print(f"sequences_str: \n{sequences_str}")
                print(f"ground_truth: \n{ground_truth}")
                print(f"score: \n{score}")  
                print(f"reason: \n{reason}")
                print('-' * 20)

        if save_path is not None:
            save_file.close()

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {}
            }
        else:
            return reward_tensor, {}
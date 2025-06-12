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
Preprocess the nq dataset to parquet format
"""


import re
import os
from datasets import load_dataset, Dataset


from verl.utils.hdfs_io import copy, makedirs
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName', type=str, default='math-500')


    args = parser.parse_args()
    datasetName = args.datasetName
    data_source = datasetName




    if(datasetName == 'math-500'):
        dataset = load_dataset('HuggingFaceH4/MATH-500')


        test_data = dataset['test']


        questions = test_data['problem']
        solutions = test_data['solution']


    elif(datasetName == 'GSM8K'):
        dataset = load_dataset("openai/gsm8k", "main")


        test_data = dataset['test']


        questions = test_data['question']
        solutions = test_data['answer']


    elif(datasetName == 'AIME'):
        dataset = load_dataset('AI-MO/aimo-validation-aime')


        test_data = dataset['train']
        questions = test_data['problem']
        solutions = test_data['solution']


    elif(datasetName == 'AMC'):
        dataset = load_dataset('AI-MO/aimo-validation-amc')


        test_data = dataset['train']
        questions = test_data['problem']
        solutions = test_data['answer']
        solutions = [str(s) for s in solutions]


    elif(datasetName == 'Olympiad'):
        dataset = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP")


        test_data = dataset['train']
        questions = test_data['question']
        solutions = test_data['final_answer']


        solutions = [s[0] for s in solutions]


    else:
        raise ValueError("Invalid datasetName")




    # create dataset from the questions and solutions json


    dataset = {
        'problem': questions,
        'answer': solutions
    }


    # create a dataset from the json
    dataset = Dataset.from_dict(dataset)




    def make_map_fn(split):


        def process_fn(example, idx):


            re_python_template_sys = """You are **Phi**, a reasoning language model trained by Microsoft. Your job is to reach precise answers through careful reasoning and tool use when needed.

# RESPONSE FORMAT

<think>
... step-by-step reasoning ...
<tool_call>{"name":"<tool-name>","arguments":"<json-string-of-parameters>"}</tool_call>
<tool_result>{response}</tool_result>
... (repeat reasoning / tool call / tool result as needed) ...
</think>
<answer> final solution for the user (no tool use here) </answer>
<|im_end|>

**Structure Rules:**
- All reasoning goes between <think> and </think> (thinking block). 
- Within the thinking block, whenever a tool would improve your answer, prefer invoking it using <tool_call>...</tool_call> instead of relying solely on memory.
- Issue one valid <tool_call>...</tool_call> at a time; further tool calls can be sequentially interleaved throughout the reasoning process.
- Close </think> only when reasoning is fully complete.  
- Only the text after </think> is shown to user - this must be be concise, accurate, and self-contained. Never reference the thought block or invoke tool calls in this block.
- Provide the final answer for the user inside the <answer> </answer> tags.

# AVAILABLE TOOLS

```json
[
  {
    "type": "function", 
    "function": {
      "name": "run_python",
      "description": "Execute Python code. Available packages: numpy, scipy, sympy, pandas, matplotlib, requests",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "Python code to execute. Use print() for output"
          },
          "new_session: {
            "type": "bool",
            "description": "Whether to start a new python session rather than reuse the persistent session for this task; Default is False."
          },                
          "libraries": {
            "type": "array",
            "description": "A list of third-party libraries used in the code (e.g., ['numpy', 'pandas']). Optional, only needed for clarity or reproducibility.",
            "items": {
              "type": "string"
            }              
          }
        },        
        "required": ["code"]
      }
    }
  }
]
```

**Tool Calling Rules:**
- Format: Begin each real call on its own line as  
  <tool_call>{"name": "<function-name>", "arguments": "<json-string-of-parameters>"}</tool_call>
  Example  
  <tool_call>{"name": "run_python", "arguments": "{\"code\": \"print(2+2)\"}"}</tool_call>  
- Schema fidelity (strict): <function-name> must exactly match one of the function names in the provided schema; <json-string-of-parameters> must be a valid JSON string whose keys and value types match the function's "parameters" specification; escape characters correctly inside the <json-string-of-parameters> so it can be parsed correctly (\\" for quotes, \\n for newlines, \\\\ for backslashes, etc.).
- **Do not invent** tools or arguments that are not defined in the schema.
- After each valid tool call (starting at a new line), the platform injects the response within <tool_result>...</tool_result>; continue reasoning only after it appears.  
- Any <tool_call>...</tool_call> not begining at a new line is treated as ordinary text and no <tool_result>...</tool_result> is provided.

**Error Handling:**
If a tool fails, the platform injects a JSON error object such as <tool_result>{"error":"<ERROR_STRING>"}</tool_result>. Phi should analyize error in thinking block and either retry with fixes (using a new <tool_call>) or switch approaches if tools fails.
 """


            example['question'] = example['problem'].strip()


            solution = {
                "target": example['answer'],
            }


            data = {
            "data_source": data_source,
            "prompt": [{'role': 'system', 'content': re_python_template_sys},
                       {'role': 'user', 'content': example['question']}],
            "ability": "fact-reasoning",
            "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
            "extra_info": {
                'split': split,
                    'index': idx,
            }
        }
            return data


        return process_fn


    test_dataset = dataset.map(function=make_map_fn('test'), with_indices=True)


    test_dataset.to_parquet(os.path.join('./eval_dataset', f'{datasetName}.parquet'))
    # # Then, split it into two parts
    # split_ratio = 0.5  # or whatever ratio you want
    # split = test_dataset.train_test_split(test_size=split_ratio, seed=42)


    # # You now have two datasets
    # test_part_1 = split['train']
    # test_part_2 = split['test']
    # test_part_1.to_parquet(f'eval_dataset/{datasetName}_part1.parquet')


    # # Step 3: Split test_part_2 into 3 chunks
    # n = len(test_part_2)
    # chunk_size = n // 3


    # chunks = []
    # for i in range(3):
    #     start = i * chunk_size
    #     end = (i + 1) * chunk_size if i < 2 else n  # Make sure last chunk includes any leftovers
    #     chunk = test_part_2.select(range(start, end))
    #     chunks.append(chunk)


    # # Step 4: Save each chunk
    # for idx, chunk in enumerate(chunks):
    #     chunk.to_parquet(f'eval_dataset/{datasetName}_part2_chunk{idx+1}.parquet')
    # # sample 100 from the test dataset
    # test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)


    # train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
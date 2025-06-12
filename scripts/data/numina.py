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
import datasets


from verl.utils.hdfs_io import copy, makedirs
import argparse
from math_verify import parse, LatexExtractionConfig, ExprExtractionConfig, verify








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./numina_dataset')


    args = parser.parse_args()


    data_source = 'numina'


    dataset = datasets.load_dataset("AI-MO/NuminaMath-CoT")


    train_dataset = dataset['train']


    # split the train dataset into train and test
    train_dataset = train_dataset.train_test_split(test_size=0.001, seed=2)
    test_dataset = train_dataset['test']
    train_dataset = train_dataset['train']




    def extract_answer(
    solution: str,
    match_types: list[str] = ["latex", "expr"],
    precision: int = 6,
    strict: bool = True,
):
        """Helper function to compare strings using the math extraction metrics"""
        # Convert string match_types to ExtractionTarget objects
        extraction_targets = []
        for match_type in match_types:
            if match_type == "latex":
                extraction_targets.append(LatexExtractionConfig(boxed_match_priority=0))
            elif match_type == "expr":
                extraction_targets.append(ExprExtractionConfig())


        parsed = parse(solution, extraction_targets)
        return parsed


    def make_map_fn(split):


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




        def process_fn(example, idx):
            example['question'] = example['problem'].strip()


            solution = {
                "target": example['solution'],
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




    def filter_fn(solution):
        if('\\boxed' not in solution):
            return False


        # count of \\boxed in solution
        count = solution.count('\\boxed')
        if count > 1:
            # breakpoint()
            return False
        try:
            extracted = extract_answer(solution)
            # breakpoint()
            if extracted is None or extracted[1] in ['', 'A', 'B', 'C', 'D', '(A)', '(B)', '(C)', '(D)']:
                return False
        except Exception as e:
            print(f"Error in extracting answer: {e}")
            # breakpoint()
            return False
        return True


    def batch_filter(batch_example):
        return [filter_fn(solution) for solution in batch_example['solution']]


    print(f"train_dataset: {len(train_dataset)}")
    print(f"test_dataset: {len(test_dataset)}")


    train_dataset = train_dataset.shuffle(seed=42).select(range(20000))


    train_dataset = train_dataset.filter(batch_filter, batched=True)
    test_dataset = test_dataset.filter(batch_filter, batched=True)




    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)


    # sample 50 from the test dataset
    test_dataset = test_dataset.shuffle(seed=42).select(range(50))
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)


    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
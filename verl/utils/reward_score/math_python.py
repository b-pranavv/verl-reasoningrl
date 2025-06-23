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

import re
import string
import random
from math_verify import parse, LatexExtractionConfig, ExprExtractionConfig, verify


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# def validate_format(text: str) -> tuple[bool, str]:
#     # check if <think></think>, <answer></answer> is paired
#     if text.count('<think>') != text.count('</think>'):
#         return False, "<think> </think> not paired"

#     if text.count('<think>') == 0 or text.count('</think>') == 0:
#         return False, "<think> or </think> not found"

#     if text.count('<answer>') != 1 or text.count('</answer>') != 1:
#         return False, "<answer> or </answer> not found"        

#     # check the order of search/result
#     current_pos = 0
#     while True:
#         python_pos = text.find('<python>', current_pos)
#         if python_pos == -1:
#             break

#         output_pos = text.find('<output>', python_pos)
#         python_end_pos = text.find('</python>', python_pos)
#         output_end_pos = text.find('</output>', output_pos)

#         if -1 in (output_pos, python_end_pos, output_end_pos):
#             return False, "python/output tags are incomplete"

#         if not (python_pos < python_end_pos < output_pos < output_end_pos):
#             return False, "python/output tags are nested in the wrong order"

#         current_pos = output_end_pos

#     # check if \boxed{} is in the answer
#     answer_start = text.find('<answer>')
#     answer_end = text.find('</answer>')
#     if answer_start > answer_end:
#         return False, "<answer> must be before </answer>"
#     # answer_content = text[answer_start:answer_end]
#     # if '\\boxed{' not in answer_content or '}' not in answer_content:
#         # return False, "answer is missing \\boxed{} format"

#     return True, "format is correct"

def extract_output(text: str) -> str:
    """Extract the output from the text."""
    current_pos = 0
    output_text = []
    while True:
        output_pos = text.find('<tool_result>', current_pos)
        if output_pos == -1:
            break
        output_end_pos = text.find('</tool_result>', output_pos)
        if output_end_pos == -1:
            break
        output_content = text[output_pos + len('<tool_result>'):output_end_pos]
        output_text.append(output_content)
        current_pos = output_end_pos + len('</tool_result>')

    return output_text

def getToolReward(output_snippet):
    """Get the reward from the output snippet."""
    if(len(output_snippet) == 0):
        return 0, "No tool used"
    n = len(output_snippet) # total tool usage
    m = 0 # total compilation success
    for i in range(n):
        if 'Compiled successfully' in output_snippet[i]:
            m += 1

    return (m / n) * 1.0, f"total tool usage: {n}, total compilation sucess: {m}, tool score: {(m / n) * 1.0}"

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) == 0:
        return None

    # If there are 2 or more matches, return the last one
    answer = matches[-1].group(1).strip()
    return answer

def compare_strings(
    gold: str,
    pred: str,
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

    gold_parsed = parse(gold, extraction_targets)
    pred_parsed = parse(pred, extraction_targets)
    return verify(gold_parsed, pred_parsed, float_rounding=precision, strict=strict), gold_parsed, pred_parsed


def getCorrectnessReward(text, gold_output):
        answer = extract_solution(text)
        if(answer is None):
            return 0, "cannot extract answer"
        try:
            if_correct, gold_parsed, answer_parsed = compare_strings(answer, gold_output)
            if if_correct:
                return 2, "output is correct"
            else:
                return 0, "output is incorrect"
        except Exception as e:
            print("Error comparing string")
            return 0, "error comparing string"


def softFormatReward(text):
    count = 0
    if text.count('<think>') == text.count('</think>'):
        count += 0.125
    if text.count('<tool_call>') == text.count('</tool_call>'):
        count += 0.125
    if text.count('<answer>') == 1:
        count += 0.125
        # count -= len(text.split("</answer>")[-1])*0.001
    if text.count('</answer>') == 1:
        count += 0.125
        # count -= (len(text.split("</answer>")[-1]) - 1)*0.001

    return count

def hardFormatReward(text: str) -> tuple[bool, str]:


    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"        

    # check the order of search/result
    current_pos = 0
    while True:
        think_pos = text.find('<think>', current_pos)
        if think_pos == -1:
            break
        think_end_pos = text.find('</think>', think_pos)
        if think_end_pos == -1:
            return False, "<think> </think> not paired"

        python_pos = text.find('<tool_call>', think_pos)
        if python_pos == -1:
            break

        python_end_pos = text.find('</tool_call>', python_pos)
        if python_end_pos == -1:
            return False, "<tool_call> </tool_call> not paired"

        output_pos = text.find('<tool_result>', python_pos)
        if output_pos == -1:
            break
        output_end_pos = text.find('</tool_result>', output_pos)
        if output_end_pos == -1:
            return False, "<tool_result> </tool_result> not paired"

        if not (think_pos < python_pos < python_end_pos < output_pos < output_end_pos < think_end_pos):
            return False, "think/tool_call/tool_result tags are nested in the wrong order"
        current_pos = output_end_pos


    # check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    # answer_content = text[answer_start:answer_end]
    # if '\\boxed{' not in answer_content or '}' not in answer_content:
    #     return False, "answer is missing \\boxed{} format"

    return True, "format is correct"


def compute_score_math(solution_str, ground_truth, method='strict', format_score=0.5, score=1.):
    """ The scoring function for math problems.
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # print(f"solution_str: {solution_str}")

    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
        solution_str = solution_str_split[1]
    elif "<|im_start|>assistant<|im_sep|>" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant<|im_sep|>")
        solution_str = solution_str_split[1]

    else:
        ### raise error
        print("Solution_str: ", solution_str)
        raise ValueError("solution_str is not in the correct format")





    hardFormatScore, hardFormatReason = hardFormatReward(solution_str)
    if(hardFormatScore == False):
        hardFormatScore = 0
    else:
        hardFormatScore = 0.5

    softFormatScore = softFormatReward(solution_str)

    toolScore, toolReason = getToolReward(extract_output(solution_str))

    correctnessScore, correctnessReason = getCorrectnessReward(solution_str, ground_truth)


    reason = f"correctness score: {correctnessScore}, hard format score: {hardFormatScore}, soft format score: {softFormatScore}, tool score: {toolScore}, correctness reason: {correctnessReason}, format reason: {hardFormatReason}, tool reason: {toolReason}"


    return {
        'answer': correctnessScore,
        'format': hardFormatScore + softFormatScore,
        'tool': toolScore,
        "reason": reason}
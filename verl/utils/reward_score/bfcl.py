import ast
import re
from copy import deepcopy
from verl.workers.rollout.vllm_rollout.bfclEnv.bfcl_env import BfclEnv 
# from verl.workers.rollout.vllm_rollout.envs.bfcl_env import BfclEnv

def response_checker(
    model_response_list: list, ground_truth_response_list: list):
    """
    Checks if the model_response is a subsequence of the ground_truth_response.
    Each list contains the response of the function calls executed in that single turn.
    """
    # We don't need to enforce the order of the responses, because many entries have parallel operations, and so the model can execute them in any order.
    is_subsequence, missing_items = _is_subsequence_unordered(
        ground_truth_response_list, model_response_list
    )
    if not is_subsequence:
        return {
            "valid": False,
            "error_message": f"Model response execution results so far does not contain all the ground truth response execution results for turn.",
            "error_type": "multi_turn:execution_response_mismatch",
            "details": {
                "missing_items": missing_items,
                "model_response (including all previous turns)": model_response_list,
                "ground_truth_response (only the current turn)": ground_truth_response_list,
            },
        }

    return {"valid": True}


def method_invoke_order_checker(model_instances: dict, ground_truth_instances: dict):
    """
    Checks if the model_instance called the same order of methods as the ground_truth_instance.
    model_instance can call additional methods, but not skip any method that the ground_truth_instance called.

    Note: Currently, this functions only checks for the method names and not the arguments.
    """
    for class_name, ground_truth_instance in ground_truth_instances.items():
        model_instance = model_instances[class_name]

        # The get_method_called method is added by the LoggingMeta metaclass automatically
        model_invoke_order = model_instance.get_method_called()
        ground_truth_invoke_order = ground_truth_instance.get_method_called()

        # Extract the method names
        model_invoke_order = [method_call["method"] for method_call in model_invoke_order]
        ground_truth_invoke_order = [
            method_call["method"] for method_call in ground_truth_invoke_order
        ]

        is_subsequence, missing_items = _is_subsequence(
            ground_truth_invoke_order, model_invoke_order
        )
        if not is_subsequence:
            return {
                "valid": False,
                "error_message": f"Model instance for {class_name} does not match the method invoke order with ground truth instance. Missing items: {missing_items}",
                "error_type": "multi_turn:method_invoke_order_mismatch",
            }

    return {"valid": True}


#### Helper functions ####


def _compare_instances(model_obect, ground_truth_object):
    """
    Checks if the model_object has the same attributes as the ground_truth_object. They are instances of the same class.
    """
    assert type(model_obect) == type(
        ground_truth_object
    ), "Objects are not of the same type."
    differences = {}
    valid = True
    for attr_name in vars(ground_truth_object):
        # We don't check for private attributes
        if attr_name.startswith("_"):
            continue
        model_attr = getattr(model_obect, attr_name)
        ground_truth_attr = getattr(ground_truth_object, attr_name)

        if model_attr != ground_truth_attr:
            valid = False
            differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}

    return valid, differences


def _is_subsequence(list1, list2) -> tuple[bool, list]:
    """
    Checks if list1 is a subsequence of list2, i.e., all elements of list1 are present in list2 in the same order.
    Also returns the elements of list1 that are not present in list2.
    """
    # Convert list2 to an iterator to ensure that the elements are consumed only once.
    iter_list2 = iter(list2)
    return all(item in iter_list2 for item in list1), [
        item for item in list1 if item not in list2
    ]


def _is_subsequence_unordered(list1, list2) -> tuple[bool, list]:
    """
    Checks if all elements of list1 are present in list2, regardless of order.
    Also returns the elements of list1 that are not present in list2.
    """
    # Copy list2 to avoid modifying the original list during checks
    list2_copy = list2[:]
    
    # Check each item in list1 to see if it exists in list2_copy
    missing_elements = []
    for item in list1:
        try:
            # Attempt to remove one occurrence of `item` from list2_copy to handle duplicates
            list2_copy.remove(item)
        except ValueError:
            # If item is not found, add it to missing_elements
            missing_elements.append(item)
    
    # If there are missing elements, list1 is not a subsequence of list2
    is_subsequence = len(missing_elements) == 0
    return is_subsequence, missing_elements

def extract_classes(input_str):
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

def validate_tool_calls(output_str):
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


def extract_tool_calls(output_str):
    if not validate_tool_calls(output_str):
        return []

    try:
        pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
        matches = re.finditer(pattern, output_str, re.DOTALL)
        
        return [match.group(1).strip() for match in matches]
    except Exception as e:
        return []

def parse_list_of_tool_calls(outer_list):
    # Remove the outer brackets and split by '),'
    func_calls = [re.findall(r'(\w+\([^\)]*\))', inner_list) for inner_list in outer_list]

    # Convert each function call to a list of strings
    result = [call.strip() for inner_list in func_calls for call in inner_list]

    return result


def format_reward_score(solution_str):
    # <|im_start|>assistant\n<think>It seems 'Documents' is a directory, not a file. Let's find the first file in the current directory and display its last line.</think>\n<tool_call> [ls(a=True)] </tool_call><|im_end|>
    assistant_start_list = solution_str.split('<|im_start|>assistant')[1:]
    assistant_msg_list = [msg.split('<|im_end|>')[0] for msg in assistant_start_list][:-1]
    pattern = re.compile(r'<think>.*?</think>\s*<tool_call>.*?</tool_call>', re.DOTALL)

    count_matches = 0
    for msg in assistant_msg_list:
        count_matches += 1 if bool(pattern.search(msg)) else 0
    
    format_reward = min(1, count_matches / (len(assistant_msg_list))) if len(assistant_msg_list) > 0 else 0
    return format_reward

def execute_list(queries, classes, initial_config):
    bfcl_env = BfclEnv()
    execution_results, involved_instances = bfcl_env.execute_multi_turn_func_call(
        func_call_list=queries,
        initial_config=initial_config,
        involved_classes=classes,
    )

    return execution_results, involved_instances

def compare_instances(model_obect, ground_truth_object):
    """
    Checks if the model_object has the same attributes as the ground_truth_object. They are instances of the same class.
    """
    assert type(model_obect) == type(
        ground_truth_object
    ), "Objects are not of the same type."
    differences = {}
    valid = True
    
    for attr_name in vars(ground_truth_object):
        # We don't check for private attributes
        if attr_name.startswith("_"):
            continue
        model_attr = getattr(model_obect, attr_name)
        ground_truth_attr = getattr(ground_truth_object, attr_name)

        if model_attr != ground_truth_attr:
            valid = False
            differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}

    return valid, differences


def get_state_score(solution_state, ground_truth_state):
    num_state_matches = 0
    num_state_total = 0
    for key in ground_truth_state:
        valid, diff = compare_instances(solution_state[key], ground_truth_state[key])
        num_state_matches += int(valid)
        num_state_total += 1
        
    state_score = (num_state_matches / num_state_total)
    return state_score


def compute_score(solution_str, ground_truth, initial_config):
    """ The scoring function for math problems.
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """

    tool_calls = extract_tool_calls(solution_str)[2:]
    tool_calls_parsed = parse_list_of_tool_calls(tool_calls)
    ground_truth_parsed = ast.literal_eval(ground_truth)
    classes = extract_classes(solution_str)

    gt_calls = ground_truth_parsed[0]
    ground_truth_formatted = [f"[{call}]" for call in gt_calls]

    exec_sol, involved_instances_solution = execute_list(tool_calls, classes, deepcopy(initial_config))
    exec_gt, involved_instances_gt = execute_list(ground_truth_formatted, classes, deepcopy(initial_config))

    # print("tool_calls", tool_calls)
    # print("exec_sol", exec_sol)
    # print("ground_truth_formatted", ground_truth_formatted)
    # print("exec_gt", exec_gt)
    
    state_score = get_state_score(involved_instances_solution, involved_instances_gt)
    func_match = response_checker(tool_calls_parsed, ground_truth_parsed[0])
    func_score = 1 if func_match['valid'] else 0
    format_reward = format_reward_score(solution_str)
    
    
    answer_score = (func_score  + state_score) / 2
    format_score = format_reward * 0.2    

    # print(f"state_score {state_score}, func_score {func_score}, format_reward {format_reward}, answer_score {answer_score}")
    
    return {
        'answer': answer_score,
        'format': 0, # no format score
        'tool': 0,
        "reason": "test"}

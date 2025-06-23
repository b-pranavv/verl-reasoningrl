import importlib
import inspect
import json
import re
import copy
import string
import random


def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def _process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    """
    Prepends the instance name to the function name for each of the function name represented in the string, you will
    also be provided with the mapping of method name to instance name.

    Example input:
    ```
    f(x = g((1, 2), h(3)), y = (4), z = (5, 6))
    ```

    Example return:
    ```
    a.f(x=a.g((1, 2), a.h(3)), y=(4), z=(5, 6))
    ```

    Args:
        function_call_string (str): The function call string to parse.
        class_mapping (dict): A dictionary mapping method names to instance names.

    Returns:
        str: The parsed function call string with instance names prepended to method names.
    """

    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    # Regular expression to match function names
    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"

    # Replace function names with their class-prepended versions
    processed_string = re.sub(pattern, replace_function, function_call_string)

    return processed_string


class BfclEnv():
    def __init__(self, **kwargs):
        self.CLASS_FILE_PATH_MAPPING = {
            "GorillaFileSystem": "verl.workers.rollout.vllm_rollout.envs.bfcl_envs.gorilla_file_system",
            "MathAPI": "verl.workers.rollout.vllm_rollout.envs.bfcl_envs.math_api",
            "MessageAPI": "verl.workers.rollout.vllm_rollout.envs.bfcl_envs.message_api",
            "TwitterAPI": "verl.workers.rollout.vllm_rollout.envs.bfcl_envs.posting_api",
            "TicketAPI": "verl.workers.rollout.vllm_rollout.envs.bfcl_envs.ticket_api",
            "TradingBot": "verl.workers.rollout.vllm_rollout.envs.bfcl_envs.trading_bot",
            "TravelAPI": "verl.workers.rollout.vllm_rollout.envs.bfcl_envs.travel_booking",
            "VehicleControlAPI": "verl.workers.rollout.vllm_rollout.envs.bfcl_envs.vehicle_control",
        }
        self.STATELESS_CLASSES = ["MathAPI"]
        self.env_instances = {}  # Add this line to store instances
    
    def execute_multi_turn_func_call(
        self,
        func_call_list: list[str],  # a list of strings of func calls
        initial_config: str,
        involved_classes: list,
        long_context: bool = False,
    ) -> tuple[list[str], dict]:
        
        random_id = ''.join(random.choices(string.ascii_lowercase, k=10))
        initial_config = json.loads(initial_config)
        class_method_name_mapping = {}
        involved_instances = {}
        for class_name in involved_classes:
            module_name = self.CLASS_FILE_PATH_MAPPING[class_name]

            instance_name = (
                f"{random_id}_{class_name.lower()}_instance"
            )
            if instance_name not in globals():
                module = importlib.import_module(module_name)
                class_ = getattr(module, class_name)
                class_instance = class_()
                if class_name not in self.STATELESS_CLASSES:
                    class_initial_config = initial_config.get(class_name, {})
                    # Deep copy the initial configuration to avoid mutation issues
                    class_instance._load_scenario(
                        copy.deepcopy(class_initial_config), long_context=long_context
                    )
                globals()[instance_name] = class_instance
            # This happens in subsequent turns
            else:
                class_instance = globals()[instance_name]

            involved_instances[class_name] = class_instance

            # Retrieve all method names and map them to the instance
            for method_name, method in inspect.getmembers(
                class_instance, predicate=inspect.ismethod
            ):
                # Skip private methods
                if method_name.startswith("_"):
                    continue
                class_method_name_mapping[method_name] = instance_name

        execution_results = []
        for func_call in func_call_list:
            # Add the instance name to the method calls
            func_call = _process_method_calls(func_call, class_method_name_mapping)

            # Evaluate the function call
            try:
                # We need to make a copy here because otherwise the `eval(func_call)` would error. 
                func_call_copy = func_call
                # Before calling `eval`, we need to make sure that the function call is safe
                # We do so by checking if the function is `kill` or `exit`, etc.
                # Extract the function name first
                if "(" in func_call_copy:
                    func_call_copy = func_call_copy.split("(")[0]
                # Situation where the function call is a method call
                if "." in func_call_copy:
                    func_call_copy = func_call_copy.split(".")[1]
                if func_call_copy in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                    raise Exception(f"Function call {func_call_copy} is not allowed.")

                func_call_result = eval(func_call)

                if type(func_call_result) == str:
                    pass
                elif type(func_call_result) == dict:
                    # Some function returns a object instance, which is not serializable
                    try:
                        func_call_result = json.dumps(func_call_result)
                    except:
                        func_call_result = str(func_call_result)
                else:
                    func_call_result = str(func_call_result)

                execution_results.append(func_call_result)
            except Exception as e:
                execution_results.append(f"Error during execution: {str(e)}")
            
        # Clean
        for class_name in involved_classes:
            instance_name = f"{random_id}_{class_name.lower()}_instance"
            if instance_name in globals():
                del globals()[instance_name]

        return execution_results, involved_instances

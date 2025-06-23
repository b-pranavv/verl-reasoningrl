import io
import os
import signal
import subprocess
import tempfile
import traceback
from contextlib import redirect_stdout
from typing import Any, List, Tuple
from tqdm import tqdm
from verl.workers.rollout.vllm_rollout.bfclEnv.bfcl_env import BfclEnv
import resource  # For memory limiting (Unix only)


class TimeoutError(Exception):
    """Exception raised when code execution times out."""
    pass

class BfclExecutor:
    def __init__(self, timeout_seconds=5, capture_stdout=True):
        self.timeout_seconds = timeout_seconds
        self.capture_stdout = capture_stdout
    
    def execute_list(self, queries, classes, initial_config):
        bfcl_env = BfclEnv()
        execution_results, involved_instances = bfcl_env.execute_multi_turn_func_call(
            func_call_list=queries,
            initial_config=initial_config,
            involved_classes=classes,
        )

        return execution_results, involved_instances

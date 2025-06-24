
import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
from math_verify import parse, LatexExtractionConfig, ExprExtractionConfig, verify
import json
from copy import deepcopy
from huanzhi_utils import load_file
from datasets import Dataset, load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

def preprocess_python_dataset():
    
    data_source = datasets.load_dataset("AI-MO/NuminaMath-CoT")
    train_dataset = data_source['train']
    
    train_dataset =train_dataset.train_test_split(test_size=0.001, seed=2)
    
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
        
        def process_fn(example, idx):
            example['question'] = example['problem'].strip()
            data = {
                "data_source": 'python',
                "question": example['question'],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example['solution'],
                    "initial_config": None,
                },
            "extra_info": {
                'split': split,
                'index': str(idx),
                'id': str(idx),
                'involved_classes': None,
                "num_turns": None,
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


    test_dataset = test_dataset.shuffle(seed=42).select(range(50))
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    return train_dataset, test_dataset

def preprocess_bfcl_dataset():
    # Get path of the directory of this file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    multi_turn_base_data = load_file(f"{current_directory}/berkeley-function-call-leaderboard/data/BFCL_v3_multi_turn_base.json")
    multi_turn_base_answer = load_file(f"{current_directory}/berkeley-function-call-leaderboard/data/possible_answer/BFCL_v3_multi_turn_base.json")

    # Reprocess the columns into serializable format and add num_turns
    for i in range(len(multi_turn_base_data)):
        question_data = multi_turn_base_data[i]["question"]
        ground_truth = multi_turn_base_answer[i]["ground_truth"]
        initial_config = multi_turn_base_data[i]["initial_config"]
        
        # Assert number of turns matches between question and ground truth
        assert len(question_data) == len(ground_truth), f"Mismatch in number of turns for entry {i}"
        
        multi_turn_base_data[i]["num_turns"] = len(question_data)
        multi_turn_base_data[i]["question"] = json.dumps(question_data)
        multi_turn_base_data[i]["initial_config"] = json.dumps(initial_config)
        multi_turn_base_data[i]["answer"] = json.dumps(ground_truth)

    dataset = Dataset.from_list(multi_turn_base_data)
    dataset = dataset.map(lambda x: {
            "data_source": "bfcl",
            "question": x["question"],
            "ability": "function-calling",
            "reward_model": {
                "style": "rule",
                "ground_truth": deepcopy(x["answer"]),
                "initial_config": x["initial_config"],
            },
            "extra_info": {
                "id": x["id"],
                "index": x["id"],
                "involved_classes": x["involved_classes"],
                "num_turns": x["num_turns"],
            }, 
        })

    for i in range(len(dataset)):
        ground_truth_bank = json.loads(dataset[i]["reward_model"]["ground_truth"])
        user_question_bank = json.loads(dataset[i]["question"])
        assert len(ground_truth_bank) == len(user_question_bank), f"Length mismatch at index {i}: ground_truth_bank ({len(ground_truth_bank)}) != user_question_bank ({len(user_question_bank)})"

    # Get unique IDs and split those first
    unique_ids = sorted(list(set([info["id"] for info in dataset["extra_info"]])))
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.5, random_state=42)
    print(f"Train IDs: {len(train_ids)}")
    print(f"Test IDs: {len(test_ids)}")

    # Filter dataset based on IDs
    train_dataset = dataset.filter(lambda x: x["extra_info"]["id"] in train_ids)
    test_dataset = dataset.filter(lambda x: x["extra_info"]["id"] in test_ids)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # assert train_dataset and test_dataset have non-overlapping ids
    assert len(set([info["id"] for info in train_dataset["extra_info"]]) & set([info["id"] for info in test_dataset["extra_info"]])) == 0, "Train and test datasets have overlapping ids"

    return train_dataset, test_dataset

def combine_datasets():
    # Preprocess both datasets
    python_train_dataset, python_test_dataset = preprocess_python_dataset()
    bfcl_train_dataset, bfcl_test_dataset = preprocess_bfcl_dataset()
    
    # Combine the two datasets
    min_len_train = max(min(len(python_train_dataset), len(bfcl_train_dataset)), 200)  # Ensure train set has at least 200 samples
    min_len_test = max(min(len(python_test_dataset), len(bfcl_test_dataset)), 50)  # Ensure test set has at least 50 samples
    
    train_dataset = datasets.concatenate_datasets([python_train_dataset.select(range(min_len_train)), bfcl_train_dataset])
    test_dataset = datasets.concatenate_datasets([python_test_dataset.select(range(min_len_test)), bfcl_test_dataset.select(range(min_len_test))])
    
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./tool_dataset')


    args = parser.parse_args()


    train_dataset, test_dataset = combine_datasets()
    makedirs(args.local_dir, exist_ok=True)
    
    # Save the datasets to parquet files
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
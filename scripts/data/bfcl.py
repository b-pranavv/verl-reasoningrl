"""
Preprocess the bfcl dataset to parquet format
"""

import re
import os
import json
from copy import deepcopy
from huanzhi_utils import load_file
from datasets import Dataset, load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import argparse


def preprocess_bfcl_dataset() -> Dataset:
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

    return dataset_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./bfcl_dataset')

    args = parser.parse_args()

    dataset_dict = preprocess_bfcl_dataset()
    train_dataset = dataset_dict['train']
    test_dataset = dataset_dict['test']
    
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))

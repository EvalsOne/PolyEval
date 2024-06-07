import pytest
import os, sys, json
from datasets import Dataset
from zeval.evaluation import evaluate
from zeval.evaluators import JsonMatch
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 创建一个Dataset实例
dataset = Dataset.from_dict({
    'question': ["Is this a valid JSON?", "What is the user info?", "Is this nested JSON valid?", "Does this list match?"],
    'answer': [
        '{"name": "Tom","city": "Paris", "age": 18}', 
        '{"user": {"name": "Alice", "sex": "Female", "city": "London"}}',
        '{"outer": {"inner": {"key": "my value"}}}',
        '[{"item1": "3"}, {"item2": "3"}, {"item3": "2"}]'
    ],
    'ideal': [
        ['{"name": "Tom", "age": 18, "city": "Paris"}'],
        ['{"user": {"name": "Alice", "age": 25, "city": "Wonderland"}}'],
        ['{"outer": {"inner": {"key": "value"}}}'],
        ['[{"item1": "value1"}, {"item2": "value2"}]']
    ]
})

rules = {}

@pytest.mark.parametrize("evaluators, lang, rules", [
    ([JsonMatch], "zh", {})
])
def test_json(evaluators, lang, rules):
    eval_results = evaluate(dataset, evaluators, rules, lang)
    # print eval results
    formatted_results = json.dumps(eval_results, indent=4, ensure_ascii=False)
    print(formatted_results)
    assert isinstance(eval_results, list)
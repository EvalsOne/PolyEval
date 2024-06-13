import pytest
import os, sys, json
from datasets import Dataset
from polyeval.evaluation import evaluate
from polyeval.evaluators import Matchness
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dataset = Dataset.from_dict({
    'question': ["What's the capital of France?", "Who is the first president of USA?"],
    'answer': ["Paris is the capital of France.", "washington."], 
    'ideal': [["paris", "Paris"], ["George Washington", "Washington"]]
})

# match_rules = ['match', 'include', 'startswith', 'endswith', 'fuzzy_match']

@pytest.mark.parametrize("evaluators, lang, kwargs", [
    ([Matchness], "zh", {"match_rule": "match", "ignore_case": True}),
    ([Matchness], "zh", {"match_rule": "match", "ignore_case": False}),
    ([Matchness], "zh", {"match_rule": "include", "ignore_case": True}),
    ([Matchness], "zh", {"match_rule": "include", "ignore_case": False}),
    ([Matchness], "zh", {"match_rule": "startswith", "ignore_case": True}),
    ([Matchness], "zh", {"match_rule": "startswith", "ignore_case": False}),
    ([Matchness], "zh", {"match_rule": "endswith", "ignore_case": True}),
    ([Matchness], "zh", {"match_rule": "endswith", "ignore_case": False}),
    ([Matchness], "zh", {"match_rule": "fuzzy_match", "ignore_case": True}),
    ([Matchness], "zh", {"match_rule": "fuzzy_match", "ignore_case": False}),
])
def test_match(evaluators, lang, kwargs):
    eval_results = evaluate(dataset, evaluators, lang, **kwargs)
    # print eval results
    formatted_results = json.dumps(eval_results, indent=4, ensure_ascii=False)
    print(formatted_results)
    assert isinstance(eval_results, list)
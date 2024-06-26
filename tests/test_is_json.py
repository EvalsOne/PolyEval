import pytest
import os, sys, json
from datasets import Dataset
from polyeval.evaluation import evaluate
from polyeval.evaluators import IsJson
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dataset = Dataset.from_dict({
    'question': ["is it a valid json?","is it a valid json?"],
    'answer': ['"name": "John", "age": 30, "city": "New York"}','{"name": "John", "age": 30, "city": "New York"}']
})

kwargs = {}

@pytest.mark.parametrize("evaluators, lang, kwargs", [
    ([IsJson], "zh", kwargs)
])
def test_json(evaluators, lang, kwargs):
    eval_results = evaluate(dataset, evaluators, lang, **kwargs)
    # print eval results
    formatted_results = json.dumps(eval_results, indent=4, ensure_ascii=False)
    print(formatted_results)
    assert isinstance(eval_results, list)
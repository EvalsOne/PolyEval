import pytest
import os, sys, json
from datasets import Dataset
from polyeval.evaluation import evaluate
from polyeval.evaluators import Regex
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dataset = Dataset.from_dict({
    'question': ["this is the email", "this is the email"],
    'answer': ["zhangyue@gmail.com", "asdfasfd@ddddd"], 
})

regex_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

@pytest.mark.parametrize("evaluators, lang, kwargs", [
    ([Regex], "zh", {"pattern": regex_pattern}),
])
def test_regex(evaluators, lang, kwargs):
    eval_results = evaluate(dataset, evaluators, lang, **kwargs)
    # print eval results
    formatted_results = json.dumps(eval_results, indent=4, ensure_ascii=False)
    print(formatted_results)
    assert isinstance(eval_results, list)
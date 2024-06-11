import pytest
import os, sys, json
from datasets import Dataset
from polyeval.evaluation import evaluate
from polyeval.evaluators import StringDistanceEvaluator
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dataset = Dataset.from_dict({
    'answer': ["The job is completely done.", "The job is done."],
    'ideal': ["The job is done", "The job isn't done"], 
})

@pytest.mark.parametrize("evaluators, lang, kwargs", [
    ([StringDistanceEvaluator], "zh", {"metric": "damerau_levenshtein"}),
    ([StringDistanceEvaluator], "en", {"metric": "levenshtein"}),
    ([StringDistanceEvaluator], "zh", {"metric": "jaro"}),
    ([StringDistanceEvaluator], "zh", {"metric": "jaro_winkler"}),
    ([StringDistanceEvaluator], "zh", {"metric": "hamming"}),
])
def test_string_distance(evaluators, lang, kwargs):
    eval_results = evaluate(dataset, evaluators, lang, **kwargs)
    # print eval results
    formatted_results = json.dumps(eval_results, indent=4, ensure_ascii=False)
    print(formatted_results)
    assert isinstance(eval_results, list)
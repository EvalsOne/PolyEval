import pytest
import os, sys, json
from datasets import Dataset
from polyeval.evaluation import evaluate
from polyeval.evaluators import CustomYaml
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

fact_spec = """prompt: |-
    You are comparing a submitted answer to an expert answer on a given question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {input}
    ************
    [Expert]: {ideal}
    ************
    [Submission]: {completion}
    ************
    [END DATA]

    Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
    The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
    (A) The submitted answer is a subset of the expert answer and is fully consistent with it.
    (B) The submitted answer is a superset of the expert answer and is fully consistent with it.
    (C) The submitted answer contains all the same details as the expert answer.
    (D) There is a disagreement between the submitted answer and the expert answer.
    (E) The answers differ, but these differences don't matter from the perspective of factuality.
eval_type: cot_classify
choice_strings: ABCDE
"""

closedqa_spec = """prompt: |-
    You are assessing a submitted answer on a given task based on a criterion. Here is the data:
    [BEGIN DATA]
    ***
    [Task]: {input}
    ***
    [Submission]: {completion}
    ***
    [Criterion]: {criteria}
    ***
    [END DATA]
    Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the letter again by itself on a new line.

    Reasoning:
eval_type: cot_classify
choice_scores:
    "Y": 1.0
    "N": 0.0
choice_strings: 'YN'
threshold: 1
"""

zhipu_params = {
    "provider": "zhipuai",
    "model": "glm-4",
    "temperature": 0.1,
    "stream": False
}

dataset = Dataset.from_dict({
    'question': ["What's the capital of France?", "Who is the first president of USA?"],
    'answer': ["Paris is the capital of France.", "washington."], 
    'ideal': [["paris", "Paris"], ["George Washington", "Washington"]]
})

kwargs = {}
kwargs["llm"] = zhipu_params
kwargs["yaml_spec"] = fact_spec

@pytest.mark.parametrize("evaluators, lang, kwargs", [
    ([CustomYaml], "en", kwargs),
])
def test_yaml(evaluators, lang, kwargs):
    eval_results = evaluate(dataset, evaluators, lang, **kwargs)
    # print eval results
    assert isinstance(eval_results, list)
from datasets import Dataset
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Optional, Union, Dict, List

def evaluate(dataset: Dataset, evaluators: list, lang='en', **kwargs):
    all_results = []
    for evaluator in evaluators:
        dimension_results = []
        eval_instance = evaluator(lang=lang)
        for i in range(len(dataset)):
            print("kwargs in:", kwargs)
            result = eval_instance.eval(dataset[i],**kwargs)
            dimension_results.append(result)
        all_results.append(dimension_results)
    return all_results

@dataclass
class Result:
    score: float
    reasoning: str
    responses: list[dict]
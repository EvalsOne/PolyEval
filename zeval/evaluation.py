from datasets import Dataset
from dataclasses import asdict, dataclass

def evaluate(dataset: Dataset, evaluators: list, sample_kwargs: dict, lang='zh'):
    all_results = []
    for evaluator in evaluators:
        dimension_results = []
        eval_instance = evaluator(lang=lang)
        for i in range(len(dataset)):
            result = eval_instance.eval(dataset[i], sample_kwargs)
            dimension_results.append(result)
        all_results.append(dimension_results)
    return all_results

@dataclass
class Result:
    score: float
    reasoning: str
    responses: list[dict]
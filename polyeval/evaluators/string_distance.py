from typing import Any, Callable, Dict, List
from langchain.evaluation import load_evaluator, StringDistance
from datasets import Dataset
import logging
logging.basicConfig(level=logging.INFO)

class StringDistanceEvaluator:
    name = 'string_distance'

    def __init__(self, lang='zh'):
        self.language = lang

    def eval(self, dataset: Dataset, **kwargs):
        if dataset is None:
            return False, "No dataset provided"

        question = dataset.get("question", None)
        sampled = dataset.get("answer", None)
        ideal = dataset.get("ideal", None)
        
        metric = kwargs.get('metric', None)
        if not metric:
            return False, "String distance metric not provided"
        
        """
        measure the distance between the question and the ideal answers
        """
        evaluator = load_evaluator(
            "string_distance", distance=metric
        )
            
        try:
            if isinstance(ideal, str):
                ideal = [ideal]
            if not isinstance(ideal, list):
                return False, "Ideal answer is not a list"
            
            list_of_scores = []
            for i in range(len(ideal)):
                expected = ideal[i]
                result = evaluator.evaluate_strings(
                    prediction=sampled,
                    reference=expected,
                )
                list_of_scores.append(result['score'])
            avg_score = sum(list_of_scores) / len(list_of_scores)
            min_score = min(list_of_scores)
            min_key = list_of_scores.index(min_score)
            min_ideal = ideal[min_key]
            max_score = max(list_of_scores)
            max_key = list_of_scores.index(max_score)
            max_ideal = ideal[max_key]
            
            min_item = {}
            min_item['score'] = min_score
            min_item['ideal'] = min_ideal
            max_item = {}
            max_item['score'] = max_score
            max_item['ideal'] = max_ideal
            
            eval_result = {
                'metric': metric,
                'ideal': ideal,
                'sampled': sampled,
                "score": avg_score,
                "min": min_item,
                "max": max_item
            }
            return eval_result
        except Exception as e:
            return False, str(e)
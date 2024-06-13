from typing import Any, Callable, Dict, List
from langchain.evaluation import load_evaluator, EmbeddingDistance
from langchain_community.embeddings import OpenAIEmbeddings, CohereEmbeddings, BaichuanTextEmbeddings
from datasets import Dataset
import logging
import re
logging.basicConfig(level=logging.INFO)

class EmbeddingDistanceEvaluator:
    name = 'embedding_distance'

    def __init__(self, lang='zh'):
        self.language = lang

    def eval(self, dataset: Dataset, **kwargs):
        if dataset is None:
            return False, "No dataset provided"

        embedding_kwargs = kwargs.get('embedding', None)
        if not embedding_kwargs:
            return False, "No embedding parameters provided"

        metric = kwargs.get('metric', None)
        if not metric:
            return False, "Embedding distance metric not provided"

        question = dataset.get("question", None)
        sampled = dataset.get("answer", None)
        ideal = dataset.get("ideal", None)
        
        if 'provider_cls' not in embedding_kwargs:
            return False, "Embedding model provider not provided"
        provider_cls = embedding_kwargs['provider_cls']
        provider_cls = globals()[provider_cls]
        embedding_kwargs.pop('provider_cls')
        
        embedingModel = provider_cls(**embedding_kwargs)
        
        """
        measure the embedding distance between the question and the ideal answers
        """
        
        print("metric in:", metric)
        evaluator = load_evaluator(
            "embedding_distance", distance_metric=metric, embedding_model = embedingModel
        )
        
        print(evaluator)
            
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
            print(result)
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

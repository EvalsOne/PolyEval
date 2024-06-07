from zeval.utils.helpers import is_valid_json, json_match
from typing import Any, Callable, Dict, List
from jsonschema import validate, ValidationError
from datasets import Dataset
import logging
import json
logging.basicConfig(level=logging.INFO)

class IsJson:
    name = 'is_json'

    def __init__(self, lang='zh'):
        self.language = lang

    def eval(self, dataset: Dataset, sample_kwargs: Dict[str, Any] = None):
        if dataset is None:
            params = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        question = dataset["question"]
        sampled = dataset["answer"]
        
        """
        judge if the answer is a valid json
        """
        
        pass_eval = is_valid_json(sampled)
        score = 1 if pass_eval else 0
            
        try:
            eval_result = {
                'question': question,
                'sampled': sampled,
                "score": score,
                "pass_eval": pass_eval
            }
            return eval_result
        except Exception as e:
            return False, str(e)
        
class JsonMatch:
    name = 'json_match'

    def __init__(self, lang='zh'):
        self.language = lang

    def eval(self, dataset: Dataset, sample_kwargs: Dict[str, Any] = None):
        if dataset is None:
            params = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        question = dataset["question"]
        sampled = dataset["answer"]
        correct_answers = dataset["ideal"]
        
        """
        judge if the 
        """
        sampled_json: Any
        try:
            sampled_json = json.loads(sampled)
        except ValueError:
            # If the sampled string is not valid JSON, it will never match
            sampled_json = None

        # Allow the following to raise ValueError; the correct answers
        # should always be valid JSON
        correct_json = [json.loads(correct_answer) for correct_answer in correct_answers]

        matches = [json_match(sampled_json, cj) for cj in correct_json]
        picked=[sampled for i in range(len(correct_answers)) if matches[i]],
        score = 1 if True in matches else 0
        pass_eval = True in matches
            
        try:
            eval_result = {
                'question': question,
                'sampled': sampled,
                'ideal': correct_answers,
                'picked': picked,
                "score": score,
                "pass_eval": pass_eval
            }
            return eval_result
        except Exception as e:
            return False, str(e)
        
class JsonSchemaMatch:
    name = 'json_schema_match'

    def __init__(self, lang='zh'):
        self.language = lang

    def eval(self, dataset: Dataset, sample_kwargs: Dict[str, Any] = None):
        if dataset is None:
            return False, "Dataset is None"
        if sample_kwargs is None:
            sample_kwargs = {}

        results = []
        if 'schema' not in sample_kwargs:
            return False, "Schema not provided"
        schema = sample_kwargs['schema']
        question = dataset["question"]
        sampled = dataset["answer"]
        

        try:
            sampled_json = json.loads(sampled)
        except ValueError:
            sampled_json = None
        is_valid = False
        error_msg = ""

        if sampled_json is not None:
            try:
                validate(instance=sampled_json, schema=schema)
                is_valid = True
            except ValidationError as e:
                error_msg = str(e)

        eval_result = {
            'question': question,
            'score': 1 if is_valid else 0,
            'pass_eval': is_valid,
            'sampled': sampled,
            'is_valid': is_valid,
            'reasoning': error_msg
        }
        return eval_result
from typing import Any, Callable, Dict, List
from jsonschema import validate, ValidationError
from datasets import Dataset
import logging
import re
logging.basicConfig(level=logging.INFO)

class Regex:
    name = 'regex'

    def __init__(self, lang='zh'):
        self.language = lang

    def eval(self, dataset: Dataset, **kwargs):
        if dataset is None:
            return False, "No dataset provided"

        question = dataset["question"]
        sampled = dataset["answer"]
        
        pattern = kwargs.get('pattern', None)
        if not pattern:
            return False, "Regex pattern not provided"
        pattern = kwargs['pattern']
        
        """
        judge if the answer pass the regex
        """
        
        if re.match(pattern, sampled):
            score = 1
            pass_eval = True
        else:
            score = 0
            pass_eval = False
            
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
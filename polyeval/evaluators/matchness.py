from polyeval.utils.model import ModelCaller
from polyeval.utils.helpers import fuzzy_match, get_answer, normalize, get_consensus
from typing import Any, Callable, Dict
from datasets import Dataset
import logging
import json
logging.basicConfig(level=logging.INFO)

class Matchness:
    name = 'matchness'

    def __init__(self, lang='zh'):
        self.language = lang

    def eval(self, dataset: Dataset, **kwargs):
        if dataset is None:
            return False, "No dataset provided"

        match_rule = kwargs.get('match_rule', None)
        ignore_case = kwargs.get('ignore_case', None)
        if not match_rule:
            return False, "No enough params provided"

        question = dataset["question"]
        sampled = dataset["answer"]
        expected = dataset["ideal"]
        
        """
        Compute matchness score for a given question and answer pair.
        """
        
        supported_match_rules = ['match', 'include', 'startswith', 'endswith', 'fuzzy_match']
        
        if match_rule not in supported_match_rules:
            return False, f"Unknown match rule: {match_rule}"
        
        if match_rule == 'match':
            if ignore_case:
                match = sampled.lower() in [ref.lower() for ref in expected]
                score = 1 if match else 0
                pass_eval = match
            else:
                match = sampled in expected
                score = 1 if match else 0
                pass_eval = match         
            if not pass_eval:
                picked = None
            else:
                picked = sampled       
        elif match_rule == 'include':
            includes_answer = any(
                [get_answer(sampled, ref, ignore_case)
                is not None for ref in expected]
            )
            pass_eval = includes_answer
            score = 1 if pass_eval else 0
            picked = [expected[i] for i in range(len(expected)) if includes_answer]
        elif match_rule == 'fuzzy_match':
            if ignore_case:
                sampled = sampled.lower()
                expected = [ref.lower() for ref in expected]
            matches = [
                fuzzy_match(sampled, answer) for answer in expected
            ]            
            match = True in matches
            score = 1 if match else 0
            picked = [expected[i] for i in range(len(expected)) if matches[i]]
            pass_eval = match
        elif match_rule == 'startswith':
            picked = None
            if ignore_case:
                sampled = sampled.lower()
                expected = [ref.lower() for ref in expected]
            if expected:
                for ref in expected:
                    if not sampled.startswith(ref):
                        continue
                    picked = ref
                    break
            match = True if picked else False
            score = 1 if match else 0
            pass_eval = match
        elif match_rule == 'endswith':
            picked = None
            if ignore_case:
                sampled = sampled.lower()
                expected = [ref.lower() for ref in expected]
            if expected:
                for ref in expected:
                    if not sampled.endswith(ref):
                        continue
                    picked = ref
                    break
            match = True if picked else False
            score = 1 if match else 0
            pass_eval = match
            
        try:
            eval_result = {
                'question': question,
                'sampled': sampled,
                'expected': expected,
                'rule': match_rule,
                'ignore_case': ignore_case,
                "score": score,
                "picked": picked,
                "pass_eval": pass_eval
            }
            return eval_result
        except Exception as e:
            return False, str(e)
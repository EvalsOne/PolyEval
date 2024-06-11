from collections import Counter, defaultdict
import copy
import re, math, json
import string
from typing import Any, Dict, List, Mapping, Union, cast

def get_answer(text, answer_prompt, ignore_case=False):
    if ignore_case:
        idx = text.lower().rfind(answer_prompt.lower())
    else:
        idx = text.rfind(answer_prompt)

    if idx == -1:
        return None
    return text[idx:idx + len(answer_prompt)]


def get_consensus(answers):
    counts = defaultdict(int)
    for answer in answers:
        counts[answer] += 1
    counts[None] = 0
    return max(counts, key=counts.get)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    # s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except ValueError:
        return False
    
def json_match(sampled_json: Any, correct_json: Any) -> bool:
    """Return True if the sampled completion in JSON format
    matches a correct answer, component by component"""
    if sampled_json is None or correct_json is None:
        # Missing values are never correct
        return False
    if isinstance(sampled_json, dict):
        if isinstance(correct_json, dict):
            sample = cast(Mapping[str, Any], sampled_json)
            correct = cast(Mapping[str, Any], correct_json)
            all_keys = set(sample.keys()) | set(correct.keys())
            return all(json_match(sample.get(key), correct.get(key)) for key in all_keys)
        else:
            return False
    elif isinstance(sampled_json, list):
        if isinstance(correct_json, list):
            slist = cast(List[Any], sampled_json)
            clist = cast(List[Any], correct_json)
            if len(slist) != len(clist):
                # Lists must have the same length
                return False
            return all(json_match(s, c) for s, c in zip(slist, clist))
        else:
            return False
    # Not a structured item: do a direct comparison
    return sampled_json == correct_json

import logging
import string
from typing import Any, Callable, Iterable, Optional, Union, Dict, List
from polyeval.utils.model import ModelCaller
import re, yaml

import dataclasses
from dataclasses import dataclass

# This is an approximation to the type accepted as the `prompt` field to `openai.Completion.create` calls
OpenAICreatePrompt = Union[str, list[str], list[int], list[list[int]]]

# This is the type accepted as the `prompt` field to `openai.ChatCompletion.create` calls
OpenAIChatMessage = Dict[str, str]  # A message is a dictionary with "role" and "content" keys
OpenAICreateChatPrompt = List[OpenAIChatMessage]  # A chat log is a list of messages

@dataclass
class ModelGradedSpec:
    # must have
    prompt: Union[str, OpenAICreateChatPrompt]
    choice_strings: Union[list[str], str] = None

    # optional
    input_outputs: Optional[dict[str, str]] = None
    eval_type: Optional[str] = None
    choice_scores: Optional[Union[dict[str, float], str]] = None
    output_template: Optional[str] = None
    threshold: Optional[float] = None
    reverse_score: Optional[bool] = None

INVALID_STR = "__invalid__"

ANSWER_PROMPTS = {
    'en': {
        "classify": "Answer the question by printing only a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer with no other text.".strip(),
        "classify_cot": "First, answer by printing a single choice from {choices} (without quotes or punctuation) corresponding to the correct answer. Then, from the next line, explain your reasonings step by step.".strip(),
        "cot_classify": """
First, write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Then print only a single choice from {choices} (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the answer by itself on a new line.

Reasoning:""".strip(),
    },
    'zh': {
        "classify": "通过输出与正确答案对应的单个选择（无引号或标点符号）来回答问题。".strip(),
        "classify_cot": "首先，通过输出与正确答案对应的单个选择（无引号或标点符号）来回答问题。然后，从下一行开始，逐步解释您的推理。".strip(),
        "cot_classify": """
首先，逐步写出您的推理，以确保您的结论是正确的。避免一开始就简单地陈述正确答案。然后，在新的一行输出与正确答案对应的单个选择（无引号或标点符号）。

推理:""".strip(),
    }
}

MATCH_FNS = {
    "include": lambda x, y: float(x in y),
    "exact": lambda x, y: float(x == y),
    "endswith": lambda x, y: x.endswith(y),
    "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
}

def get_choice_strings(choice_strings: Union[list[str], str], n: Optional[int] = None):
    # 'choice_strings' is a list of strings that specifies the possible choices
    if choice_strings == "from_n":
        choice_strings = [str(i + 1) for i in range(n)]
    elif choice_strings == "from_n_abc":
        choice_strings = [string.ascii_lowercase[i % 26] for i in range(n)]
    elif choice_strings == "from_n_ABC":
        choice_strings = [string.ascii_uppercase[i % 26] for i in range(n)]
    # make sure each choice doesn't contain any punctuation
    for s in choice_strings:
        assert not any(c in s for c in string.punctuation), f"{s} contains punctuation"
    return choice_strings

def get_choice_score(
    choice: str,
    choice_strings: Iterable[str],
    choice_scores: Optional[Union[dict[str, float], str]] = None,
) -> Optional[float]:
    if choice_scores is None:
        return None
    if choice_scores == "from_strings":
        choice_scores = {c: float(c) for c in choice_strings}
    # assumption: each INVALID_STR contributes the lowest score
    if choice == INVALID_STR:
        return min(choice_scores.values())
    return choice_scores[choice]


def choice_to_str(choice_strings: Iterable[str]) -> str:
    """Return a string of choices, e.g. '"Yes" or "No" or "Maybe"'."""
    return " or ".join(f'"{choice}"' for choice in choice_strings)


def get_choice(
    text: str, eval_type: str, match_fn: Union[str, Callable], choice_strings: Iterable[str]
) -> str:
    
    print("text", text)
    print("eval_type", eval_type)
    print("match_fn", match_fn)
    """Clean the answer string to a choice string to one of choice_strings. Return '__invalid__.' if no match."""
    if isinstance(match_fn, str):
        match_fn = MATCH_FNS[match_fn]
    lines = text.strip().split("\n")
    
    if eval_type:
        if eval_type.startswith("cot_classify"):
            lines = lines[::-1]  # reverse lines
        for line in lines:
            line = line.strip()
            line = "".join(c for c in line if c not in string.punctuation)
            if not line:
                continue
            for choice in choice_strings:
                if match_fn(line, choice):
                    return choice
    logging.warn(f"Choices {choice_strings} not parsable for {eval_type}: {text}")
    return INVALID_STR

def append_answer_prompt(
    lang: str,
    prompt: OpenAICreateChatPrompt,
    eval_type: str,
    choice_strings: Optional[Iterable[str]] = None,
    answer_prompt: Optional[str] = None,
) -> OpenAICreateChatPrompt:
    """Append answer prompt to prompt."""

    if not eval_type:
        eval_type = "cot_classify"
    answer_prompt = answer_prompt or ANSWER_PROMPTS[lang][eval_type]
    
    print("applying answer_prompt", answer_prompt)
        
    if choice_strings is not None:        
        answer_prompt = answer_prompt.format(choices=choice_to_str(choice_strings))
        
    assert isinstance(answer_prompt, str), f"prompt must be str, not {type(answer_prompt)}"
    prompt += "\n\n" + answer_prompt
    return prompt

def get_score(text: str) -> float:
    """
    from the last line of the given text, parse a number as the evaluation score.
    """
    lines = text.strip().split("\n")
    if not lines:
        return False

    last_line = lines[-1].strip()
    match = re.search(r"[-+]?\d*\.?\d+", last_line)
    if match:
        score_str = match.group()
        try:
            score = float(score_str)
            return score
        except ValueError:
            pass

    return False

def _create_model_graded_spec_alt(data: dict) -> ModelGradedSpec:
    # 过滤出 ModelGradedSpec 中定义的属性
    valid_keys = {field.name for field in dataclasses.fields(ModelGradedSpec)}
    filtered_data = {k: v for k, v in data.items() if k in valid_keys}

    # 使用 **kwargs 动态创建 ModelGradedSpec 实例
    model_graded_spec = ModelGradedSpec(**filtered_data)
    return model_graded_spec

def get_yaml_spec_from_string(yaml_string: str) -> Optional[ModelGradedSpec]:
    """
    Parse a YAML string to create a ModelGradedSpec object.
    """
    try:
        registry = {}
        d = yaml.safe_load(yaml_string)
        if d is None or not isinstance(d, dict):
            # no entries in the file
            return

        for name, spec in d.items():
            if isinstance(spec, dict):
                if "key" in spec:
                    raise ValueError(
                        f"key is a reserved keyword, but was used in {name}"
                    )
                if "group" in spec:
                    raise ValueError(
                        f"group is a reserved keyword, but was used in {name}"
                    )
                if "cls" in spec:
                    raise ValueError(
                        f"cls is a reserved keyword, but was used in {name}"
                    )
                    
                if "class" in spec:
                    spec["cls"] = spec["class"]
                    del spec["class"]
            registry[name] = spec
        return _create_model_graded_spec_alt(registry)

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {e}")
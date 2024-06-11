from polyeval.utils.model import ModelCaller
from polyeval.utils.classify_helpers import ANSWER_PROMPTS, INVALID_STR, ModelGradedSpec, get_choice_strings, append_answer_prompt, get_choice, get_choice_score, get_score, get_yaml_spec_from_string
from typing import Any, Callable, Iterable, Optional, Union, Dict, List
from unionllm import UnionLLM, unionchat
from datasets import Dataset
import logging
import json, time
import inspect

logging.basicConfig(level=logging.INFO)

class CustomYaml:
    name = 'custom_yaml'

    def __init__(self, lang='en'):
        self.language = lang

    def eval(self, dataset: Dataset, **kwargs):
        if dataset is None:
            return False, "No dataset provided"
        sample_kwargs = kwargs.get('llm', None)
        if not sample_kwargs:
            return False, "No sampling parameters provided"

        question = dataset.get("question", None)
        sampled = dataset.get("answer", None)
        ideal = dataset.get("ideal", None)
        ideal_string = "\n".join(ideal) if isinstance(ideal, list) else ideal
        context = dataset.get("context", None)
        context_string = "\n".join(context) if isinstance(context, list) else context
        
        criteria = kwargs.get('criteron')

        """
        Parse yaml config and run modelgraded eval
        """
        
        yaml_spec = kwargs.get("yaml_spec")
        
        print("类型",type(yaml_spec))
        if isinstance(yaml_spec, str):
            self.yaml_config = get_yaml_spec_from_string(kwargs.get("yaml_spec"))
        elif yaml_spec.prompt:
            self.yaml_config = yaml_spec
        else:
            return False, "No yaml spec provided"            
        
        if self.yaml_config.prompt:
            self.yaml_config.prompt = self.yaml_config.prompt.format(completion=sampled, input=question, ideal=ideal_string, criteria=criteria, context=context_string)

        # run modelgraded eval
        classify_result = classify(
            self,
            mg_specs=self.yaml_config,
            sample_kwargs=sample_kwargs,
        )
        
        reasoning = classify_result['completion']
        # reasoning = ANSWER_PROMPTS[self.language]['reasoning'].format(answer=sampled, question=question, expected=ideal_string)
        responses = [classify_result['raw_response']]

        try:
            eval_result = {
                "question": question,
                "sampled": sampled,
                "expected": ideal,
                "grade": classify_result['choice'],
                "score": classify_result['score'],
                "pass_eval": classify_result['pass_eval'],
                "reasoning": reasoning,
                "responses": responses
            }
                        
            return eval_result
        except Exception as e:
            return False, str(e)
        
def classify(
    self,
    mg_specs: ModelGradedSpec,
    sample_kwargs: Optional[dict[str, Any]] = None,
    n: Optional[int] = None,
    match_fn: str = "starts_or_endswith",
) -> str:
    sample_kwargs = sample_kwargs or {}
    
    print("mg_specs", mg_specs)

    # get choice strings
    if mg_specs.choice_strings is not None:
        choice_strings = get_choice_strings(mg_specs.choice_strings, n=n)
    else:
        choice_strings = None

    # append answer prompt
    prompt = mg_specs.prompt
    
    eval_type = mg_specs.eval_type if mg_specs.eval_type else "cot_classify"
            
    prompt = append_answer_prompt(
        lang=self.language,
        prompt=prompt,
        eval_type=eval_type,
        choice_strings=choice_strings,
        answer_prompt=mg_specs.output_template,
    )
        
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        return False, "Prompt is not a string"
        
    # call model
    model_caller = ModelCaller(config={})
    evaluation_response, elapsed_time = model_caller.call(messages=messages, **sample_kwargs)
    evaluation_answer = ModelCaller.parse_response(self, evaluation_response)
    evaluation_response.elapsed_time = elapsed_time
    evaluation_response.prompt = json.dumps(messages)
    evaluation_response.completion = evaluation_answer
    
    print("answer", evaluation_answer)

    # result_score = self.extract_score(evaluation_answer)

    # get choice option
    if choice_strings:
        choice = get_choice(evaluation_answer, eval_type, match_fn, choice_strings)
        print("choice", choice)
    else:
        choice = None

    # get score
    if choice and choice != INVALID_STR and mg_specs.choice_scores is not None:
        score = get_choice_score(choice, choice_strings, mg_specs.choice_scores)
    elif evaluation_answer is not None:
        # try to get a score from the evaluation
        try:
            score = get_score(evaluation_answer)
        except ValueError:
            score = None
    else:
        score = None
        
    if score > 0 and mg_specs.threshold:
        pass_eval = score >= mg_specs.threshold
    else:
        pass_eval = None

    return dict(
        completion = evaluation_answer,
        score=score,
        choice=choice,
        pass_eval=pass_eval,
        sampled=[evaluation_answer],
        eval_prompt=prompt,
        invalid_choice=choice == INVALID_STR,
        raw_response=evaluation_response
    )
from zeval.utils.model import ModelCaller
from typing import Any, Callable, Dict
from datasets import Dataset
import logging
import json
logging.basicConfig(level=logging.INFO)

class ContextPrecision:
    name = 'context_precision'

    PROMPTS = {
        'en': {
            'CONTEXT_PRECISION_TEMPLATE': """\
Given a question and a context, verify if the information in the given context is useful in answering the question. Return a Yes/No answer.
question: {question}
context:
{context}
answer:
""",
            'REASONING_MESSAGE': """
My reasoning process is as follows:
I will go through each context, verify whether the information in the given context helps answer the question, and return a "Yes" or "No" answer, which I will convert to True or False accordingly.

Here are my judgments for each context:
{judges}

Then, by calculating the cumulative precision of all the correct responses (True) up to that point, the average precision score is: {result}
"""
        },
        'zh': {
            'CONTEXT_PRECISION_TEMPLATE': """\
给定一个问题和一个上下文，验证给定的上下文中的信息是否有助于回答这个问题。返回一个“Yes”或“No”的答案。
问题：{question}
上下文：
{context}
答案：
""",
            'REASONING_MESSAGE': """
我的推理过程如下：
我会遍历每个上下文信息，验证给定的上下文中的信息是否有助于回答这个问题。返回一个“Yes”或“No”的答案，并将其转化为True或False的相应。

以下依次是我对每个上下文的判断：
{judges}

然后，通过计算排在前面的所有正确响应（True）的累计精度的平均值，得出的平均精度得分是：{result}
"""
        }
    }

    def __init__(self, lang='zh'):
        self.language = lang

    def get_prompt(self, key):
        return self.PROMPTS[self.language][key]

    def eval(self, dataset: Dataset, sample_kwargs: Dict[str, Any] = None):
        if dataset is None:
            params = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        question = dataset["question"]
        context = dataset["context"]
        
        """
        Compute context precision based on the question and context.
        """
        determine_results, determine_responses = self.determine_usefulness(question, context, sample_kwargs)

        result_score = self.calculate_average_precision(determine_results)

        string_results = [str(result) for result in determine_results]
        judges = ",".join(string_results)

        reasoning = self.get_prompt('REASONING_MESSAGE').format(judges=judges, result=result_score)
        eval_result = {"score": result_score, "reasoning": reasoning, "responses": determine_responses}
        return eval_result

    def determine_usefulness(self, question: str, contexts: list[str], sample_kwargs: dict = None) -> list[bool]:
        determine_responses = []
        determine_results = []
        for context in contexts:
            determine_prompt = self.get_prompt('CONTEXT_PRECISION_TEMPLATE').format(question=question, context=context)
            if isinstance(determine_prompt, str):
                messages = [{"role": "user", "content": determine_prompt}]

            determine_response, elapsed_time = ModelCaller.call(self, messages=messages, **sample_kwargs)
            determine_result = ModelCaller.parse_response(self, determine_response)
            
            determine_response.elapsed_time = elapsed_time
            determine_response.prompt = json.dumps(determine_prompt)
            determine_response.completion = determine_result

            determine_results.append("Yes" in determine_result)
            determine_responses.append(determine_response)

        return determine_results, determine_responses

    def calculate_average_precision(self, responses: list[bool]) -> float:
        if not responses:
            return 0.0
        numerator, denominator = 0, 0
        for i, resp in enumerate(responses, start=1):
            if resp:
                numerator += sum(responses[:i]) / i
                denominator += 1
        return numerator / denominator if denominator > 0 else 0.0

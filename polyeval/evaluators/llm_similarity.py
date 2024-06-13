from polyeval.utils.model import ModelCaller
from typing import Any, Callable, Dict
from datasets import Dataset
import logging
import re, json, logging
logging.basicConfig(level=logging.INFO)

class LLMSimilarity:
    name = 'llm_similarity'

    PROMPTS = {
        'en': {
            'PROMPT_TEMPLATE': """
Task: Estimate the correlation between the two given strings. In your evaluation, consider not only direct connections but also indirect and subtle associations.
For example, if 'bridge' appears in the first string and 'water' in the second, consider the compound term 'water bridge.' Similarly, for 'book' and 'wisdom,' consider phrases like 'books are keys to wisdom.'
You should use the following scoring scale from 0.00 to 1.00 to assess correlation, where:
0.00 signifies no correlation whatsoever.
0.50 indicates a moderate level of correlation. This means there are several significant connections between the terms in the two strings, but these are not overwhelming.
1.00 is reserved for ONLY two strings that are completely identical.

Strings to compare:
string_one: {answer}
string_two: {ideal}

Your final output should follow this format (note: include the brackets):
Reasoning: <your reasoning>
Final answer: [<float, rounded to the 100th place>]""",
            'REASONING_MESSAGE': """
My reasoning process is as follows:
First, I generated the most likely question for the given answer, then I converted the generated question and the original question into vectors using embeddings, and finally calculated the cosine similarity between the vectors.

The generated answer by the large model is:
{answer}

The question I generated is:
{gen_question}

The original question provided by the user is:
{question}

By analyzing the meaning of the two questions, the derived answer relevance score is: {result}
"""
        },
        'zh': {
            'PROMPT_TEMPLATE': """
任务：估计两个给定字符串之间的相关度。在你的评估中，不仅要考虑直接的联系，还要考虑间接和微妙的关联。
例如，如果第一个字符串中出现了‘桥’，第二个中出现了‘水’，你可以考虑组合词‘水上桥梁’。同样地，对于‘书’和‘智慧’，考虑像‘书籍是智慧的钥匙’这样的短语。
你应该使用以下从0.00到1.00的评分标准来评估相关性，其中：
0.00表示完全没有相关性。
0.50表示中等程度的相关性。这意味着两个字符串中的术语有几个显著的联系，但这些联系不是压倒性的。
1.00仅保留给完全相同的两个字符串。

要比较相关度的字符串：
string_one: {answer}
string_two: {ideal}

你的最终输出应该按照以下格式（注意：包括方括号）：
理由：<your reasoning>
最终答案：[<float, rounded to the 100th place>]""",
            'REASONING_MESSAGE': """
我的推理过程如下：
首先，我根据给定的答案生成了一个最可能的问题，然后我将生成的问题和原始问题分别通过embedding转化成向量，然后再计算向量之间的余弦相似度。

大模型生成的答案是：
{answer}

我由此生成的问题是：
{gen_question}

用户原来的问题是：
{question}

通过分析两个问题的语义，我给出的回答相似度得分是：{result}
"""
        }
    }
    
    PRE_ANSWER = {
        'en': 'Final answer:',
        'zh': '最终答案：'
    }

    def __init__(self, lang='zh'):
        self.language = lang

    def get_prompt(self, key):
        return self.PROMPTS[self.language][key]

    def eval(self, dataset: Dataset, **kwargs):
        if dataset is None:
            return False, "No dataset provided"

        print("sample_kwargs trying:", kwargs)
        sample_kwargs = kwargs.get('llm', None)
        if not sample_kwargs:
            return False, "No sampling parameters provided"

        context = '\n'.join(dataset["context"])
        
        question = dataset.get("question", None)
        answer = dataset.get("answer", None)
        ideal = dataset.get("ideal", None)
        
        ideal = "\n".join(ideal) if isinstance(ideal, list) else ideal

        prompt = self.get_prompt('PROMPT_TEMPLATE').format(answer=answer, ideal=ideal)
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]

        similarity_response, elapsed_time = ModelCaller.call(self, messages=messages, **sample_kwargs)
        similarity_answer = ModelCaller.parse_response(self, similarity_response)
        similarity_response.elapsed_time = elapsed_time
        similarity_response.prompt = json.dumps(prompt)
        similarity_response.completion = similarity_answer

        result_score = self.extract_score(similarity_answer)

        reasoning = similarity_answer
        responses = [similarity_response]
        eval_result = {"score": result_score, "reasoning": reasoning, "responses": responses}
        
        print("similar eval_result", eval_result)

        return eval_result

    def extract_score(self, response_content: str) -> float:
        """
        Extracts the similarity score from the content of a GPT model's response.

        Args:
            response_content: The content of a GPT model's response.

        Returns:
            The similarity score as a float. If no score could be extracted, returns 0.0.
        """
        try:
            pre_answer = self.PRE_ANSWER[self.language]  
            match = re.search(f"{pre_answer}\s*\[(.+?)]", response_content).group(1)
            score = float(match)
            logging.debug(f"response_content: {response_content}, score: {score}")
        except AttributeError as e:
            print(e)
            score = 0.0
            logging.warning("Answer not found in response, score set to 0, will autofail validation scoring.")
        return score
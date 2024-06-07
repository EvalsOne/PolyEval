from zeval.utils.model import ModelCaller
from typing import Any, Callable, Dict
from datasets import Dataset
import logging
import re, json, logging
logging.basicConfig(level=logging.INFO)

class AnswerRelevancy:
    name = 'answer_relevancy'

    PROMPTS = {
        'en': {
            'QUESTION_TEMPLATE': """
Generate question for the given answer.
Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India 
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?

Answer:{answer}
Question:
""",
            'CORRELATION_PROMPT_TEMPLATE': """
Task: Estimate the correlation between the two given strings. In your evaluation, consider not only direct connections but also indirect and subtle associations.
For example, if 'bridge' appears in the first string and 'water' in the second, consider the compound term 'water bridge.' Similarly, for 'book' and 'wisdom,' consider phrases like 'books are keys to wisdom.'
You should use the following scoring scale from 0.00 to 1.00 to assess correlation, where:
0.00 means no correlation at all.
0.50 means moderate correlation. This implies several significant links between the terms in the two strings, but not overwhelmingly so.
1.00 is reserved for two identical strings.

Strings to compare:
string_one: {statement_one}
string_two: {statement_two}

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
            'QUESTION_TEMPLATE': """
为给定的答案生成问题。
答案：
PSLV-C56任务预定于2023年7月30日星期日IST上午6:30 / UTC凌晨1:00发射。它将从位于印度安得拉邦斯里赫里戈塔的萨蒂什·达瓦恩航天中心发射。
问题：
PSLV-C56任务的预定发射日期和时间是什么时候？它将从哪里发射？

答案：{answer}
问题：
""",
            'CORRELATION_PROMPT_TEMPLATE': """
任务：估计两个给定字符串之间的相关度。在你的评估中，不仅要考虑直接的联系，还要考虑间接和微妙的关联。
例如，如果第一个字符串中出现了‘桥’，第二个中出现了‘水’，你可以考虑组合词‘水上桥梁’。同样地，对于‘书’和‘智慧’，考虑像‘书籍是智慧的钥匙’这样的短语。
你应该使用以下从0.00到1.00的评分标准来评估相关性，其中：
0.00表示完全没有相关性。
0.50表示中等程度的相关性。这意味着两个字符串中的术语有几个显著的联系，但这些联系不是压倒性的。
1.00仅保留给完全相同的两个字符串。

要比较相关度的字符串：
string_one: {statement_one}
string_two: {statement_two}

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

    def eval(self, dataset: Dataset, sample_kwargs: Dict[str, Any] = None):
        if dataset is None:
            params = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        question = dataset["question"]
        answer = dataset["answer"]
        context = '\n'.join(dataset["context"])

        question_prompt = self.get_prompt('QUESTION_TEMPLATE').format(answer=answer)
        if isinstance(question_prompt, str):
            messages = [{"role": "user", "content": question_prompt}]

        gen_question_response, elapsed_time = ModelCaller.call(self, messages=messages, **sample_kwargs)
        gen_question = ModelCaller.parse_response(self, gen_question_response)
        gen_question_response.elapsed_time = elapsed_time
        gen_question_response.prompt = json.dumps(question_prompt)
        gen_question_response.completion = gen_question

        correlation_prompt = self.get_prompt('CORRELATION_PROMPT_TEMPLATE').format(statement_one=question, statement_two=gen_question)
        if isinstance(correlation_prompt, str):
            messages = [{"role": "user", "content": correlation_prompt}]

        correlation_response, elapsed_time = ModelCaller.call(self, messages=messages, **sample_kwargs)
        correlation_result = ModelCaller.parse_response(self, correlation_response)
        correlation_response.elapsed_time = elapsed_time
        correlation_response.prompt = json.dumps(correlation_prompt)
        correlation_response.completion = correlation_result

        result_score = self.extract_score(correlation_result)

        reasoning = self.get_prompt('REASONING_MESSAGE').format(answer=answer, gen_question=gen_question, question=question, result=result_score)

        responses = [gen_question_response, correlation_response]
        eval_result = {"score": result_score, "reasoning": reasoning, "responses": responses}

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
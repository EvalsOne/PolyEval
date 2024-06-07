from zeval.utils.model import ModelCaller
from typing import Any, Callable, Dict
from datasets import Dataset
import logging
import json
logging.basicConfig(level=logging.INFO)

class ContextRecall:
    name = 'context_recall'

    PROMPTS = {
        'en': {
            'CONTEXT_RECALL_RA': """
Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not.
Think in steps and reason before coming to conclusion.

context: Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
answer: Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895 
classification
1. Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. The date of birth of Einstein is mentioned clearly in the context. So [Attributed]
2. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. The exact sentence is present in the given context. So [Attributed]
3. He published 4 papers in 1905. There is no mention about papers he wrote in given the context. So [Not Attributed]
4. Einstein moved to Switzerland in 1895. There is not supporting evidence for this in the given the context. So [Not Attributed]

context:{context}
answer:{ideal}
classification:
""",
            'REASONING_MESSAGE': """
My reasoning process is as follows:
First, I analyze each sentence of the ideal answer to determine if it can be attributed to the given context.

Ideal answer: {ideal}
Context: {context}

Here are my classifications:
{statements}

Then, I calculate the score based on the number of attributed sentences. The resulting score is: {final_score}
"""
        },
        'zh': {
            'CONTEXT_RECALL_RA': """
给定一个上下文和一个答案，请分析答案中的每个句子，判断这个句子是否能归因于给定的上下文。
思考步骤并在得出结论前进行推理。

上下文：爱因斯坦（1879年3月14日 - 1955年4月18日）是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他最著名的成就是发展相对论理论，同时他也对量子力学做出了重要贡献，因此在20世纪初几十年现代物理学在重新塑造自然科学理解方面的革命性成就中，他是一个核心人物。他的质能等价公式E=mc²源自相对论，被称为“世界上最著名的方程”。他因“对理论物理的贡献，特别是发现光电效应定律”而获得1921年的诺贝尔物理学奖，这是量子理论发展中的一个关键步骤。他的工作也因对科学哲学的影响而闻名。在1999年由英国《物理世界》杂志组织的全球130位顶尖物理学家的调查中，爱因斯坦被评为有史以来最伟大的物理学家。他的智力成就和原创性使爱因斯坦成为天才的代名词。
答案：爱因斯坦出生于1879年3月14日，是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大和最具影响力的科学家之一。他因“对理论物理的贡献”获得了1921年诺贝尔物理学奖。他在1905年发表了4篇论文。爱因斯坦于1895年搬到了瑞士。
分类：
1. 爱因斯坦出生于1879年3月14日，是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大和最具影响力的科学家之一。爱因斯坦的出生日期在上下文中有明确提及。因此[归因]
2. 他因“对理论物理的贡献”获得了1921年诺贝尔物理学奖。给定上下文中有这个确切的句子。因此[归因]
3. 他在1905年发表了4篇论文。在给定的上下文中没有提到他写的论文。因此[未归因]
4. 爱因斯坦于1895年搬到了瑞士。在给定的上下文中没有支持这一点的证据。因此[未归因]

上下文：
{context}
答案：{ideal}
分类：
""",
            'REASONING_MESSAGE': """
我的推理过程如下：
首先，我分析理想答案(ideal)中的每个句子，判断该句子是否能归因于给定的上下文(Context)。

理想答案：{ideal}
上下文：
{context}

以下是我的分类：
{statements}

然后，我根据归因的句子数量计算得分，得出的得分是：{final_score}
"""
        }
    }
    
    ATTRIBUTE_WORDS = {
        'en': {
            'yes': '[Attributed]',
            'no': '[No Attributed]'
        },
        'zh': {
            'yes': '[归因]',
            'no': '[未归因]'
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

        context = dataset["context"]
        ideal = dataset["ideal"]

        gt = "\n".join(ideal) if isinstance(ideal, list) else ideal
        ctx = "\n".join(context) if isinstance(context, list) else context
        
        classification, attribute_response = self.classify_context_recall(ctx, gt, sample_kwargs)
        result_score = self.calculate_score(classification)

        responses = [attribute_response]
        reasoning = self.get_prompt('REASONING_MESSAGE').format(ideal=gt, context=ctx, statements="\n".join(classification), final_score=result_score)
        eval_result = {"score": result_score, "reasoning": reasoning, "responses": responses}

        return eval_result

    def classify_context_recall(self, context: str, ideal: str, sample_kwargs: dict = None) -> list[str]:
        prompt = self.get_prompt('CONTEXT_RECALL_RA').format(context=context, ideal=ideal)
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        attribute_response, elapsed_time = ModelCaller.call(self, messages=messages, **sample_kwargs)
        attribute_answer = ModelCaller.parse_response(self, attribute_response)
        attribute_response.elapsed_time = elapsed_time
        attribute_response.prompt = json.dumps(prompt)
        attribute_response.completion = attribute_answer

        classification = []
        attributed = self.ATTRIBUTE_WORDS[self.language]['yes']
        for line in attribute_answer.split('\n'):
            if attributed in line or attributed in line:
                classification.append(line.strip())

        return classification, attribute_response

    def calculate_score(self, classification: list[str]) -> float:
        numerator = sum(self.ATTRIBUTE_WORDS[self.language]['yes'] in sentence for sentence in classification)
        denominator = len(classification)
        return numerator / denominator if denominator > 0 else 0

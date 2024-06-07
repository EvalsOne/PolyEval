from zeval.utils.model import ModelCaller
from typing import Any, Callable, Dict
from datasets import Dataset
import logging
import json
logging.basicConfig(level=logging.INFO)

class Faithfulness:
    name = 'faithfulness'

    PROMPTS = {
        'en': {
            'LONG_FORM_ANSWER_PROMPT': """\
Given a question and answer, create one or more statements from each sentence in the given answer.
question: Who was Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements:
Albert Einstein was born in Germany.
Albert Einstein was best known for his theory of relativity.
question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alcohol
statements:
Cadmium Chloride is slightly soluble in alcohol.
question: Were Shahul and Jithin of the same nationality?
answer: They were from different countries.
statements:
Shahul and Jithin were from different countries.
question:{question}
answer: {answer}
statements:
""",
            'NLI_STATEMENTS_MESSAGE': """
Prompt: Natural language inference
Consider the given context and following statements, then determine whether they are supported by the information present in the context.Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

Context:
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
statements:
1. John is majoring in Biology.
2. John is taking a course on Artificial Intelligence.
3. John is a dedicated student.
4. John has a part-time job.
5. John is interested in computer programming.

Answer:
1. John is majoring in Biology.
Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
2. John is taking a course on Artificial Intelligence.
Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI. Verdict: No.
3. John is a dedicated student.
Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
4. John has a part-time job.
Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job.  Verdict: No.
5. John is interested in computer programming.
Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
Final verdict for each statement in order: No. No. Yes. No. Yes.
context:
{context}
statements:
{statements}
Answer:
""",
            'REASONING_MESSAGE': """
My reasoning process is as follows:
First, I created the following statements from each sentence of the given answer:
{statements}
Then, I considered the context information and conducted natural language inference for each statement. Finally, I derived the final judgment for each statement.
{final_statement}
"""
        },
        'zh': {
            'LONG_FORM_ANSWER_PROMPT': """\
给定一个问题和答案，请从给定答案的每个句子中创建一个或多个陈述。
问题：阿尔伯特·爱因斯坦是谁，他最著名的是什么？
答案：他是一名出生在德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的物理学家之一。他最著名的是发展相对论理论，他还为量子力学理论的发展做出了重要贡献。
陈述：
阿尔伯特·爱因斯坦出生在德国。
阿尔伯特·爱因斯坦以他的相对论理论而闻名。
问题：氯化镉在这种化学物质中略微可溶，它还被称为什么？
答案：酒精
陈述：
氯化镉在酒精中略微可溶。
问题：Shahul和Jithin是同一国籍吗？
答案：他们来自不同的国家。
陈述：
Shahul和Jithin来自不同的国家。
问题：{question}
答案：{answer}
陈述：
""",
            'NLI_STATEMENTS_MESSAGE': """
提示：自然语言推理
考虑给定的背景和以下陈述，然后确定这些陈述是否得到了上下文中的信息支持。在得出结论（是/否）之前，为每个陈述提供简要说明。在最后以指定格式为每个陈述提供最终裁决。不要偏离指定格式。

上下文：
约翰是XYZ大学的学生。他正在攻读计算机科学学位。他这学期注册了几门课程，包括数据结构、算法和数据库管理。约翰是一个勤奋的学生，花大量时间学习和完成作业。他经常待在图书馆里晚上工作在他的项目上。
陈述：
1.约翰主修生物学。
2.约翰正在上一门关于人工智能的课程。
3.约翰是一个专注的学生。
4.约翰有一份兼职工作。
5.约翰对计算机编程感兴趣。

答案：
1.约翰主修生物学。
解释：明确提到约翰的专业是计算机科学。没有信息表明他主修生物学。 裁决：No.
2.约翰正在上一门关于人工智能的课程。
解释：上下文提到约翰当前注册的课程，而没有提到人工智能。因此，不能推断约翰正在上人工智能课程。裁决：No.
3.约翰是一个专注的学生。
解释：提示中提到他花大量时间学习和完成作业。此外，还提到他经常待在图书馆里晚上工作在他的项目上，这暗示了他的专注。裁决：Yes.
4.约翰有一份兼职工作。
解释：上下文中没有提供约翰有兼职工作的信息。因此，不能推断约翰有兼职工作。裁决：No.
5.约翰对计算机编程感兴趣。
解释：上下文提到约翰正在攻读计算机科学学位，这暗示了他对计算机编程的兴趣。裁决：No.
每个陈述的最终裁决顺序：No. No. Yes. No. No.
上下文：
{context}
陈述：
{statements}
答案：
""",
            'REASONING_MESSAGE': """
我的推理过程如下：
首先，我从给定答案的每个句子中创建以下的陈述：
{statements}
然后，我考虑了上下文信息，并对每个陈述进行了自然语言推理。最后，我得出了每个陈述的最终裁决。
{final_statement}
"""
        }
    }
    
    PRE_ANSWER = {
        'en': 'Final verdict for each statement in order:',
        'zh': '每个陈述的最终裁决顺序：'
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
        context = "\n".join(dataset["context"])

        human_prompt = self.get_prompt('LONG_FORM_ANSWER_PROMPT').format(question=question, answer=answer)
        
        messages = [
            {"content": human_prompt, "role": "user"},
        ]
        statements_response, elapsed_time = ModelCaller.call(self, messages=messages, **sample_kwargs)
        
        statements = ModelCaller.parse_response(self, statements_response)

        statements_response.elapsed_time = elapsed_time
        statements_response.prompt = json.dumps(human_prompt)
        statements_response.completion = statements

        nli_prompt = self.get_prompt('NLI_STATEMENTS_MESSAGE').format(context=context, statements=statements)
        
        messages = [
            {"content": nli_prompt, "role": "user"},
        ]
        nli_response, elapsed_time = ModelCaller.call(self, messages=messages, **sample_kwargs)
                
        final_statement = ModelCaller.parse_response(self, nli_response)
        nli_response.elapsed_time = elapsed_time
        nli_response.prompt = json.dumps(nli_prompt)
        nli_response.completion = final_statement
        
        result_score = self.analyze_nli_response(nli_response.completion)

        responses = [statements_response, nli_response]
        reasoning = self.get_prompt('REASONING_MESSAGE').format(statements=statements, final_statement=final_statement)
        
        eval_result = {"score": result_score, "reasoning": reasoning, "responses": responses}

        return eval_result

    def analyze_nli_response(self, final_statement):
        if not final_statement:
            return 0.0
        pre_answer = self.PRE_ANSWER[self.language]
        if pre_answer in final_statement:
            final_statement = final_statement[final_statement.find(pre_answer) + len(pre_answer):]
            final_statement = final_statement.strip().split(".")
            final_statement = list(filter(None, final_statement))

            score = sum(
                0 if "Yes" in answer else 1
                for answer in final_statement
            )
            score = score / len(final_statement)
        else:
            score = 0.0
        return 1 - score
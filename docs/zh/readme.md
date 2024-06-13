 # PolyEval: 支持多语言的LLM系统评估框架

 
 PolyEval是一个多元化的针对LLM系统的评估框架。它继承了众多基于大语言模型提示语和算法规则的评估器，原生支持多语言的提示语模版和评估理由，方便接入各种生成和嵌入模型，可以基于YAML轻松扩展自己的评估器，而且项目的体积非常小。

## 安装方式

```bash
pip install polyeval
```

如果需要在本地进行开发，可以通过以下方式安装：

```bash
pip install -e .
```

## 使用方式

调用方式示例：

```python
import os
from datasets import Dataset 
from polyeval import evaluate, LLMSimilarity

os.environ["OPENAI_API_KEY"] = "your-openai-key"

dataset = Dataset.from_dict({
    `question`: [`When was the first super bowl?`],
    `answer`: [`The first superbowl was held on Jan 15, 1967`],
    `context` : [[`The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,`]],
    `ideal`: [`The first superbowl was held on January 15, 1967`]
})

kwargs = {}
kwargs[`llm`] = {`provider`: `openai`, `model`: `gpt-4o`, `temperature`: 0.0}
eval_results = evaluate(dataset, [faithfulness], "en", **kwargs)
```
其中，`Faithfulness`是一个RAG评估器，`dataset`是一个包含了问题、答案、上下文和理想答案的数据集，`en`是提示语模版的语言，`**kwargs`是传递给评估器的参数，不同评估器需要接收的参数是不同的。

如果评估中需要调用大语言模型，则需要在kwargs中包含`llm`参数，并将调用模型所需的参数都包含在字典中；如果评估中需要调用嵌入模型，则需要在kwargs中包含`embedding`参数，并将调用模型所需的参数都包含在字典中。

| 参数名称        | 格式说明                                                        |
|-----------------|-----------------------------------------------------------------|
| grade           | 整数/字符串格式，表示评估的等级                                   |
| score           | 浮点数格式，表示评估的分数                                      |
| pass_eval       | 布尔值格式，表示评估是否通过                                    |
| reasoning       | 字符串格式，提供评估的详细原因或解释                              |
| responses       | 对象列表格式，其中包含每次大模型调用的ModelResponse对象。<br/>每个对象中额外封装了一些其他属性，具体包括： |

ModelResponse对象结构中的额外属性：
| 属性名称        | 格式说明                              |
|-----------------|---------------------------------------|
| elapsed_time    | 浮点数格式，表示响应所用的时间（以秒为单位）|
| prompt          | 字符串格式，表示发送给模型的提示语        |
| completion      | 字符串格式，表示模型的生成结果            |


## 支持的评估器种类

### **RAG评估器** 
支持RAG评估相关的四种不同评估指标。关于RAG评估体系可以参见[Ragas的介绍文档](https://docs.ragas.io/en/stable/concepts/metrics/index.html)。我们在PolyEval中实现了四种RAG评估器，并实现了模版提示语的多语言支持。结果的评分为0或1之间的任意小数。

### **基于YAML进行大模型评估** 
可以通过传入自定义的YAML配置文件进行大模型评估。这种评估方式的优势在于可以根据具体的评估需求自定义评估模版，从而实现更加灵活的评估。YAML配置文件的格式要求可以参考[https://docs.evalsone.com/Faq/Metrics/setting_custom_rating_metrics/]。结果的形式分为评级、分数和判定三种，具体取决于YAML配置文件。

### **文本相似度比较** 
支持多种文本相似度的比较方式，具体包括：
- 基于字符串距离（调用LangChain的评估器实现），结果得分范围根据具体的比较算法而定。
- 基于嵌入距离（调用LangChain的评估器实现），结果得分范围根据具体的比较算法而定。
- 基于大模型进行语义比较，结果的评分为0或1之间的任意小数。
    
### **JSON评估** 
支持多种JSON相关的评估方式，具体包括：
- JSON格式校验
- JSON Matchness
- JSON Schema Validate
结果的评分为0或1。0表示不通过，1表示通过。

### **文本匹配**
支持多种文本匹配的方式，具体包括 `match`, `include`, `startswith`, `endswith`, `fuzzy_match`，并可以设置在比较过程中是否忽略大小写。结果的评分为0或1。0表示不通过，1表示通过。

### **正则表达式评估** 
可以通过自定义正则表达式评估。结果的评分为0或1。0表示不通过，1表示通过。

## 支持的评估器列表

| 评估器名称 | 分类       | 评估器介绍                   | 实现机制  | 所需kwargs参数                | 所需数据组件                   |
|---------------------|---------------|----------------------|-------------|------------------|------------------------|
| [Faithfulness](/polyeval/faithfulness.py)       | RAG评估          | 评估生成结果的准确性              | 基于大模型提示语  | `llm`: 生成模型的调用参数，字典格式                    |     `answer` `context`     |
| [AnswerRelevancy](/polyeval/answer_relevancy.py)    | RAG评估          | 评估生成结果回答的相关性            | 基于大模型提示语  | `llm`: 生成模型的调用参数，字典格式                 |     `question` `answer`       |
| [ContextRecall](/polyeval/context_recall.py)      | RAG评估          | 评估上下文召回率                | 基于大模型提示语  | `llm`: 生成模型的调用参数，字典格式                 |   `context` `ideal`   |
| [ContextPrecision](/polyeval/context_precision.py)   | RAG评估          | 评估上下文准确性                | 基于大模型提示语  | `llm`: 生成模型的调用参数，字典格式                 | `question` `context` |
| [CustomYaml](/polyeval/custom_yaml.py)         | 大模型评估         | 基于自定义的YAML配置进行评估    | 基于大模型提示语       |  `yaml_specs`: 评估配置文件, yaml格式<br/>  `llm`: 生成模型的调用参数，字典格式           |  `question` `answer` `context` `ideal`    |
| [StringDistanceEvaluator](/polyeval/string_distance.py)          | 文本相似度         | 比较字符串之间的距离             | 基于比较算法 | `metric`: 文本比较算法，字符串格式式                |     `answer` `ideal`     |
| [EmbeddingDistanceEvaluator](/polyeval/embedding_distance.py)           | 文本相似度         | 基于向量嵌入的距离计算相似度         | 基于嵌入模型和比较算法     | `metric`: 向量比较算法，字符串格式<br/> `embedding`: 生成模型的调用参数，字典格式<br/>                 |    `answer` `ideal`    |
| [LLMSimilarity](/polyeval/llm_similarity.py)        | 文本相似度         | 使用大模型进行语义层面的比较        | 基于大模型提示语    |      `llm`: 生成模型的调用参数，字典格式           |      `answer` `ideal`          |
| [IsJson](/polyeval/json.py)           | JSON评估        | 校验JSON的格式是否正确           | 基于规则 |                   |       `answer`        |
| [JsonMatch](/polyeval/json.py)       | JSON评估        | 评估JSON数据的一致性             | 基于规则 |                   |    `answer` `ideal`     |
| [JsonSchemaMatch](/polyeval/json.py) | JSON评估        | 校验JSON是否符合预定义的Schema    | 基于规则 | `schema`: 预定义的JSON schema，JSON格式 |   `answer`   |
| [Matchness](/polyeval/matchness.py)           | 文本匹配      | 通过多种方式比较生成结果与理想答案是否匹配           | 基于规则 |  `match_rule`: 匹配规则<br/>   `ignore_case`: 是否忽略大小写          |   `answer` `ideal`   |
| [Regex](/polyeval/regex.py)           | 正则表达式评估      | 自定义正则表达式进行评估           | 基于规则 |  `regex`: 用于校验的正则表达式             |   `answer` `ideal`   |


## 多语言支持

虽然一些评估框架提供了多语言的适配方案，但PolyEval的目标是提供更加完整的多语言支持，从评估提示语模版到计算规则等。目前调用评估器时使用的语言参数`lang`可以是`en`或`zh`，分别代表英文和简体中文。

```python
# 使用英文提示语模版进行faithfulness评估
eval_results = evaluate(dataset, [faithfulness], "en", **kwargs)

# 使用简体中文提示语模版进行faithfulness评估
eval_results = evaluate(dataset, [faithfulness], "zh", **kwargs)
```

未来，我们希望能够在社区的支持和帮助下，逐步增加更多语言的支持。

以下是一个大模型评级的YAML配置示例：
    
```yaml
prompt: |-
  You are comparing a submitted answer to an expert answer on a given question. Here is the data:
  [BEGIN DATA]
  ************
  [Question]: {input}
  ************
  [Expert]: {ideal}
  ************
  [Submission]: {completion}
  ************
  [END DATA]

  Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
  The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
  (A) The submitted answer is a subset of the expert answer and is fully consistent with it.
  (B) The submitted answer is a superset of the expert answer and is fully consistent with it.
  (C) The submitted answer contains all the same details as the expert answer.
  (D) There is a disagreement between the submitted answer and the expert answer.
  (E) The answers differ, but these differences don`t matter from the perspective of factuality.
eval_type: cot_classify
choice_strings: ABCDE
choice_scores:
  "A": 0.8
  "B": 0.8
  "C": 0.8
  "D": 0.0
  "E": 0.5
threshold: 0.5
reverse_score: 0
answer_prompt: ""
```

具体的YAML配置文件格式可以参考[https://docs.evalsone.com/Faq/Metrics/setting_custom_rating_metrics/]

## 模型的集成 Model Integration

PolyEval能够通过UnionLLM可以自由集成数百种大语言模型用于评估，还通过LangChain使用嵌入模型。

例如：如果在调用faithfulness评估器时，需要使用OpenAI的gpt-4o模型，可以通过以下方式进行配置：

```python
kwargs = {}
kwargs[`llm`] = {`provider`: `openai`, `model`: `gpt-4o`, `temperature`: 0.0}
eval_results = evaluate(dataset, [faithfulness], "en", **kwargs)
```

如果更换成Anthropic提供的模型，只需要将传入的字典更新成：

```python
kwargs = {}
kwargs[`llm`] = {`provider`: `anthropic`, `model`: `claude-3-orpus`, `temperature`: 0.0}
eval_results = evaluate(dataset, [faithfulness], "en", **kwargs)
```

通过UnionLLM调用各种大模型的方式可以参照：
https://github.com/EvalsOne/UnionLLM

如果评估中需要使用嵌入模型，可以通过以下方式进行配置：

```python
kwargs = {}
kwargs[`embedding`] = {`provider_cls`: `OpenAIEmbeddings`, `model`: `text-embedding-3-large`}
```

目前`provider_cls`暂时只支持OpenAIEmbeddings, CohereEmbeddings和BaichuanTextEmbeddings三种，未来的版本会扩展更多嵌入模型的支持。

## 单元测试

PolyEval提供了一系列单元测试脚本，可以通过以下方式进行批量测试。测试前需要安装pytest，并且在脚本中将调用模型所必需的大模型环境变量设置好，如API Key。

```bash
pytest tests
```

或者，可以通过以下方式进行单个测试（忽略警告）：

```bash
pytest tests/test_faithfulness.py -W ignore
```

## 使用EvalsOne云端评估

[EvalsOne](https://evalsone.com)是一个评估LLM系统的完整解决方案，它不仅提供了云端的评估运行环境（PolyEval中的所有评估器都可以在EvalsOne运行），还提供了方便的样本管理、模型调用、可视化结果分析，以及通过Fork的方式迭代评估等功能。[了解更多](https://docs.evalsone.com)

## 参与PolyEval项目

PolyEval欢迎社区参与和贡献。用户可以通过提交代码、改进文档、提供多语言的提示语模版，或提供新的评估器算法来参与到项目的发展中。社区的积极参与将有助于不断提升PolyEval的功能和性能，推动大语言模型评估技术的发展。

## ❤️ 感谢

在项目开发过程中，我们从Ragas, OpenAI Evals中得到了很多启发，并参考了它们的实现代码和评估配置，为此我们在这里表示深深的感谢。
 PolyEval是一个多元化的针对LLM系统的评估框架。它继承了众多基于大语言模型提示语和算法规则的评估器，原生支持多语言的提示语模版，方便接入各种生成和嵌入模型，轻量级并可以基于YAML轻松扩展。

# 使用方式示例

```python
import os
from datasets import Dataset 
from polyeval import evaluate, LLMSimilarity

os.environ["OPENAI_API_KEY"] = "your-openai-key"

dataset = Dataset.from_dict({
    'question': ['When was the first super bowl?'],
    'answer': ['The first superbowl was held on Jan 15, 1967'],
    'context' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,']],
    'ideal': ['The first superbowl was held on January 15, 1967']
})

kwargs = {}
kwargs['llm'] = {'provider': 'openai', 'model': 'gpt-4o', 'temperature': 0.0}
eval_results = evaluate(dataset, [faithfulness], "en", **kwargs)
```
其中，`faithfulness`是一个RAG评估器，`dataset`是一个包含了问题、答案、上下文和理想答案的数据集，`en`是提示语模版的语言，`**kwargs`是传递给评估器的参数，不同评估器需要接收的参数是不同的。

如果评估中需要调用大语言模型，则需要在kwargs中包含`llm`参数，并将调用模型所需的参数都包含在字典中；如果评估中需要调用嵌入模型，则需要在kwargs中包含`embedding`参数，并将调用模型所需的参数都包含在字典中。

# 支持的评估器 Evaluators

- **RAG评估器** 支持RAG评估相关的四种不同评估指标
	- Faithfulness
	- Answer relevancy
	- Context recall
	- Context precision
- **基于YAML进行大模型评估** 兼容OpenAI Evals的YAML模版
	- 基于客观标准的评估
	- 基于事实进行大模型评级
	- 可以自定义YAML模版灵活扩展
- **文本相似度比较** 支持多种文本相似度的比较方式
	- 基于字符串距离
	- 基于嵌入距离
	- 基于大模型进行语义比较
- **JSON评估** 支持多种JSON相关的评估指标
	- JSON格式校验
	- JSON Matchness
	- JSON Schema Validate
- **正则表达式评估** 可以通过自定义正则表达式评估

# 完整支持多语言

虽然一些评估框架提供了多语言的适配方案，但PolyEval的目标是提供更加完整的多语言支持，从评估提示语模版到计算规则等。目前PolyEval支持英文和简体中文的提示语模版，也希望社区能够提供更多的支持。

# 基于YAML扩展自己的评估指标 Create custom YAML evaluators

PolyEval允许用户通过YAML文件轻松创建和扩展自己的评估模板。这种方式不仅简化了评估器的开发过程，还提高了评估框架的灵活性和适应性。

# 模型的集成 Model Integration

PolyEval能够通过UnionLLM集成各种大语言模型用于评估，还通过LangChain嵌入模型。

# 使用EvalsOne云端评估

PolyEval支持使用EvalsOne进行云端评估。这一功能使用户可以在云端运行大量评估任务，提升评估效率并减轻本地计算资源的压力。

# 贡献

PolyEval欢迎社区贡献。用户可以通过提交代码、改进文档、提供多语言的提示语模版，或提供新的评估器算法来参与到项目的发展中。社区的积极参与将有助于不断提升PolyEval的功能和性能，推动大语言模型评估技术的发展。

# 感谢

在项目开发过程中，我们参照了Ragas和OpenAI Evals的实现机制及提示语模版。
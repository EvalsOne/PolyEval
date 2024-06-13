# PolyEval: A Multilingual Evaluation Framework for LLM-based Systems

 English Version | [中文版](/docs/zh/readme.md)



PolyEval is a diversified evaluation framework for LLM systems. It inherits numerous evaluators based on large language model prompts and algorithmic rules, natively supports multilingual prompt templates and evaluation rationales, can easily integrate various generative and embedding models, can extend custom evaluators easily based on YAML, and the project's size is very small.

## Installation

```bash
pip install polyeval
```

If you need to develop locally, you can install it in the following way:

```bash
pip install -e .
```

## Usage

Example of usage:

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
Here, `Faithfulness` is a RAG evaluator, `dataset` is a dataset containing the question, answer, context, and ideal answer, `en` is the language of the prompt template, and `**kwargs` are the parameters passed to the evaluator, which vary for different evaluators.

If a large language model needs to be called during evaluation, the `llm` parameters must be included in kwargs, with all required model parameters included in the dictionary; if an embedding model needs to be called during evaluation, the `embedding` parameters must be included in kwargs, with all required model parameters included in the dictionary.

| Parameter Name   | Description                                                     |
|------------------|-----------------------------------------------------------------|
| grade            | Integer/String format, indicating the evaluation grade          |
| score            | Float format, indicating the evaluation score                   |
| pass_eval        | Boolean format, indicating whether the evaluation passed        |
| reasoning        | String format, providing detailed reasons or explanations for the evaluation |
| responses        | List of objects, each containing a ModelResponse object. <br/> Each object additionally encapsulates some other attributes, specifically: |

Extra attributes in the ModelResponse object structure:
| Attribute Name   | Description                               |
|------------------|-------------------------------------------|
| elapsed_time     | Float format, indicating the response time (in seconds) |
| prompt           | String format, indicating the prompt sent to the model |
| completion       | String format, indicating the model's generated result |

## Supported Evaluators

### **RAG Evaluators**
Supports four different evaluation metrics related to RAG. For information on the RAG evaluation system, refer to [Ragas's introduction documentation](https://docs.ragas.io/en/stable/concepts/metrics/index.html). We have implemented four types of RAG evaluators in PolyEval and supported multilingual prompt templates. The scores range from 0 to 1.

### **Large Model Evaluations Based on YAML**
A large model evaluation can be performed by passing a custom YAML configuration file. The advantage of this evaluation method is that evaluation templates can be customized according to specific evaluation needs, for more flexible evaluations. The format requirements for YAML configuration files can be found at [https://docs.evalsone.com/Faq/Metrics/setting_custom_rating_metrics/]. The result can be in the form of ratings, scores, or determinations, depending on the YAML configuration file.

### **Text Similarity Comparisons**
Supports multiple text similarity comparison methods, specifically:
- Based on string distance (implemented by calling LangChain's evaluator), the score range depends on the specific comparison algorithm.
- Based on embedding distance (implemented by calling LangChain's evaluator), the score range depends on the specific comparison algorithm.
- Based on large models for semantic comparison, scores range from 0 to 1.
    
### **JSON Evaluations**
Supports multiple JSON-related evaluation methods, specifically:
- JSON format validation
- JSON Matchness
- JSON Schema Validate
The scores range from 0 to 1. 0 means fail, 1 means pass.

### **Text Matching**
Supports multiple text matching methods, specifically `match`, `include`, `startswith`, `endswith`, `fuzzy_match`, and optionally ignoring case in comparisons. The scores range from 0 to 1. 0 means fail, 1 means pass.

### **Regular Expression Evaluations**
Custom regular expression evaluations. The scores range from 0 to 1. 0 means fail, 1 means pass.

## Supported Evaluator List

| Evaluator Name           | Category       | Description                          | Implementation Mechanism  | Required kwargs Parameters        | Required Data Components          |
|--------------------------|----------------|--------------------------------------|---------------------------|-----------------------------------|------------------------------------|
| [Faithfulness](/polyeval/faithfulness.py)             | RAG Evaluation | Evaluates the accuracy of the generated results | Based on large model prompts  | `llm`: dictionary of model call parameters | `answer` `context`                |
| [AnswerRelevancy](/polyeval/answer_relevancy.py)          | RAG Evaluation | Evaluates the relevancy of the generated answer | Based on large model prompts  | `llm`: dictionary of model call parameters | `question` `answer`               |
| [ContextRecall](/polyeval/context_recall.py)            | RAG Evaluation | Evaluates context recall rate        | Based on large model prompts  | `llm`: dictionary of model call parameters | `context` `ideal`                 |
| [ContextPrecision](/polyeval/context_precision.py)         | RAG Evaluation | Evaluates context precision          | Based on large model prompts  | `llm`: dictionary of model call parameters | `question` `context`              |
| [CustomYaml](/polyeval/custom_yaml.py)               | Large Model Evaluation | Evaluates based on custom YAML configuration | Based on large model prompts  | `yaml_specs`: YAML format evaluation configuration file <br/> `llm`: dictionary of model call parameters | `question` `answer` `context` `ideal` |
| [StringDistanceEvaluator](/polyeval/string_distance.py)  | Text Similarity            | Compares the distance between strings     | Based on comparison algorithms | `metric`: text comparison algorithm in string format | `answer` `ideal`                  |
| [EmbeddingDistanceEvaluator](/polyeval/embedding_distance.py) | Text Similarity         | Computes similarity based on embedding distance | Based on embedding model and comparison algorithms | `metric`: vector comparison algorithm in string format <br/> `embedding`: dictionary of model call parameters | `answer` `ideal`                  |
| [LLMSimilarity](/polyeval/llm_similarity.py)            | Text Similarity            | Uses large model for semantic comparison  | Based on large model prompts  | `llm`: dictionary of model call parameters | `answer` `ideal`                  |
| [IsJson](/polyeval/json.py)                   | JSON Evaluation             | Validates JSON format correctness         | Based on rules             |                                    | `answer`                           |
| [JsonMatch](/polyeval/json.py)                | JSON Evaluation             | Evaluates JSON data consistency           | Based on rules             |                                    | `answer` `ideal`                  |
| [JsonSchemaMatch](/polyeval/json.py)          | JSON Evaluation             | Validates JSON conformity to predefined schema | Based on rules             | `schema`: predefined JSON schema in JSON format | `answer`                           |
| [Matchness](/polyeval/matchness.py)                | Text Matching               | Compares generated results with ideal answers using various methods | Based on rules             | `match_rule`: matching rule <br/> `ignore_case`: whether to ignore case | `answer` `ideal`                  |
| [Regex](/polyeval/regex.py)                    | Regular Expression Evaluation | Custom evaluations using regular expressions | Based on rules             | `regex`: regular expression for validation | `answer` `ideal`                  |

## Multilingual Support

While some evaluation frameworks offer multilingual support, PolyEval aims to provide more comprehensive multilingual support, from evaluation prompt templates to calculation rules. Currently, the `lang` parameter used when calling the evaluator can be `en` for English or `zh` for Simplified Chinese.

```python
# Evaluate faithfulness using English prompt template
eval_results = evaluate(dataset, [faithfulness], "en", **kwargs)

# Evaluate faithfulness using Simplified Chinese prompt template
eval_results = evaluate(dataset, [faithfulness], "zh", **kwargs)
```

In the future, we hope to gradually add support for more languages with the help and support of the community.

Here is an example of a large model rating YAML configuration:

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

The format of the YAML file configuration can be referenced at [https://docs.evalsone.com/Faq/Metrics/setting_custom_rating_metrics/].

## Model Integration

PolyEval can freely integrate hundreds of large language models for evaluation through UnionLLM, and it also uses LangChain for embedding models.

For example, if you need to use OpenAI's gpt-4o model when calling the Faithfulness evaluator, you can configure it as follows:

```python
kwargs = {}
kwargs[`llm`] = {`provider`: `openai`, `model`: `gpt-4o`, `temperature`: 0.0}
eval_results = evaluate(dataset, [faithfulness], "en", **kwargs)
```

If you switch to a model provided by Anthropic, you only need to update the dictionary to:

```python
kwargs = {}
kwargs[`llm`] = {`provider`: `anthropic`, `model`: `claude-3-orpus`, `temperature`: 0.0}
eval_results = evaluate(dataset, [faithfulness], "en", **kwargs)
```

The method for calling various large models through UnionLLM can be referred to:
https://github.com/EvalsOne/UnionLLM.

If you need to use an embedding model in the evaluation, you can configure it as follows:

```python
kwargs = {}
kwargs[`embedding`] = {`provider_cls`: `OpenAIEmbeddings`, `model`: `text-embedding-3-large`}
```

At present, `provider_cls` temporarily only supports OpenAIEmbeddings, CohereEmbeddings, and BaichuanTextEmbeddings. Future versions will expand to support more embedding models.

## Unit Testing

PolyEval offers a series of unit testing scripts that can be tested in batch as follows. You need to install pytest before testing, and set the necessary large model environment variables such as API Key in the script.

```bash
pytest tests
```

Alternatively, you can run a single test (ignoring warnings) as follows:

```bash
pytest tests/test_faithfulness.py -W ignore
```

## Using EvalsOne Cloud Evaluation

[EvalsOne](https://evalsone.com) is a comprehensive solution for evaluating LLM systems. It not only provides a cloud-based evaluation environment where all evaluators in PolyEval can run, but also offers convenient sample management, model calls, visualization of results, and functionality such as iterating evaluations through Forks. [Learn more](https://docs.evalsone.com)

## Contributing to the PolyEval Project

PolyEval welcomes community participation and contributions. Users can participate in the development of the project by submitting code, improving documentation, providing multilingual prompt templates, or offering new evaluator algorithms. Active community involvement will help continuously enhance the functionality and performance of PolyEval, driving the development of large language model evaluation technology.

## ❤️ Acknowledgements

During the development of this project, we were inspired by Ragas, OpenAI Evals, and referred to their implementation code and evaluation configurations. We express our deep gratitude here.
```

If there are any additional sections in need of translation, please let me know!
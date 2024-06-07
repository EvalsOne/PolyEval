# test_json_schema_match.py
import pytest
import os, sys, json
from datasets import Dataset
from jsonschema import Draft7Validator
from zeval.evaluation import evaluate
from zeval.evaluators import JsonSchemaMatch

# 创建一个测试用的schema
schema = {
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "description": {
      "type": "string"
    },
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city and state e.g. San Francisco, CA"
        },
        "unit": {
          "type": "string",
          "enum": [
            "c",
            "f"
          ]
        }
      },
      "required": [
        "location"
      ]
    }
  },
  "required": [
    "name",
    "description",
    "parameters"
  ]
}



# 创建一个Dataset实例
dataset = Dataset.from_dict({
    'question': ["Is this a valid JSON according to the schema?", "Is this a valid JSON according to the schema?", "Is this a valid JSON according to the schema?"],
    'answer': [
        '{ "name": "get_weather", "description": "Determine weather in my location", "parameters": { "locion": "", "unit": "f" } }',
        '{"name": "Weather Report", "description": "Daily weather updates", "parameters": {"location": "San Francisco, CA", "unit": "c"}}', 
        '{"name": "Weather Report", "description": "Daily weather updates", "parameters": {"unit": "c"}}']
})

@pytest.mark.parametrize("evaluators, lang, schema", [
    ([JsonSchemaMatch], "en", {"schema": schema}),
])
def test_json_schema_match(evaluators, lang, schema):
    eval_results = evaluate(dataset, evaluators, schema, lang)
    
    # format the results
    formatted_results = json.dumps(eval_results, indent=4, ensure_ascii=False)
    print(formatted_results)

if __name__ == "__main__":
    pytest.main()

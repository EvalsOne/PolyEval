import pytest
import os, sys
from datasets import Dataset
from polyeval.evaluation import evaluate
from polyeval.evaluators import LLMSimilarity
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

query = "Where is France and what is its capital?"
ideal = ["France is in Western Europe and its capital is Paris."]
messages = [{"role": "user", "content": query}]
completion = "France is located in Western Europe and its capital is Paris. Paris is situated in the north-central part of the country, along the Seine River."
context = ['Paris, city and capital of France, situated in the north-central part of the country. People were living on the site of the present-day city, located along the Seine River some 233 miles (375 km) upstream from the river’s mouth on the English Channel (La Manche), by about 7600 BCE. The modern city has spread from the island (the Île de la Cité) and far beyond both banks of the Seine.']

dataset = Dataset.from_dict({
    'question': [query],
    'answer': [completion], 
    'context': [context],
    'ideal': [ideal]
})

eval_params = {
    "provider": "zhipuai",
    "model": "glm-4",
    "temperature": 0.1,
    "stream": False
}

kwargs = {}
kwargs["llm"] = eval_params

@pytest.mark.parametrize("evaluators, lang, kwargs", [
    ([LLMSimilarity], "zh", kwargs)
])

def test_llm_similarity(evaluators, lang, kwargs):
    eval_results = evaluate(dataset, evaluators, lang, **kwargs)
    
    print(eval_results)
    assert isinstance(eval_results, list)
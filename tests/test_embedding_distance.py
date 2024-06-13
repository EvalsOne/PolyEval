import pytest
import os, sys, json
from datasets import Dataset
from polyeval.evaluation import evaluate
from polyeval.evaluators import EmbeddingDistanceEvaluator
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dataset = Dataset.from_dict({
    'answer': ["The job is completely done.", "The job is done."],
    'ideal': ["The job is done", "The job isn't done"], 
})

os.environ["OPENAI_API_KEY"]='your_openai_api_key_here'
embedding_params = {"provider_cls": "BaichuanTextEmbeddings", "model": "Baichuan-Text-Embedding"}

@pytest.mark.parametrize("evaluators, lang, kwargs", [
    ([EmbeddingDistanceEvaluator], "zh", {"metric": "cosine", "embedding": embedding_params}),
    ([EmbeddingDistanceEvaluator], "en", {"metric": "euclidean", "embedding": embedding_params}),
    ([EmbeddingDistanceEvaluator], "zh", {"metric": "manhattan", "embedding": embedding_params}),
    ([EmbeddingDistanceEvaluator], "zh", {"metric": "chebyshev", "embedding": embedding_params}),
    ([EmbeddingDistanceEvaluator], "zh", {"metric": "hamming", "embedding": embedding_params}),
])
def test_embedding_distance(evaluators, lang, kwargs):
    eval_results = evaluate(dataset, evaluators, lang, **kwargs)
    # print eval results
    formatted_results = json.dumps(eval_results, indent=4, ensure_ascii=False)
    print(formatted_results)
    assert isinstance(eval_results, list)
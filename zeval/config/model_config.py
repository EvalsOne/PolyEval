# zeval/zeval/config/model_config.py
class ModelConfig:
    def __init__(self, provider, **model_kwargs):
        self.provider = provider
        self.model_kwargs = model_kwargs

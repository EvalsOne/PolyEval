# zeval/zeval/utils/model_caller.py
import litellm
from litellm import completion
from unionllm import UnionLLM, unionchat
import time

class ModelCaller:
    def __init__(self, config):
        self.config = config

    def call(self, **params):
        # merge the parameters
        effective_params = {**params}
        start_time = time.time()

        try:
            # self.client = UnionLLM(**params)
            # response = self.client.completion(**effective_params)
            response = unionchat(**effective_params)
        except:
            try:
                # remove the provider key from the parameters for litellm
                effective_params.pop('provider', None)
                response = completion(**effective_params)
            except Exception as e:
                return False, str(e)
          
        end_time = time.time()
        elapsed_time = end_time - start_time
        return response, elapsed_time
            
    def parse_response(self, response):
        try:
            # get the final statement from the model response
            choices = response.choices
            final_statement = choices[0].message.content 
            return final_statement
        except Exception as e:
            return False
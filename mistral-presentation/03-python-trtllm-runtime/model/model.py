from utils import download_engine
from tensorrt_llm import ModelRunner
from logits_processor import logits_post_processor
from pathlib import Path

class Model:
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.secrets = kwargs["secrets"]
        self.data_dir = kwargs["data_dir"]
        self.model = None

    def load(self):
        self.engine_repository = self.config["model_metadata"]["engine_repository"]
        download_engine(
            engine_repository=self.engine_repository,
            fp=self.data_dir,
            
        )
        self.model = ModelRunner.from_dir(self.data_dir, rank=0, debug_mode=False)
    
    def predict(self, request):
        prompt = request["prompt"]
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, end_id=self.tokenizer.eos_token_id, pad_id=self.tokenizer.eos_token_id, logits_processor=logits_post_processor)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
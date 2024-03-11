import torch
from transformers import AutoTokenizer
from pathlib import Path
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo
from typing import List, Tuple
from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData
from pydantic import BaseModel

class AnswerFormat(BaseModel):
        last_name: str
        year_of_birth: int

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
parser = JsonSchemaParser(AnswerFormat.model_json_schema())

def _build_regular_tokens_list(tokenizer) -> List[Tuple[int, str, bool]]:
    token_0 = [tokenizer.encode("0")[-1]]
    regular_tokens = []
    vocab_size = tokenizer.vocab_size
    for token_idx in range(vocab_size):
        if token_idx in tokenizer.all_special_ids:
            continue
        # We prepend token 0 and skip the first letter of the result to get a space if the token is a start word.
        tensor_after_0 = torch.tensor(token_0 + [token_idx], dtype=torch.long)
        decoded_after_0 = tokenizer.decode(tensor_after_0)[1:]
        decoded_regular = tokenizer.decode(token_0)
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append(
            (token_idx, decoded_after_0, is_word_start_token))
    return regular_tokens

def build_token_enforcer(tokenizer, character_level_parser):
    regular_tokens = _build_regular_tokens_list(tokenizer)

    def _decode(tokens: List[int]) -> str:
        tensor = torch.tensor(tokens, dtype=torch.long)
        return tokenizer.decode(tensor)

    tokenizer_data = TokenEnforcerTokenizerData(regular_tokens, _decode,
                                                tokenizer.eos_token_id)
    return TokenEnforcer(tokenizer_data, character_level_parser)

token_enforcer = build_token_enforcer(tokenizer, parser)

def logits_post_processor(req_id: int, logits: torch.Tensor, ids: List[List[int]]):
    del req_id

    def _trim(ids):
        return [x for x in ids if x != tokenizer.eos_token_id]

    allowed = token_enforcer.get_allowed_tokens(_trim(ids[0]))
    allowed_tensor = torch.tensor(allowed, device=logits.device).unsqueeze(0).unsqueeze(0)
    
    mask = torch.ones_like(logits, device=logits.device) * (-1e9)  # Fill the mask with large negative values
    mask.scatter_(2, allowed_tensor, 0)  # Set allowed indices to 0
    mask = mask.to(logits.device) 
    logits += mask  # Apply the mask
    
    return logits

"""
engine_directory = Path("./")

model = ModelRunner.from_dir(engine_directory, rank=0, debug_mode=False)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
parser = JsonSchemaParser(AnswerFormat.model_json_schema())
token_enforcer = build_token_enforcer(tokenizer, parser)

input_text = "Translate the following English text to French: 'Hello, how are you?'"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, end_id=tokenizer.eos_token_id, pad_id=tokenizer.eos_token_id, logits_processor=logits_post_processor)

print(tokenizer.decode(output[0], skip_special_tokens=True))
"""
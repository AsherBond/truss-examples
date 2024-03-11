from pathlib import Path
from builder.types import TRTLLMModelArchitecture
TENSORRT_LLM_GIT_REPO_PATH = Path("/app/TensorRT-LLM/")

DEFAULT_PLUGIN_CONFIG = {
    "gemm_plugin": "float16",
    "gpt_attention_plugin": "float16",
    "context_fmha": True,
    "remove_input_padding": True,
}

ENGINE_DIRECTORY_PATH = Path("/app/engines/")
ENGINE_DIRECTORY_PATH.mkdir(exist_ok=True)

# The mistral architecture, for example, is considered a llama architecture
# but it's confusing for users to specify "llama" as the architecture when
# the model architecture is actually mistral. This mapping allows us to
# convert the user-specified architecture to the correct one when searching
# for the `convert_checkpoint` script.
ARCHITECTURE_TO_CONVERT_CHECKPOINT_ARCHITECTURE = {
    TRTLLMModelArchitecture.LLAMA: TRTLLMModelArchitecture.LLAMA,
    TRTLLMModelArchitecture.MISTRAL: TRTLLMModelArchitecture.LLAMA,
    TRTLLMModelArchitecture.DEEPSEEK: TRTLLMModelArchitecture.LLAMA,
}
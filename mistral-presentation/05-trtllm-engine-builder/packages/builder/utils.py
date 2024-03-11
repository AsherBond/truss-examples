import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from builder.constants import TENSORRT_LLM_GIT_REPO_PATH, ARCHITECTURE_TO_CONVERT_CHECKPOINT_ARCHITECTURE
from builder.types import TRTLLMModelArchitecture
from huggingface_hub import snapshot_download


def download_huggingface_checkpoint(
    repo: str, hf_auth_token: Optional[str], dst: Optional[Path] = None
) -> Path:
    huggingface_checkpoint_dir = tempfile.mkdtemp()
    snapshot_download(
        repo,
        local_dir=huggingface_checkpoint_dir,
        local_dir_use_symlinks=False,
        max_workers=4,
        **({"use_auth_token": hf_auth_token} if hf_auth_token is not None else {}),
    )
    return Path(huggingface_checkpoint_dir)


def execute_command(command) -> None:
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Standard Output:\n", process.stdout)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The command '{command[0]}' is not found. Make sure it is installed and in your PATH."
        )

def generate_convert_checkpoint_filepath(
    base_model_architecture: TRTLLMModelArchitecture,
) -> Path:
    return (
        TENSORRT_LLM_GIT_REPO_PATH
        / "examples"
        / ARCHITECTURE_TO_CONVERT_CHECKPOINT_ARCHITECTURE[base_model_architecture].value
        / "convert_checkpoint.py"
    )

import argparse
import logging

import yaml
from typing import Optional
from pathlib import Path
from builder.constants import ENGINE_DIRECTORY_PATH
from builder.models.llama import LlamaBuilder
from builder.types import TRTLLMModelArchitecture, TrussTRTLLMConfiguration
from builder.utils import execute_command

ARCHITECTURE_TO_BUILDER = {
    TRTLLMModelArchitecture.LLAMA: LlamaBuilder,
    TRTLLMModelArchitecture.MISTRAL: LlamaBuilder,
    TRTLLMModelArchitecture.DEEPSEEK: LlamaBuilder,
}

def build_engine(engine_configuration: TrussTRTLLMConfiguration, checkpoint_dir_path: Optional[Path] = None):
    # TODO(Abu): Investigate why this is necessary. 
    # Something about how we install mpi causes this to be necessary.
    execute_command(["ldconfig"])
    model_architecture = (
        engine_configuration.build.base_model_architecture
    )
    if model_architecture not in ARCHITECTURE_TO_BUILDER:
        raise ValueError(f"Model architecture {model_architecture} not supported.")
    builder = ARCHITECTURE_TO_BUILDER[model_architecture](engine_configuration)
    builder.build(
        checkpoint_dir_path=checkpoint_dir_path,
        engine_serialization_path=ENGINE_DIRECTORY_PATH
    )

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build engine utility v2.")
    parser.add_argument(
        "--config",
        type=str,
        help="Engine config to build from.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        help="Directory containing the checkpoint to build from.",)

    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as file:
        engine_configuration = TrussTRTLLMConfiguration(**yaml.safe_load(file))
        build_engine(engine_configuration, args.checkpoint_dir)

if __name__ == "__main__":
    main()

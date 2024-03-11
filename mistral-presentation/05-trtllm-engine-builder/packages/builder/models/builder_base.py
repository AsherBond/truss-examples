from abc import ABC, abstractmethod
from pathlib import Path

import torch
from builder.constants import DEFAULT_PLUGIN_CONFIG, ENGINE_DIRECTORY_PATH
from builder.errors import EngineBuildError
from builder.types import (
    TrussTRTLLMBuildConfiguration,
    TrussTRTLLMConfiguration,
    TrussTRTLLMPluginConfiguration,
)
from builder.utils import download_huggingface_checkpoint
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.commands.build import parallel_build
from tensorrt_llm.plugin import PluginConfig
from typing import Optional

class GenericModelBuilder(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def build(self, args) -> Path:
        pass

    @abstractmethod
    def convert_checkpoint(self, args) -> Path:
        pass


class TRTLLMEngineBuilder(GenericModelBuilder):
    def __init__(self, config: TrussTRTLLMConfiguration):
        super().__init__(config)

    def prepare_plugin_config(
        self, user_plugin_config: TrussTRTLLMPluginConfiguration
    ) -> PluginConfig:
        plugin_config = PluginConfig(**DEFAULT_PLUGIN_CONFIG)
        plugin_config.update_from_dict(user_plugin_config.dict())
        return plugin_config

    def prepare_build_config(
        self,
        user_build_config: TrussTRTLLMBuildConfiguration,
        plugin_config: PluginConfig,
    ) -> BuildConfig:
        build_config = BuildConfig(
            max_input_len=user_build_config.max_input_len,
            max_output_len=user_build_config.max_output_len,
            max_batch_size=user_build_config.max_batch_size,
            max_beam_width=user_build_config.max_beam_width,
            max_prompt_embedding_table_size=user_build_config.max_prompt_embedding_table_size,
            gather_context_logits=(
                True if user_build_config.gather_all_token_logits else False
            ),
            gather_generation_logits=(
                True if user_build_config.gather_all_token_logits else False
            ),
            strongly_typed=user_build_config.strongly_typed,
            plugin_config=plugin_config,
        )
        return build_config

    def build(self, checkpoint_dir_path: Optional[Path] = None, engine_serialization_path: Path = ENGINE_DIRECTORY_PATH) -> Path:
        if not checkpoint_dir_path:
            if not self.config.build.huggingface_ckpt_repository:
                raise ValueError(
                    "Either a checkpoint directory or a Huggingface checkpoint repository must be provided."
                )
            checkpoint_dir_path = download_huggingface_checkpoint(
                self.config.build.huggingface_ckpt_repository,  # type: ignore
                hf_auth_token=None,
            )

        trtllm_checkpoint_dir = self.convert_checkpoint(checkpoint_dir_path)

        plugin_configuration = self.prepare_plugin_config(
            self.config.build.plugin_configuration
        )
        build_configuration = self.prepare_build_config(self.config.build, plugin_configuration)  # type: ignore

        workers = torch.cuda.device_count()
        kwargs = {
            "logits_dtype": "float16",
            "use_fused_mlp": self.config.build.plugin_configuration.use_fused_mlp,
            "weight_only_precision": None,
        }
        try:
            parallel_build(
                str(trtllm_checkpoint_dir),
                build_configuration,
                engine_serialization_path,
                workers,
                "info",
                None,
                **kwargs,
            )
        except Exception as e:
            raise EngineBuildError(e)
        return engine_serialization_path

    @abstractmethod
    def convert_checkpoint(self, huggingface_checkpoint_dir: Path) -> Path:
        pass

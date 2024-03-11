import tempfile
from pathlib import Path

from builder.errors import CheckpointConversionError
from builder.models.builder_base import TRTLLMEngineBuilder
from builder.types import TRTLLMQuantizationType, TrussTRTLLMConfiguration
from builder.utils import execute_command, generate_convert_checkpoint_filepath


class LlamaBuilder(TRTLLMEngineBuilder):
    def __init__(self, engine_configuration: TrussTRTLLMConfiguration):
        super().__init__(engine_configuration)

    def convert_checkpoint(self, huggingface_checkpoint_dir: Path):
        serialization_dir = tempfile.mkdtemp()
        build_config = self.config.build
        quant_type = build_config.quantization_type  # type: ignore
        tp = build_config.tensor_parallel_count
        pp = build_config.pipeline_parallel_count
        base_args = [
            f"--tp_size={tp}",
            f"--pp_size={pp}",
            f"--model_dir={str(huggingface_checkpoint_dir)}",
            f"--output_dir={serialization_dir}",
        ]

        if quant_type == TRTLLMQuantizationType.NO_QUANT.value:
            # do nothing
            base_args = base_args
        elif quant_type == TRTLLMQuantizationType.WEIGHTS_ONLY_INT8.value:
            base_args = base_args + [
                "--use_weight_only",
                "--weight_only_precision",
                "int8",
            ]
        elif quant_type == TRTLLMQuantizationType.WEIGHTS_KV_INT8.value:
            base_args = base_args + [
                "--use_weight_only",
                "--weight_only_precision",
                "int8",
                "--int8-kv_cache",
            ]
        elif quant_type == TRTLLMQuantizationType.WEIGHTS_ONLY_INT4.value:
            base_args = base_args + [
                "--use_weight_only",
                "--weight_only_precision",
                "int4",
            ]
        elif quant_type == TRTLLMQuantizationType.WEIGHTS_KV_INT4.value:
            base_args = base_args + [
                "--use_weight_only",
                "--weight_only_precision",
                "int4",
                "--int8-kv_cache",
            ]
        elif quant_type == TRTLLMQuantizationType.SMOOTH_QUANT.value:
            base_args = base_args + [
                "--smoothquant",
                "0.5",
                "--int8_kv_cache",
                "--per_token",
                "--per_channel",
            ]
        elif quant_type == TRTLLMQuantizationType.FP8.value:
            base_args = base_args + ["--enable_fp8"]
        elif quant_type == TRTLLMQuantizationType.FP8_KV.value:
            base_args = base_args + ["--enable_fp8", "--fp8_kv_cache"]

        path_to_convert_checkpoint_file = generate_convert_checkpoint_filepath(self.config.build.base_model_architecture)  # type: ignore
        command = ["python3", str(path_to_convert_checkpoint_file)] + base_args
        try:
            execute_command(command)
        except Exception as e:
            raise CheckpointConversionError(e)
        return serialization_dir

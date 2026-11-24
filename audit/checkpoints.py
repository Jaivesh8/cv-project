import json
import os
import struct
import sys


def load_checkpoint(model_id, revision, dtype, offline):
    from transformers import (
        AutoConfig, AutoModelForImageTextToText, GenerationConfig,
        Idefics3ForConditionalGeneration, SmolVLMForConditionalGeneration,
    )

    options = dict(torch_dtype=dtype, attn_implementation="sdpa", local_files_only=offline, token=False)
    if sys.platform != "win32":
        return AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, trust_remote_code=False, **options
        )

    # The PyTorch-backed Safetensors mmap path can fault for these larger files on Windows.
    # Load the same tensors by offset and let Transformers handle key matching and tied weights.
    from huggingface_hub import hf_hub_download

    config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=offline, token=False)
    filename = hf_hub_download(model_id, "model.safetensors", revision=revision,
                               local_files_only=offline, token=False)
    weights = read_weights(filename, dtype)
    model_type = {"idefics3": Idefics3ForConditionalGeneration, "smolvlm": SmolVLMForConditionalGeneration}
    model, info = model_type[config.model_type].from_pretrained(
        None, config=config, state_dict=weights, output_loading_info=True, **options
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint did not load exactly: {info}")
    model.generation_config = GenerationConfig.from_pretrained(
        model_id, revision=revision, local_files_only=offline, token=False
    )
    return model


def read_weights(filename, destination_dtype):
    import torch

    types = dict(BF16=torch.bfloat16, F16=torch.float16, F32=torch.float32,
                 I64=torch.int64, I32=torch.int32, BOOL=torch.bool)
    weights = {}
    with open(filename, "rb") as stream:
        header_len, = struct.unpack("<Q", stream.read(8))
        if not 0 < header_len <= 100_000_000:
            raise ValueError("Invalid Safetensors header length")
        header = json.loads(stream.read(header_len))
        start = stream.tell()
        length = os.fstat(stream.fileno()).st_size - start
        header.pop("__metadata__", None)
        for key, info in header.items():
            low, high = info["data_offsets"]
            if not 0 <= low < high <= length:
                raise ValueError(f"Invalid tensor offsets: {key}")
            data = bytearray(high - low)
            stream.seek(start + low)
            if stream.readinto(data) != len(data):
                raise ValueError(f"Incomplete tensor buffer: {key}")
            source = torch.frombuffer(data, dtype=types[info["dtype"]]).reshape(info["shape"])
            weights[key] = source.to(destination_dtype) if source.is_floating_point() else source
    return weights

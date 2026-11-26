"""Explicit QLoRA placement shared by training and evaluation.

CPU embedding/output projections leave room for 4-bit decoder layers and their
adapter gradients. This is fixed single-process model placement, not inference
weight staging or FSDP. Whether a particular model/run fits still needs measuring.
"""
from __future__ import annotations


def _configure_fixed_cpu_io(model, *, decoder_dtype):
    """Keep Qwen2 vocabulary weights/computation on CPU, with autograd transfers.

    Accelerate's mixed CPU/CUDA device map is used only while loading shards. Its
    offload hooks otherwise replace CPU weights with meta tensors and stage them
    to CUDA for every forward. Detaching restores those weights on CPU. Ordinary
    PyTorch forward hooks below transfer activations, never vocabulary weights.
    """
    import torch
    from accelerate.hooks import remove_hook_from_module

    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError("Fixed CPU vocabulary placement currently supports Qwen2 only")
    if getattr(model, "_kernelbench_fixed_cpu_io", False):
        raise ValueError("Fixed CPU vocabulary placement has already been installed")
    remove_hook_from_module(model, recurse=True)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("CPU weights were not materialized when removing inference offload hooks")
    decoder_device = next(model.model.layers.parameters()).device
    if decoder_device.type != "cuda":
        raise RuntimeError("Fixed CPU vocabulary placement requires CUDA decoder layers")
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    if embedding.weight.device.type != "cpu" or head.weight.device.type != "cpu":
        raise RuntimeError("Vocabulary weights must remain on CPU after loading")
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
    model.model.rotary_emb.to(decoder_device)

    def embedding_input_cpu(module, args):
        return (args[0].to("cpu"), *args[1:])

    def embedding_output_cuda(module, args, output):
        return output.to(device=decoder_device, dtype=decoder_dtype)

    def metadata_cuda(module, args, kwargs):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(decoder_device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            return value
        # Input IDs stay on CPU for the embedding. Positional/mask tensors must
        # follow decoder activations, including generate()'s cache_position.
        return args, {key: value if key == "input_ids" else move(value) for key, value in kwargs.items()}

    def head_input_cpu(module, args):
        # .to() preserves the CPU->CUDA backward edge to decoder adapters. The
        # frozen fp32 CPU weight is never copied to the GPU, including backward.
        return (args[0].to(device="cpu", dtype=module.weight.dtype), *args[1:])

    embedding.register_forward_pre_hook(embedding_input_cpu)
    embedding.register_forward_hook(embedding_output_cuda)
    model.model.register_forward_pre_hook(metadata_cuda, with_kwargs=True)
    head.register_forward_pre_hook(head_input_cpu)
    model._kernelbench_fixed_cpu_io = True
    model._kernelbench_device_placement = {
        "model.embed_tokens": "cpu", "model.layers": str(decoder_device),
        "model.norm": str(decoder_device), "model.rotary_emb": str(decoder_device),
        "lm_head": "cpu", "logits": "cpu", "weight_staging": False,
    }
    # PEFT uses hf_device_map CPU/disk entries to attach inference dispatch hooks
    # again on adapter reload. Placement is now owned by ordinary PyTorch hooks;
    # retaining that stale dispatch hint would silently undo fixed CPU execution.
    if hasattr(model, "hf_device_map"):
        delattr(model, "hf_device_map")
    if any(hasattr(module, "_hf_hook") for module in model.modules()):
        raise RuntimeError("Unexpected inference dispatch hook remains installed")
    return model


def load_quantized_base(cfg, *, gradient_checkpointing: bool):
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.dtype]
    cpu_offload = cfg.qlora_cpu_offload
    device_map = ({"model.embed_tokens": "cpu", "model.layers": 0,
                   "model.norm": 0, "lm_head": "cpu"} if cpu_offload else {"": 0})
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype, llm_int8_enable_fp32_cpu_offload=cpu_offload,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, revision=cfg.model_revision,
        quantization_config=quantization, device_map=device_map,
        torch_dtype=dtype, local_files_only=True, trust_remote_code=False,
    )
    if cpu_offload:
        model = _configure_fixed_cpu_io(model, decoder_dtype=dtype)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if cpu_offload:
        if model.get_input_embeddings().weight.device.type != "cpu" or model.get_output_embeddings().weight.device.type != "cpu":
            raise RuntimeError("Preparation moved fixed CPU vocabulary weights")
        if next(model.model.layers.parameters()).device.type != "cuda":
            raise RuntimeError("Preparation moved CUDA decoder weights")
    return model

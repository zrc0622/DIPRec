"""Optional heavy-dependency runtime helpers used on remote training machines."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .interest import TokenRegistry, register_sid_tokens, register_tokens
from .modeling import InterestParameterRouter


def require_torch_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on remote environment
        raise RuntimeError("Install torch and transformers before running training/evaluation") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool | None = False,
) -> list[int]:
    kwargs = dict(tokenize=True, add_generation_prompt=add_generation_prompt, return_tensors=None)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return [int(value) for value in ids]


def thinking_prompt_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    ids = apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=True)
    tail = tokenizer.decode(ids[-32:], skip_special_tokens=False).rstrip()
    if not tail.endswith("<think>"):
        ids.extend(tokenizer.encode("<think>", add_special_tokens=False))
    return ids


def encode_one(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected one token for {text!r}, got {ids}")
    return int(ids[0])


def load_model_runtime(
    model_name_or_path: str,
    sid_map: Mapping[str, Sequence[str]],
    parameterization: str,
    training: bool,
    include_interest: bool = True,
) -> tuple[Any, Any, TokenRegistry | None, InterestParameterRouter | None]:
    torch, AutoModelForCausalLM, AutoTokenizer = require_torch_transformers()
    source = Path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    registry = register_tokens(tokenizer, model, sid_map) if include_interest else None
    if registry is None:
        register_sid_tokens(tokenizer, model, sid_map)
    router = None
    adapter_config = source / "diprec_adapter_config.json"
    if adapter_config.is_file():
        saved = json.loads(adapter_config.read_text(encoding="utf-8"))
        saved_mode = saved.get("mode", "independent_head")
        if parameterization != saved_mode:
            raise ValueError(
                f"Checkpoint uses interest_parameterization={saved_mode}, requested {parameterization}"
            )
    if include_interest and parameterization == "independent_head":
        assert registry is not None
        routed_ids = [
            registry.interest_begin_id,
            registry.interest_end_id,
            registry.interest_pad_id,
            *registry.interest_token_ids,
        ]
        router = InterestParameterRouter(model, routed_ids)
        if adapter_config.is_file():
            state_path = source / "diprec_interest_adapter.pt"
            if not state_path.is_file():
                raise FileNotFoundError(f"Missing independent interest adapter weights: {state_path}")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            router.adapter.load_state_dict(state)
        router.assert_parameter_isolation(registry.sid_token_ids)
    elif include_interest and parameterization != "disjoint_rows":
        raise ValueError("interest_parameterization must be independent_head or disjoint_rows")
    model.config.use_cache = not training
    return model, tokenizer, registry, router


def save_runtime(model: Any, tokenizer: Any, router: InterestParameterRouter | None, output_dir: str | Path, mode: str) -> None:
    import torch

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    if router is not None:
        state_dict = {
            key: value for key, value in state_dict.items() if not key.startswith("diprec_interest_adapter.")
        }
    model.save_pretrained(destination, state_dict=state_dict, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    if router is not None:
        config = {
            "mode": mode,
            "interest_token_ids": router.adapter.global_ids.detach().cpu().tolist(),
        }
        (destination / "diprec_adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        torch.save(router.adapter.state_dict(), destination / "diprec_interest_adapter.pt")

"""Independent interest embedding/output rows for a Hugging Face causal LM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - remote environment concern
        raise RuntimeError("PyTorch is required for DIPRec model training") from exc
    return torch


class InterestParameterRouter:
    """Route interest vocabulary IDs through independent parameters.

    Input positions are replaced by a dedicated embedding and their logits by
    a dedicated output head. SID and normal vocabulary rows remain in the base
    model, retaining Hugging Face generation compatibility.
    """

    def __init__(self, model: Any, interest_token_ids: Sequence[int]):
        torch = _torch()
        nn = torch.nn
        token_ids = sorted(set(int(value) for value in interest_token_ids))
        if not token_ids:
            raise ValueError("interest_token_ids must not be empty")
        input_embedding = model.get_input_embeddings()
        hidden_size = int(input_embedding.embedding_dim)

        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(len(token_ids), hidden_size)
                self.output_head = nn.Linear(hidden_size, len(token_ids), bias=False)
                self.register_buffer("global_ids", torch.tensor(token_ids, dtype=torch.long), persistent=True)
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)

        if hasattr(model, "diprec_interest_adapter"):
            adapter = model.diprec_interest_adapter
            existing = adapter.global_ids.detach().cpu().tolist()
            if existing != token_ids:
                raise ValueError(f"Attached interest IDs {existing} differ from requested {token_ids}")
        else:
            adapter = Adapter()
            adapter.to(device=input_embedding.weight.device, dtype=input_embedding.weight.dtype)
            model.add_module("diprec_interest_adapter", adapter)

        self.model = model
        self.adapter = adapter
        self.input_embedding = input_embedding
        self.output_embedding = model.get_output_embeddings()
        if self.output_embedding is None:
            raise ValueError("Model has no output embedding/lm_head")
        self._embedding_hook = self.input_embedding.register_forward_hook(self._replace_input_embeddings)
        self._head_hook = self.output_embedding.register_forward_hook(self._replace_interest_logits)

    def _local_indices(self, input_ids: Any) -> tuple[Any, Any]:
        torch = _torch()
        global_ids = self.adapter.global_ids.to(input_ids.device)
        matches = input_ids.unsqueeze(-1).eq(global_ids)
        mask = matches.any(dim=-1)
        local = matches.to(torch.int64).argmax(dim=-1)
        return mask, local

    def _replace_input_embeddings(self, _module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        input_ids = inputs[0]
        mask, local = self._local_indices(input_ids)
        if not mask.any():
            return output
        independent = self.adapter.embedding(local)
        return _torch().where(mask.unsqueeze(-1), independent, output)

    def _replace_interest_logits(self, _module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = inputs[0]
        independent = self.adapter.output_head(hidden)
        result = output.clone()
        result[..., self.adapter.global_ids.to(result.device)] = independent
        return result

    def close(self) -> None:
        self._embedding_hook.remove()
        self._head_hook.remove()

    def assert_parameter_isolation(self, sid_token_ids: Sequence[int]) -> None:
        if set(self.adapter.global_ids.detach().cpu().tolist()) & set(map(int, sid_token_ids)):
            raise AssertionError("Interest adapter routes at least one SID token ID")
        base_parameter_ids = {id(parameter) for parameter in self.input_embedding.parameters()}
        base_parameter_ids.update(id(parameter) for parameter in self.output_embedding.parameters())
        adapter_parameter_ids = {id(parameter) for parameter in self.adapter.parameters()}
        if base_parameter_ids & adapter_parameter_ids:
            raise AssertionError("Interest adapter shares a Parameter object with base SID parameters")


def save_interest_adapter(router: InterestParameterRouter, output_dir: str | Path, mode: str) -> None:
    torch = _torch()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config = {"mode": mode, "interest_token_ids": router.adapter.global_ids.detach().cpu().tolist()}
    (destination / "diprec_adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.save(router.adapter.state_dict(), destination / "diprec_interest_adapter.pt")


def attach_interest_adapter(model: Any, checkpoint_dir: str | Path, strict: bool = True) -> InterestParameterRouter:
    torch = _torch()
    source = Path(checkpoint_dir)
    config = json.loads((source / "diprec_adapter_config.json").read_text(encoding="utf-8"))
    router = InterestParameterRouter(model, config["interest_token_ids"])
    state = torch.load(source / "diprec_interest_adapter.pt", map_location="cpu", weights_only=True)
    router.adapter.load_state_dict(state, strict=strict)
    return router

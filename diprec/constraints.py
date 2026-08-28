"""State constraints for the interest stage and catalog SID stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .data import parse_sid_levels


@dataclass
class SIDTrie:
    children: dict[int, "SIDTrie"] = field(default_factory=dict)
    terminal: bool = False

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[int]]) -> "SIDTrie":
        root = cls()
        count = 0
        for sequence in sequences:
            if len(sequence) != 3:
                raise ValueError(f"Catalog SID paths must have three tokens, got {sequence}")
            node = root
            for token_id in sequence:
                node = node.children.setdefault(int(token_id), cls())
            node.terminal = True
            count += 1
        if not count:
            raise ValueError("Cannot construct an empty SID trie")
        return root

    def allowed(self, prefix: Sequence[int], eos_token_id: int) -> list[int]:
        node = self
        for token_id in prefix:
            child = node.children.get(int(token_id))
            if child is None:
                return [eos_token_id]
            node = child
        if node.terminal:
            return [eos_token_id]
        return sorted(node.children)

    def contains(self, sequence: Sequence[int]) -> bool:
        node = self
        for token_id in sequence:
            child = node.children.get(int(token_id))
            if child is None:
                return False
            node = child
        return node.terminal


def build_sid_trie(tokenizer: Any, sid_map: Mapping[str, Sequence[str]]) -> SIDTrie:
    sequences = []
    for levels in sid_map.values():
        ids = []
        for token in parse_sid_levels(levels):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"SID token {token!r} maps to {encoded}, expected one ID")
            ids.append(int(encoded[0]))
        sequences.append(ids)
    return SIDTrie.from_sequences(sequences)


def sid_prefix_allowed_fn(trie: SIDTrie, prompt_length: int, eos_token_id: int):
    def allowed(_batch_id: int, input_ids: Any) -> list[int]:
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        return trie.allowed(ids[prompt_length:], eos_token_id)

    return allowed


def interest_prefix_allowed_fn(
    interest_ids: Sequence[int],
    pad_id: int,
    end_id: int,
    end_think_ids: Sequence[int],
    prompt_length: int,
    k: int,
    eos_token_id: int,
):
    """Constrain exactly k plan tokens, then INT_END, </think>, EOS."""

    code_ids = sorted(set(int(value) for value in interest_ids))
    pad_id = int(pad_id)
    suffix = [int(end_id), *[int(value) for value in end_think_ids], int(eos_token_id)]

    def allowed(_batch_id: int, input_ids: Any) -> list[int]:
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        generated = ids[prompt_length:]
        if len(generated) < k:
            # Match the SFT label grammar: interest codes are unique and PAD
            # is a suffix, never a gap followed by another code.
            if pad_id in generated:
                return [pad_id]
            used = set(map(int, generated))
            return [token_id for token_id in code_ids if token_id not in used] + [pad_id]
        suffix_position = len(generated) - k
        if suffix_position < len(suffix):
            return [suffix[suffix_position]]
        return [int(eos_token_id)]

    return allowed

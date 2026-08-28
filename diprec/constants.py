"""Shared constants and dataset aliases."""

SCHEMA_VERSION = "diprec.long_history.v1"

DATASET_ALIASES = {
    "Office": "Office_Products",
    "Office_Products": "Office_Products",
    "Games": "Video_Games",
    "Video_Games": "Video_Games",
    "Industrial": "Industrial_and_Scientific",
    "Industrial_and_Scientific": "Industrial_and_Scientific",
}

DATASET_LABELS = {
    "Office_Products": "office products",
    "Video_Games": "video games",
    "Industrial_and_Scientific": "industrial and scientific items",
}

INTEREST_BEGIN = "<INT_BEGIN>"
INTEREST_END = "<INT_END>"
INTEREST_PAD = "<INT_PAD>"


def canonical_dataset(name: str) -> str:
    try:
        return DATASET_ALIASES[name.strip()]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unsupported dataset {name!r}; choose one of: {supported}") from exc

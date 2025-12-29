#!/usr/bin/env python3

# pyre-strict

from typing import Any, Optional, Protocol, Tuple, Type

import torch
from packaging.version import Version


class CacheLike(Protocol):
    """Protocol for cache-like objects."""


class DynamicCacheLike(CacheLike, Protocol):
    """Protocol for dynamic cache-like objects."""

    @classmethod
    def from_legacy_cache(
        cls: Type["DynamicCacheLike"],
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> "DynamicCacheLike": ...


# TODO: this should be removed. It no longer work in latest transformers versions
# DynamicCache no longer even has the function from_legacy_cache
# https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/cache_utils.py#L895
# Should investigate other options to support seqeuntial decoding like log_processor
def convert_legacy_cache_if_needed(past_key_values: Any) -> CacheLike:
    """
    If applicable, convert past_key_values to DynamicCache for transformers models.

    Minimum required transformers version is 4.43.0:
    - Cache and DynamicCache are available in transformers.cache_utils
    - GenerationMixin._update_model_kwargs_for_generation requires "cache_position"
      as a mandatory argument (introduced in v4.39.0, made mandatory in v4.43.0)
    - "use_cache" argument is supported (added in v4.41.0, optional w/ default True)

    Older versions lack these cache utilities or have inconsistent APIs, making them
    incompatible with our caching logic.
    """
    min_version = Version("4.43.0")
    try:
        import transformers  # noqa: F401  # type: ignore
        from transformers.cache_utils import DynamicCache  # noqa: F401  # type: ignore

        transformers_version = Version(transformers.__version__)
        assert transformers_version >= min_version, (
            f"transformers version {transformers.__version__} is not supported. "
            f"Please upgrade to version {min_version} or higher."
        )
    except ImportError:
        return past_key_values

    # rough check if past_key_values is Tuple[Tuple[torch.Tensor]]
    if isinstance(past_key_values, tuple):
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values

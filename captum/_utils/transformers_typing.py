#!/usr/bin/env python3

# pyre-strict

from typing import Any, cast, Dict, Optional, Protocol, Tuple, Type

import torch

from packaging.version import Version
from torch import nn


class CacheLike(Protocol):
    """Protocol for cache-like objects."""


class DynamicCacheLike(CacheLike, Protocol):
    """Protocol for dynamic cache-like objects."""

    @classmethod
    def from_legacy_cache(
        cls: Type["DynamicCacheLike"],
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> "DynamicCacheLike": ...


# Minimum required transformers version.
# Starting from v4.43.0:
# - Cache and DynamicCache are available in transformers.cache_utils
# - GenerationMixin._update_model_kwargs_for_generation requires "cache_position"
#   as a mandatory argument (introduced in v4.39.0, made mandatory in v4.43.0)
# - "use_cache" argument is supported (introduced in v4.41.0, optional w/ default True)
# Older versions lack these cache utilities or have inconsistent APIs, making them
# incompatible with our caching logic.
_MIN_TRANSFORMERS_VERSION = Version("4.43.0")

transformers_installed: bool
_transformers_version: Optional[Version]
Cache: Optional[Type[CacheLike]]
DynamicCache: Optional[Type[DynamicCacheLike]]

try:
    import transformers  # noqa: F401  # type: ignore
    from transformers.cache_utils import (  # noqa: F401  # type: ignore
        Cache as _Cache,
        DynamicCache as _DynamicCache,
    )

    _transformers_version = Version(transformers.__version__)
    assert _transformers_version >= _MIN_TRANSFORMERS_VERSION, (
        f"transformers version {transformers.__version__} is not supported. "
        f"Please upgrade to version {_MIN_TRANSFORMERS_VERSION} or higher."
    )
    transformers_installed = True
    Cache = _Cache
    DynamicCache = cast(Optional[Type[DynamicCacheLike]], _DynamicCache)
except ImportError:
    transformers_installed = False
    _transformers_version = None
    Cache = DynamicCache = None


def update_model_kwargs(
    model_kwargs: Dict[str, Any],
    model: nn.Module,
    input_ids: torch.Tensor,
    caching: bool,
) -> None:
    if not supports_caching(model):
        return
    if caching:
        # Enable caching: cache_position and use_cache are both supported in >= 4.43.0
        cache_position = torch.arange(
            input_ids.shape[1], dtype=torch.int64, device=input_ids.device
        )
        model_kwargs["cache_position"] = cache_position
        model_kwargs["use_cache"] = True
    else:
        # Disable caching
        model_kwargs["use_cache"] = False


def supports_caching(model: nn.Module) -> bool:
    if not transformers_installed:
        # Not a transformers model
        return False
    try:
        from transformers.generation.utils import GenerationMixin  # type: ignore
    except ImportError:
        return False
    if not isinstance(model, GenerationMixin):
        # Model isn't a GenerationMixin, we don't support additional caching logic
        # for it
        return False
    # Cache is mandatory for all models in >= 4.43.0
    return True

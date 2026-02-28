from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import smote_variants as sv

from configs.schemas import MethodDefinitionModel


def _build_from_definition(definition: MethodDefinitionModel) -> Any:
    params = dict(definition.params or {})

    if definition.source == 'smote_variants':
        cls = getattr(sv, definition.class_name, None)
        if cls is None:
            raise KeyError(f"Unknown smote_variants class: {definition.class_name}")
        return cls(**params)

    module = importlib.import_module(definition.module)
    cls = getattr(module, definition.class_name, None)
    if cls is None:
        raise KeyError(
            f"Unknown local class '{definition.class_name}' in module '{definition.module}'"
        )
    return cls(**params)


def build_method(
    name: str,
    methods_registry: Dict[str, MethodDefinitionModel],
    override_params: Optional[Dict[str, Any]] = None,
) -> Any:
    if name not in methods_registry:
        raise KeyError(f"Unknown method '{name}'")

    definition = methods_registry[name]
    if override_params:
        definition = definition.model_copy(
            update={'params': {**dict(definition.params or {}), **override_params}}
        )

    return _build_from_definition(definition)


def list_available_methods(methods_registry: Dict[str, MethodDefinitionModel]) -> list[str]:
    return sorted(methods_registry.keys())

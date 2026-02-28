import pytest

pytest.importorskip("smote_variants")

from configs.validation import load_validated_benchmark_bundle
from src.methods.registry import build_method, list_available_methods


def test_registry_builds_known_method():
    bundle = load_validated_benchmark_bundle("experiment/base_experiment.yaml")

    method = build_method("SMOTE", bundle.methods_registry)
    assert method.__class__.__name__ == "SMOTE"


def test_registry_rejects_unknown_method():
    bundle = load_validated_benchmark_bundle("experiment/base_experiment.yaml")

    with pytest.raises(KeyError):
        build_method("NOT_A_METHOD", bundle.methods_registry)


def test_registry_lists_methods():
    bundle = load_validated_benchmark_bundle("experiment/base_experiment.yaml")

    available = list_available_methods(bundle.methods_registry)
    assert "SMOTE" in available
    assert available == sorted(available)

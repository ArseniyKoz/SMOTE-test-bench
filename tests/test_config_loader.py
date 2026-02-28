import pytest

pytest.importorskip("yaml")
pytest.importorskip("pydantic")

from configs.config_loader import ConfigLoader
from configs.validation import load_validated_benchmark_bundle


def test_load_experiment_config_contains_required_keys():
    cfg = ConfigLoader("experiment/base_experiment.yaml").load()

    assert "datasets" in cfg
    assert "methods" in cfg
    assert "experiment_config" in cfg


def test_validated_bundle_has_cross_references_resolved():
    bundle = load_validated_benchmark_bundle("experiment/base_experiment.yaml")

    assert bundle.experiment.datasets
    assert bundle.experiment.methods

    for dataset in bundle.experiment.datasets:
        assert dataset in bundle.datasets_registry

    for method in bundle.experiment.methods:
        assert method in bundle.methods_registry

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("clearml")
pytest.importorskip("smote_variants")

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from experiments import experiment_runner as er


class IdentityResampler:
    def fit_resample(self, x, y):
        return np.asarray(x), np.asarray(y)


def _build_toy_dataset():
    x, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.75, 0.25],
        random_state=42,
    )
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df, {"source": "unit-test"}


def test_run_single_experiment_smoke_without_clearml(monkeypatch, tmp_path):
    monkeypatch.setattr(er, "fetch_dataset", lambda _name, _preprocessed=False: _build_toy_dataset())
    monkeypatch.setattr(
        er.ClassifierPool,
        "get_classifiers",
        lambda self: {
            "LogisticRegression": LogisticRegression(
                random_state=self.random_state,
                max_iter=400,
            )
        },
    )

    cfg = er.ExperimentConfig()
    cfg.cv_folds = 2
    cfg.test_size = 0.25
    cfg.selected_classifiers = ["LogisticRegression"]
    cfg.enable_scatter_plots = False
    cfg.enable_roc_curves = False
    cfg.enable_precision_recall_curves = False
    cfg.results_dir = str(tmp_path)
    cfg.save_results = True

    runner = er.ExperimentRunner(config=cfg, create_clearml_task=False)
    results = runner.run_single_experiment("toy_dataset", IdentityResampler())

    assert "metadata" in results
    assert "dataset_info" in results
    assert "cross_validation_results" in results
    assert "cross_validation_imbalanced_results" in results
    assert "cross_validation_delta_stats" in results
    assert "final_test_results" in results

    assert "LogisticRegression" in results["cross_validation_results"]
    assert "LogisticRegression" in results["cross_validation_imbalanced_results"]
    assert "LogisticRegression" in results["final_test_results"]

    delta_stats = results["cross_validation_delta_stats"]["LogisticRegression"]
    assert "balanced_accuracy" in delta_stats
    assert "positive_delta_rate" in delta_stats["balanced_accuracy"]

    run_dir = tmp_path / runner.run_id
    manifest = run_dir / "manifest.json"
    assert manifest.exists()

    method_dir = run_dir / "toy_dataset" / "IdentityResampler"
    assert (method_dir / "experiment_results_toy_dataset_IdentityResampler.json").exists()
    assert (method_dir / "results_summary_toy_dataset_IdentityResampler.csv").exists()

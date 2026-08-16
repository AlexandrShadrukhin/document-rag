from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.lab.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    changed_reindex_fields,
    requires_reindex,
)


def settings_for_test(tmp_path: Path) -> Settings:
    return Settings(
        qdrant_path=tmp_path / "baseline/qdrant",
        corpus_path=tmp_path / "baseline/corpus.jsonl",
        manifest_path=tmp_path / "baseline/manifest.json",
        wiki_dump_path=tmp_path / "source/wiki.bz2",
        wiki_corpus_path=tmp_path / "corpus",
        benchmarks_path=tmp_path / "benchmarks",
        experiments_path=tmp_path / "experiments",
    )


def test_reindex_classification_and_baseline_protection(tmp_path: Path) -> None:
    settings = settings_for_test(tmp_path)
    index_experiment = ExperimentConfig(
        name="chunk-300", corpus=tmp_path / "corpus", index_config={"chunk_size": 300}
    )
    query_experiment = ExperimentConfig(
        name="rerank-30",
        corpus=tmp_path / "corpus",
        query_config={"rerank_candidates": 30},
    )
    assert requires_reindex(settings, index_experiment) is True
    assert changed_reindex_fields(settings, index_experiment) == {"chunk_size"}
    assert requires_reindex(settings, query_experiment) is False

    isolated, root, reindex = ExperimentRunner(settings).prepare(index_experiment)
    assert reindex is True
    assert str(isolated.qdrant_path).startswith(str(root))
    assert isolated.qdrant_path != settings.qdrant_path

    query_settings, _, query_reindex = ExperimentRunner(settings).prepare(query_experiment)
    assert query_reindex is False
    assert query_settings.qdrant_path == settings.qdrant_path


def test_settings_resolve_relative_runtime_paths() -> None:
    settings = Settings(qdrant_path=Path("data/portable-index/qdrant"))
    assert settings.qdrant_path.is_absolute()
    assert settings.qdrant_path.parts[-2:] == ("portable-index", "qdrant")

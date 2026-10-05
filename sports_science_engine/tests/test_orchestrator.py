from datetime import date
from pathlib import Path

from ssve.orchestrator import Orchestrator, load_config


def test_manifest_lists_four_agents():
    cfg = load_config(Path(__file__).parents[1] / "config" / "manifest.yaml")
    assert [a["module"] for a in cfg["agents"]] == ["ssve.ingestion", "ssve.rag",
                                                   "ssve.synthesis", "ssve.vision"]


def test_weekly_run_end_to_end_offline(tmp_path, corpus, pipeline, monkeypatch):
    cfg = {
        "ingestion": {},
        "synthesis": {"themes": ["workload injury"], "output_dir": str(tmp_path),
                      "competitors": [{"name": "WADA"}],
                      "email": {"to": ["team@example.org"], "dry_run": True}},
    }
    orch = Orchestrator(cfg, pipeline=pipeline)
    monkeypatch.setattr(orch, "ingest", lambda report: corpus)
    report = orch.run_weekly(week_ending=date(2026, 10, 5))
    names = sorted(p.name for p in report.outputs)
    assert names == ["Sports_Science_Weekly_Digest___05_Oct_2026.eml",
                     "digest_2026-10-05.md", "memo_wada_2026-10-05.md"]
    assert report.indexed == len(corpus)

import json
from pathlib import Path

from stresstest.cli import run_demo, main

FIXTURE = "fixtures/tutoring_claim.json"


def test_run_demo_writes_files(tmp_path):
    from stresstest.claim import load_claim

    claim = load_claim(FIXTURE)
    report = run_demo(tmp_path, claim, n_replicates=300)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "evidence_report.html").exists()
    assert (tmp_path / "evidence_unmeasured_confounding.md").exists()
    assert (tmp_path / "evidence_differential_attrition.html").exists()
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["offline"] is True
    assert data["proposer"] == "deterministic_fallback"
    assert len(data["results"]) == 2


def test_main_demo_returns_zero(tmp_path, capsys):
    rc = main(["demo", "--out", str(tmp_path), "--fixture", FIXTURE, "--replicates", "200"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RESEARCH CLAIM STRESS-TEST" in out
    assert "does NOT prove" in out


def test_main_check_claim(tmp_path, capsys):
    rc = main(
        ["check-claim", "--text",
         "The program improved scores by 5 points (p = 0.02, N = 200 per group). "
         "Observational cohort. Baseline 60, SD 12.",
         "--out", str(tmp_path), "--replicates", "200"]
    )
    assert rc == 0
    assert (tmp_path / "report.json").exists()

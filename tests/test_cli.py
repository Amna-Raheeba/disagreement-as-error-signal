import json
import os

from disagreement_signal.cli import main


def test_cli_demo(capsys):
    code = main(["demo", "--seed", "7"])
    out = capsys.readouterr().out
    assert code == 0
    assert "disagreement" in out


def test_cli_no_command(capsys):
    code = main([])
    assert code == 2


def test_cli_evaluate_writes_files(tmp_path):
    out_dir = str(tmp_path / "results")
    code = main(["evaluate", "--seed", "7", "--out", out_dir, "--limit", "15"])
    assert code == 0
    for name in ("evaluation.json", "evaluation.csv", "evaluation_summary.md", "HEADLINE.md"):
        assert os.path.exists(os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "evaluation.json")) as fh:
        payload = json.load(fh)
    assert "pooled" in payload
